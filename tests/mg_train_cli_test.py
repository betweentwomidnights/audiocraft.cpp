import os
"""Native trainer regression: checkpoint invariance, track isolation, prompt sidecars."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
try:
    import gguf
    import numpy as np
except ImportError:
    print('SKIP: numpy/gguf unavailable')
    sys.exit(0)


def write(path,arch,tensors,metadata):
    w=gguf.GGUFWriter(str(path),arch)
    for k,v in metadata.items():
        if isinstance(v,int):w.add_uint32(k,v)
        elif isinstance(v,float):w.add_float32(k,v)
        else:w.add_string(k,v)
    for k,v in tensors.items():w.add_tensor(k,np.ascontiguousarray(v))
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()


def main(executable):
    rng=np.random.default_rng(77)
    shapes={'lm.cond_proj.weight':(8,6),'lm.cond_proj.bias':(8,),
            'lm.out_norm.weight':(8,),'lm.out_norm.bias':(8,)}
    for q in range(4):shapes[f'lm.emb.{q}.weight']=(12,8);shapes[f'lm.head.{q}.weight']=(11,8)
    for i in range(2):
        for norm in ('norm1','norm_cross','norm2'):
            shapes[f'lm.{i}.{norm}.weight']=(8,);shapes[f'lm.{i}.{norm}.bias']=(8,)
        for stem,shape in [('self.qkv',(24,8)),('self.out',(8,8)),('cross.q',(8,8)),('cross.kv',(16,8)),('cross.out',(8,8)),('ff.0',(16,8)),('ff.2',(8,16))]:shapes[f'lm.{i}.{stem}.weight']=shape
    tensors={k:rng.normal(0,.15,shape).astype(np.float32) for k,shape in shapes.items()}
    for k,v in tensors.items():
        if 'norm' in k:v.fill(0 if k.endswith('bias') else 1)
    config={'general.name':'MusicGen CLI fixture','ac.architecture':'audiocraft','ac.format_version':1,
        'ac.lm.family':'musicgen','ac.lm.dim':8,'ac.lm.layers':2,'ac.lm.heads':2,'ac.lm.head_dim':4,
        'ac.lm.ff_dim':16,'ac.lm.cond_dim':6,'ac.lm.n_q':4,'ac.lm.card':11,'ac.lm.special_token_id':11,
        'ac.lm.pos_denominator':3,'ac.lm.max_period':10000.,'ac.lm.norm_eps':1e-5,'ac.lm.cfg_coef':3.,
        'ac.lm.sample_rate':32000,'ac.lm.max_duration':30.}
    with tempfile.TemporaryDirectory(prefix='ac-mg-cli-') as tmp:
        root=Path(tmp);write(root/'base.gguf','ac-musicgen-lm',tensors,config)
        base_bytes=(root/'base.gguf').read_bytes()
        for split in ('train','validation'):
            (root/split).mkdir()
            for i in range(2):write(root/split/f'{i}.gguf','ac-musicgen-train-cache',
                {'codes':rng.integers(0,11,(4,7),dtype=np.int32),'context':rng.normal(0,.1,(3,8)).astype(np.float32)},
                {'ac.cache.version':1,'ac.cache.base_name':config['general.name'],'ac.cache.source_id':f'{split}-{i}',
                 'ac.cache.prompt':'glitch hop, 92 bpm, D minor' if split=='train' else 'HELD OUT CAPTION'})
        common=[executable,'--lm',str(root/'base.gguf'),'--cache',str(root/'train'),'--validation',str(root/'validation'),
                '--frames','7','--prefix-frames','3','--steps','4','--rank','2','--first-layer','0','--device',os.environ.get('AC_TRAIN_TEST_DEVICE','cpu')]
        for name,extra in [('plain',['--checkpoint-every','0']),('checkpointed',['--checkpoint-every','2'])]:
            r=subprocess.run(common+['--out',str(root/f'{name}.gguf')]+extra,capture_output=True,text=True)
            assert r.returncode==0,r.stderr
            pool=json.loads((root/f'{name}.prompts.json').read_text(encoding='utf-8'))
            assert pool['dice']['instrumental']==['glitch hop, 92 bpm, D minor']
            assert pool['source']['files']==2
        assert (root/'checkpointed.step-2.prompts.json').exists()
        assert (root/'plain.gguf').read_bytes()==(root/'checkpointed.gguf').read_bytes()
        assert (root/'base.gguf').read_bytes()==base_bytes
        adapter=gguf.GGUFReader(str(root/'plain.gguf'))
        assert adapter.fields['ac.lora.type'].contents()=='dora-rows'
        assert len(adapter.tensors)==30
        del adapter
        # Fresh crops, accumulation, AdamW, warmup and FFN targets must survive exact resume.
        controls = ['--accumulation-steps','3','--warmup-steps','3','--lr-schedule','inverse',
                    '--weight-decay','0.01','--beta2','0.95','--random-crop','--train-ffn']
        cropped = common + ['--frames','5'] + controls
        for name, extra in [('controls', ['--checkpoint-every','2']),
                            ('resumed', ['--checkpoint-every','0','--resume',str(root/'controls.step-2.gguf.state')])]:
            run = subprocess.run(cropped + ['--out',str(root/(name+'.gguf'))] + extra, capture_output=True,text=True)
            assert run.returncode == 0, run.stderr
        assert (root/'controls.gguf').read_bytes() == (root/'resumed.gguf').read_bytes(), 'resume changed weights'
        assert (root/'controls.gguf.state').read_bytes() == (root/'resumed.gguf.state').read_bytes()
        scored = subprocess.run(cropped + ['--evaluate-adapter',str(root/'controls.gguf'),'--eval-unconditional'],capture_output=True,text=True)
        assert scored.returncode == 0, scored.stderr
        a = gguf.GGUFReader(str(root/'controls.gguf'))
        assert any('.ff.0.' in t.name for t in a.tensors), 'FFN targets missing'
        del a
        changed = subprocess.run(cropped + ['--out',str(root/'bad-resume.gguf'),'--resume',str(root/'controls.step-2.gguf.state'),'--lr','0.0002'],capture_output=True,text=True)
        assert changed.returncode != 0 and not (root/'bad-resume.gguf').exists()
        for flag,value in [('--accumulation-steps','0'),('--warmup-steps','-1'),('--beta2','1'),('--weight-decay','nan'),('--lr-schedule','bogus')]:
            bad = subprocess.run(common+['--out',str(root/'bad-controls.gguf'),flag,value],capture_output=True,text=True)
            assert bad.returncode != 0 and not (root/'bad-controls.gguf').exists()
        full=common.copy()
        del full[full.index('--prefix-frames'):full.index('--prefix-frames')+2]
        r=subprocess.run(full+['--out',str(root/'full.gguf'),'--checkpoint-every','0'],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        adapter=gguf.GGUFReader(str(root/'full.gguf'))
        assert adapter.fields['ac.lora.prefix_frames'].contents()==0
        assert adapter.fields['ac.lora.training_objective'].contents()=='generation-cross-entropy'
        del adapter
        r=subprocess.run(full+['--out',str(root/'lora.gguf'),'--adapter-type','lora','--checkpoint-every','0'],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        adapter=gguf.GGUFReader(str(root/'lora.gguf'))
        assert adapter.fields['ac.lora.type'].contents()=='lora' and len(adapter.tensors)==20
        del adapter
        r=subprocess.run(full+['--evaluate-adapter',str(root/'full.gguf'),'--eval-samples','0'],capture_output=True,text=True)
        assert r.returncode==0 and '[eval]' in r.stderr,r.stderr
        for extra in (['--prefix-frames','7'],['--adapter-type','invalid'],['--cfg-dropout','nan'],['--lr','inf'],['--alpha','-2'],['--first-layer','-2']):
            r=subprocess.run(common+['--out',str(root/'invalid.gguf')]+extra,capture_output=True,text=True)
            assert r.returncode!=0 and not (root/'invalid.gguf').exists()
        r=subprocess.run(common+['--out',str(root/'overlap.gguf'),'--validation',str(root/'train')],capture_output=True,text=True)
        assert r.returncode!=0 and 'overlap' in r.stderr
        r=subprocess.run(common+['--out',str(root/'plain.gguf')],capture_output=True,text=True)
        assert r.returncode!=0
    print('PASS MusicGen trainer and training-only prompt pools')


if __name__=='__main__':main(sys.argv[1])
