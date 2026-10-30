"""Explicit opt-in oracle using an installed AudioCraft runtime, not a mirrored DiT.

Run with that runtime's Python, --runtime DIR, --probe mf_train_parity_probe,
--out DIR, and optionally --gguf-path the repo venv's site-packages.
"""
import argparse
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--gguf-path', type=Path)
parser.add_argument('--probe', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', default='cpu', help='native probe backend; torch oracle stays CPU F32')
parser.add_argument('--weight-type', choices=('f32', 'f16'), default='f32')
parser.add_argument('--adapter-type', choices=('lora', 'dora-rows'), default='dora-rows')
args = parser.parse_args()
sys.path.insert(0, str(args.runtime))
if args.gguf_path:
    sys.path.append(str(args.gguf_path))
import gguf
from audiocraft.models.lm import LMModel
from torch.nn.utils import parametrize
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import ConditionFuser

torch.set_num_threads(1)
args.out.mkdir(parents=True, exist_ok=True)
if any(args.out.iterdir()):
    parser.error('--out must be an empty directory')

def write(path, tensors, metadata, arch):
    w = gguf.GGUFWriter(str(path), arch)
    for key, value in metadata.items():
        if isinstance(value, bool): w.add_bool(key, value)
        elif isinstance(value, int): w.add_uint32(key, value)
        elif isinstance(value, float): w.add_float32(key, value)
        else: w.add_string(key, value)
    for key, value in tensors.items():
        dtype = np.float16 if arch == 'ac-musicgen-lm' and value.ndim == 2 and args.weight_type == 'f16' else np.float32
        w.add_tensor(key, np.ascontiguousarray(value, dtype=dtype))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

class Delta(torch.nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.A = torch.nn.ParameterList([p[0] for p in pairs])
        self.B = torch.nn.ParameterList([p[1] for p in pairs])
        self.M = torch.nn.ParameterList([p[2] for p in pairs]) if args.adapter_type == 'dora-rows' else None
    def forward(self, weight):
        direction = weight + torch.cat([b @ a for a, b in zip(self.A, self.B)], dim=0)
        if self.M is None: return direction
        magnitude = torch.cat(list(self.M)).reshape(-1,1)
        return magnitude * direction / torch.sqrt(torch.sum(direction**2,dim=1,keepdim=True)+weight.shape[1]*1e-12)

def run(mode, zero_b):
    name = mode + ('-zeroB' if zero_b else '-nonzeroB')
    folder = args.out / name
    fixture, native = folder / 'fixture', folder / 'native'
    fixture.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(44)
    model=LMModel(DelayedPatternProvider(4,delays=[0,1,2,3]),None,
        ConditionFuser({'cross':['description'],'sum':[],'prepend':[],'input_interpolate':[]}),
        n_q=4,card=11,dim=16,num_heads=2,hidden_scale=2,norm_first=True,bias_proj=False,
        num_layers=2,dropout=0,bias_ff=False,bias_attn=False,custom=True,memory_efficient=False,
        cross_attention=True,positional_embedding='sin',causal=True,activation='gelu',device='cpu')
    with torch.no_grad():
        for key, p in model.named_parameters():
            if 'norm' in key: p.fill_(0 if key.endswith('bias') else 1)
            else: p.normal_(0, 0.15)
        if args.weight_type == 'f16':
            # Compare against exactly the quantized frozen weights, with F32 arithmetic.
            for p in list(model.parameters()):
                if p.ndim == 2: p.copy_(p.half().float())
    model.requires_grad_(False)
    state = model.state_dict()
    base = {}
    def take(dst, src): base[dst] = state[src].numpy().copy()
    for q in range(4):
        take(f'lm.emb.{q}.weight',f'emb.{q}.weight');take(f'lm.head.{q}.weight',f'linears.{q}.weight')
    for kind in ('weight','bias'):take('lm.out_norm.'+kind,'out_norm.'+kind)
    base['lm.cond_proj.weight']=np.eye(16,dtype=np.float32)
    base['lm.cond_proj.bias']=np.zeros(16,np.float32)
    params = {}
    for i, layer in enumerate(model.transformer.layers):
        dst, src = f'lm.{i}.', f'transformer.layers.{i}.'
        for norm in ('norm1', 'norm_cross', 'norm2'):
            for kind in ('weight', 'bias'): take(dst + norm + '.' + kind, src + norm + '.' + kind)
        take(dst + 'self.qkv.weight', src + 'self_attn.in_proj_weight')
        take(dst + 'self.out.weight', src + 'self_attn.out_proj.weight')
        cross = state[src + 'cross_attention.in_proj_weight'].numpy()
        base[dst + 'cross.q.weight'] = cross[:16].copy()
        base[dst + 'cross.kv.weight'] = cross[16:].copy()
        take(dst + 'cross.out.weight', src + 'cross_attention.out_proj.weight')
        for j, lin in [(0, 'linear1'), (2, 'linear2')]: take(dst + f'ff.{j}.weight', src + lin + '.weight')
        def pair(stem):
            shape = base[dst + stem + '.weight'].shape
            a = torch.nn.Parameter(torch.randn(2, shape[1]) * 0.02)
            b = torch.nn.Parameter(torch.zeros(shape[0], 2) if zero_b else torch.randn(shape[0], 2) * 0.03)
            params[dst + stem + '.weight.lora_A'] = a
            params[dst + stem + '.weight.lora_B'] = b
            if args.adapter_type == 'dora-rows':
                w = torch.from_numpy(base[dst + stem + '.weight'])
                m = torch.nn.Parameter(torch.sqrt(torch.sum(w*w,dim=1)+shape[1]*1e-12))
                if not zero_b:
                    with torch.no_grad(): m.mul_(1.05)
                params[dst + stem + '.weight.magnitude'] = m
                return a,b,m
            return a, b
        parametrize.register_parametrization(layer.self_attn, 'in_proj_weight', Delta([pair('self.qkv')]))
        parametrize.register_parametrization(layer.self_attn.out_proj, 'weight', Delta([pair('self.out')]))
        parametrize.register_parametrization(layer.cross_attention, 'in_proj_weight', Delta([pair('cross.q'), pair('cross.kv')]))
        parametrize.register_parametrization(layer.cross_attention.out_proj, 'weight', Delta([pair('cross.out')]))
    config={'general.name':'MusicGen PyTorch oracle','ac.architecture':'audiocraft','ac.format_version':1,
        'ac.lm.family':'musicgen','ac.lm.dim':16,'ac.lm.layers':2,'ac.lm.heads':2,
        'ac.lm.head_dim':8,'ac.lm.ff_dim':32,'ac.lm.cond_dim':16,'ac.lm.n_q':4,'ac.lm.card':11,
        'ac.lm.special_token_id':11,'ac.lm.pos_denominator':7,'ac.lm.max_period':10000.,
        'ac.lm.norm_eps':1e-5,'ac.lm.cfg_coef':3.,'ac.lm.sample_rate':32000,'ac.lm.max_duration':30.}
    write(fixture / 'base.gguf', base, config, 'ac-musicgen-lm')
    write(fixture / 'adapter.gguf', {k: p.detach().numpy() for k, p in params.items()},
          {'ac.lora.version': 1, 'ac.lora.rank': 2, 'ac.lora.alpha': 2., 'ac.lora.type': args.adapter_type, 'ac.lora.base_name': config['general.name']},
          'ac-musicgen-lora')
    codes=torch.randint(0,11,(1,4,7))
    context=torch.randn(1,3,16)
    if mode=='null':context.zero_()
    if mode=='padded':context[:,1:]=0
    codes.numpy().astype('<i4').tofile(fixture/'codes.i32')
    context.numpy().astype('<f4').tofile(fixture/'context.f32')
    result=subprocess.run([str(args.probe),str(fixture),str(native),mode,args.device],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    native_losses=np.loadtxt(native/'losses.txt')
    optimizer = torch.optim.Adam(list(params.values()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    details = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction=model.compute_predictions(codes,[],{'description':(context,torch.ones(1,3))},keep_only_valid_steps=False)
        mask=prediction.mask.clone()
        if mode!='noprefix':mask[:,:,:3]=False
        loss=torch.nn.functional.cross_entropy(prediction.logits[mask],codes[mask])
        loss.backward()
        assert abs(loss.item() - native_losses[step]) < 2e-5, (name, step, loss.item(), native_losses[step])
        torch_grads, native_grads = [], []
        for key, p in params.items():
            g = p.grad.detach().numpy() if p.grad is not None else np.zeros_like(p.detach().numpy())
            ng = np.fromfile(native / (f'step{step}-' + key + '.grad.f32'), dtype='<f4').reshape(g.shape)
            assert np.allclose(g, ng, rtol=0.003, atol=2e-5), (name, step, key, float(np.max(np.abs(g-ng))))
            if mode == 'null' and '.cross.' in key:
                assert np.count_nonzero(g) == 0 and np.count_nonzero(ng) == 0, 'masked cross-attention received gradient'
            if zero_b and step == 0 and key.endswith('lora_A'):
                assert np.count_nonzero(g) == 0 and np.count_nonzero(ng) == 0, 'zero-B A gradient is nonzero'
            torch_grads.append(g.ravel()); native_grads.append(ng.ravel())
        g, ng = np.concatenate(torch_grads), np.concatenate(native_grads)
        relative = float(np.linalg.norm(g-ng) / max(np.linalg.norm(g), 1e-12))
        assert relative < 0.003, (name, step, relative)
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
        optimizer.step()
        updated = gguf.GGUFReader(str(native / f'step{step+1}.gguf'))
        error = max(float(np.max(np.abs(t.data - params[t.name].detach().numpy()))) for t in updated.tensors)
        assert error < 3e-5, (name, step, 'Adam', error)
        details.append({'loss_torch': loss.item(), 'loss_native': float(native_losses[step]),
                        'gradient_relative_l2': relative, 'gradient_norm': float(np.linalg.norm(g)),
                        'adam_max_abs_difference': error})
    print(f'PASS {name}: max gradient relative L2 {max(d["gradient_relative_l2"] for d in details):.3g}', flush=True)
    return details

results = {mode + ('-zeroB' if zero else '-nonzeroB'): run(mode, zero)
           for mode in ('conditional', 'padded', 'null', 'noprefix') for zero in (True, False)}
(args.out / 'results.json').write_text(json.dumps({'runtime': str(args.runtime), 'native_device': args.device,
                                                'base_weight_type': args.weight_type,
                                                'adapter_type': args.adapter_type,
                                                'cases': results}, indent=2), encoding='utf-8')
