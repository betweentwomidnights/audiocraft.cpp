#!/usr/bin/env python3
"""Cache native EnCodec codes and projected T5 states for MusicGen adapters.

Caption text is data. Split whole tracks before selecting windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import wave
import gguf
import numpy as np
from prepare_mf_lora import read_pcm
from train_cache_windows import window_starts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','out','bin','lm','codec','t5'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache-seconds',type=float,default=0,help='encode longer windows (up to 30s) for fresh trainer crops; 0 follows --seconds')
    p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--windows-per-track',type=int,default=4)
    p.add_argument('--max-tracks',type=int,default=0)
    p.add_argument('--validation-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--device',default='cpu')
    p.add_argument("--cover-track",action="store_true",help="tile the whole source with cache windows")
    a=p.parse_args()
    a.seconds = a.cache_seconds or a.seconds
    if not 0 < a.seconds <= 30: p.error("cache seconds must be in (0,30]")
    frames=round(a.seconds*50)
    if not 0<a.seconds<=30 or abs(frames/50-a.seconds)>1e-6 or a.windows_per_track<=0 or a.max_tracks<0 or not 0<=a.validation_fraction<1:
        p.error('invalid cache settings')
    tracks=sorted(a.dataset.glob('*.wav'))
    if not tracks:p.error('no WAV tracks')
    random.Random(a.seed).shuffle(tracks)
    if a.max_tracks:tracks=tracks[:a.max_tracks]
    nv=round(len(tracks)*a.validation_fraction)
    if a.validation_fraction and len(tracks)>1:nv=min(len(tracks)-1,max(1,nv))
    validation=set(tracks[:nv])
    reader=gguf.GGUFReader(str(a.lm))
    base_name=str(reader.fields['general.name'].contents())
    tensors={t.name:t for t in reader.tensors}
    projection=tensors['lm.cond_proj.weight'].data.astype(np.float32)
    bias=tensors['lm.cond_proj.bias'].data.astype(np.float32)
    width=projection.shape[1];dim=projection.shape[0]
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.iterdir()):p.error('output directory must be empty')
    for split in ('train','validation'):(a.out/split).mkdir()
    suffix='.exe' if (a.bin/'ac-encodec.exe').exists() else ''
    manifest={'version':1,'frames':frames,'seed':a.seed,'base_name':base_name,'lm':str(a.lm.resolve()),'codec':str(a.codec.resolve()),'t5':str(a.t5.resolve()),'windows':[]}
    captions=[]
    with tempfile.TemporaryDirectory(prefix='ac-mg-lora-') as tmp:
        tmp=Path(tmp)
        for track in tracks:
            caption=track.with_suffix('.txt').read_text(encoding='utf-8-sig').strip()
            split='validation' if track in validation else 'train'
            if split=='train' and caption and caption not in captions:captions.append(caption)
            with wave.open(str(track),'rb') as w:rate,length=w.getframerate(),w.getnframes()
            window=round(a.seconds*rate)
            if length<window:raise ValueError(f'track shorter than window: {track}')
            (tmp / "caption.txt").write_text(caption, encoding="utf-8")
            subprocess.run([str(a.bin/('ac-textenc'+suffix)),'--t5',str(a.t5),'--prompt-file',str(tmp/'caption.txt'),'--out',str(tmp/'hidden.f32'),'--device',a.device],check=True)
            hidden=np.fromfile(tmp/'hidden.f32',dtype='<f4').reshape(-1,width)
            context=hidden@projection.T+bias
            if not caption:context.fill(0) # Null after the biased output projection.
            source_id=hashlib.sha256(track.name.encode('utf-8')).hexdigest()
            for i,start in enumerate(window_starts(length,window,a.windows_per_track,a.cover_track)):
                audio,_=read_pcm(track,int(start),window)
                with wave.open(str(tmp/'window.wav'),'wb') as w:
                    w.setnchannels(audio.shape[1]);w.setsampwidth(2);w.setframerate(rate)
                    w.writeframes((np.clip(audio,-1,1-1/32768)*32768).astype('<i2').tobytes())
                subprocess.run([str(a.bin/('ac-encodec'+suffix)),'--codec',str(a.codec),'--encode',str(tmp/'window.wav'),'--seconds',str(a.seconds),'--out-codes',str(tmp/'codes.i32'),'--device',a.device],check=True)
                codes=np.fromfile(tmp/'codes.i32',dtype='<i4')
                if codes.size!=4*frames or not np.isfinite(context).all():raise ValueError('invalid codec/context shape')
                name=source_id[:16]+f'-{i:03d}.gguf'
                w=gguf.GGUFWriter(str(a.out/split/name),'ac-musicgen-train-cache')
                w.add_uint32('ac.cache.version',1);w.add_string('ac.cache.base_name',base_name)
                w.add_string('ac.cache.source_id',source_id);w.add_string('ac.cache.prompt',caption)
                w.add_string('ac.cache.caption_transport','utf8-file')
                w.add_tensor('codes',codes.reshape(4,frames));w.add_tensor('context',np.ascontiguousarray(context,dtype=np.float32))
                w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                manifest['windows'].append({'cache':f'{split}/{name}','source':str(track.resolve()),'caption':caption,'start_seconds':start/rate})
                print(f'{split}: {track.name} @ {start/rate:.2f}s',flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (a.out/'prompts.json').write_text(json.dumps({'version':1,'source':{'lora':'training-cache','files':len(tracks)-nv,'unique_prompts':len(captions)},'dice':{'instrumental':captions}},indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
