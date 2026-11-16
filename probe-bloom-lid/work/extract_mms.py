import csv, os, numpy as np, torch, time, sys
from transformers import Wav2Vec2Model
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
W=B+'work/wav16k/'; OUT=B+'work/mms300m_feats.npy'
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
LAYERS=[0,3,6,9,12,15,18,21,24]
dev='mps' if torch.backends.mps.is_available() else 'cpu'
m=Wav2Vec2Model.from_pretrained('facebook/mms-300m').eval()
try:
    m=m.to(dev)
    with torch.no_grad(): m(torch.zeros(1,16000,device=dev),output_hidden_states=True)
except Exception as e:
    print('MPS failed, CPU fallback',e); dev='cpu'; m=m.to('cpu')
print('device',dev, 'nlayers', m.config.num_hidden_layers, flush=True)
feats=np.zeros((len(rows),len(LAYERS),1024),dtype=np.float32)
t0=time.time()
with torch.no_grad():
    for i,r in enumerate(rows):
        x=np.load(W+r['fn'].replace('.mp3','.npy'))[:160000].astype(np.float32)
        if len(x)<8000: x=np.pad(x,(0,8000-len(x)))
        x=(x-x.mean())/(x.std()+1e-7)
        hs=m(torch.from_numpy(x)[None].to(dev),output_hidden_states=True).hidden_states
        for j,l in enumerate(LAYERS): feats[i,j]=hs[l][0].mean(0).float().cpu().numpy()
        if i%100==0:
            print(i, round(time.time()-t0,1), flush=True)
            if i%300==0: np.save(OUT+".partial.npy",feats)
np.save(OUT,feats); print('saved',feats.shape,'secs',round(time.time()-t0,1))
