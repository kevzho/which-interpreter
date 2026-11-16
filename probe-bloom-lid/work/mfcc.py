import csv, numpy as np, torch, torchaudio, time
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
W=B+'work/wav16k/'
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
mf=torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=40,melkwargs=dict(n_fft=400,hop_length=160,n_mels=64))
F=[]
t0=time.time()
for r in rows:
    x=torch.from_numpy(np.load(W+r['fn'].replace('.mp3','.npy'))[:160000].astype(np.float32))
    if len(x)<8000: x=torch.nn.functional.pad(x,(0,8000-len(x)))
    m=mf(x)  # 40 x T
    d=torchaudio.functional.compute_deltas(m)
    z=torch.cat([m,d],0)
    F.append(torch.cat([z.mean(1),z.std(1)]).numpy())
F=np.stack(F).astype(np.float32); np.save(B+'work/mfcc_feats.npy',F); print(F.shape, round(time.time()-t0,1))
