import csv, numpy as np, torch, time
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
W=B+'work/wav16k/'
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
fe=AutoFeatureExtractor.from_pretrained('facebook/mms-lid-4017')
m=Wav2Vec2ForSequenceClassification.from_pretrained('facebook/mms-lid-4017').half().to('mps').eval()
LAY=[12,24,36,48]
H=np.zeros((len(rows),len(LAY)+1,1280),dtype=np.float32); LG=np.zeros((len(rows),4017),dtype=np.float32)
t0=time.time()
with torch.no_grad():
    for i,r in enumerate(rows):
        x=np.load(W+r['fn'].replace('.mp3','.npy'))[:160000]
        if len(x)<8000: x=np.pad(x,(0,8000-len(x)))
        iv=fe(x,sampling_rate=16000,return_tensors='pt')['input_values'].to('mps',dtype=torch.float16)
        o=m(iv,output_hidden_states=True)
        LG[i]=o.logits[0].float().cpu().numpy()
        for j,l in enumerate(LAY): H[i,j]=o.hidden_states[l][0].float().mean(0).cpu().numpy()
        H[i,-1,:1024]=m.projector(o.hidden_states[-1])[0].float().mean(0).cpu().numpy()
        if i%100==0:
            print(i,round(time.time()-t0,1),flush=True)
            if i%400==0: np.save(B+'work/lid_logits.partial.npy',LG)
np.save(B+'work/lid_logits.npy',LG); np.save(B+'work/lid_hidden.npy',H); print('saved',round(time.time()-t0,1), 'nonfinite',int((~np.isfinite(LG)).sum()),int((~np.isfinite(H)).sum()))
