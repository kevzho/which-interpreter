import csv, json, numpy as np, torch, time, collections, random
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
W=B+'work/wav16k/'
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
NOISY={("kek","Stxolil q anej Q anjob al"),("kek","Stxolil q'anej Q'anjob al"),("spa","Tipumal xnaq'tzb'il1 si"),("spa","Li kaqiyajel"),("spa","Li kaqi yajel"),("spa","Na leb")}
LANGS=['spa','mam','quc','kek','cak','kjb']
random.seed(1)
sel=[]
for L in LANGS+['tzj']:
    c=[r for r in rows if r['lang']==L and (r['lang'],r['book']) not in NOISY]
    sel+=random.sample(c,min(30,len(c)))
fe=AutoFeatureExtractor.from_pretrained('facebook/mms-lid-4017')
m=Wav2Vec2ForSequenceClassification.from_pretrained('facebook/mms-lid-4017')
id2l=m.config.id2label; l2id={v:int(k) for k,v in id2l.items()}
print('num labels',len(id2l),'ids:',{l:l2id.get(l) for l in LANGS+['tzj','eng','por','ixl','acr','tzh','tzo','yua','usp','poh','jac','ttc','mop']})
dev='mps'
try:
    m=m.half().to(dev).eval()
    with torch.no_grad(): m(torch.zeros(1,16000,dtype=torch.float16,device=dev))
except Exception as e:
    print('MPS fp16 failed',e); dev='cpu'; m=m.float().to('cpu').eval()
print('device',dev,flush=True)
ids=[l2id[l] for l in LANGS]
out=[]; t0=time.time()
with torch.no_grad():
    for i,r in enumerate(sel):
        x=np.load(W+r['fn'].replace('.mp3','.npy'))[:160000]
        inp=fe(x,sampling_rate=16000,return_tensors='pt')
        iv=inp['input_values'].to(dev, dtype=next(m.parameters()).dtype)
        lg=m(iv).logits[0].float().cpu()
        if not torch.isfinite(lg).all(): print('nonfinite logits',r['fn'])
        top=id2l[int(lg.argmax())]
        top5=[id2l[int(j)] for j in lg.topk(5).indices]
        r6=LANGS[int(lg[ids].argmax())]
        pr=torch.softmax(lg,0); 
        out.append(dict(lang=r['lang'],book=r['book'],top1=top,top5=top5,restricted=r6,p6=float(pr[ids].sum())))
        if i%50==0: print(i,round(time.time()-t0,1),flush=True)
json.dump(out,open(B+'work/lid_out.json','w'))
for L in LANGS+['tzj']:
    o=[x for x in out if x['lang']==L]
    acc=np.mean([x['restricted']==L for x in o]) if L!='tzj' else float('nan')
    print(f'{L}: n={len(o)} restricted6_acc={acc:.3f} restricted_dist={dict(collections.Counter(x["restricted"] for x in o))}')
    print(f'     unrestricted top1={collections.Counter(x["top1"] for x in o).most_common(6)}  top1==true {np.mean([x["top1"]==L for x in o]):.3f} true-in-top5 {np.mean([L in x["top5"] for x in o]):.3f} mean P(6 langs)={np.mean([x["p6"] for x in o]):.3f}')
o=[x for x in out if x['lang'] in LANGS]
y=[x['lang'] for x in o]; p=[x['restricted'] for x in o]
from sklearn.metrics import f1_score
print('restricted6 overall acc',round(np.mean([a==b for a,b in zip(y,p)]),3),'macroF1',round(f1_score(y,p,average='macro'),3))
print('binary: misroute',round(np.mean([b=='spa' for a,b in zip(y,p) if a!='spa']),3),'false-mayan',round(np.mean([b!='spa' for a,b in zip(y,p) if a=='spa']),3))
print('secs',round(time.time()-t0,1))
