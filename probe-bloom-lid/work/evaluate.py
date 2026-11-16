import csv, json, numpy as np, collections, time, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, recall_score, roc_auc_score
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
dur=np.array([0.0]*len(rows))
dd={r['fn']:float(r['dur']) for r in csv.DictReader(open(B+'work/durations.csv'))}
dur=np.array([dd[r['fn']] for r in rows])
LANGS=['spa','mam','quc','kek','cak','kjb']; LI={l:i for i,l in enumerate(LANGS)}
lang=np.array([r['lang'] for r in rows]); book=np.array([r['lang']+'|'+r['book'] for r in rows])
NOISY={("kek","Stxolil q anej Q anjob al"),("kek","Stxolil q'anej Q'anjob al"),("spa","Tipumal xnaq'tzb'il1 si"),("spa","Li kaqiyajel"),("spa","Li kaqi yajel"),("spa","Na leb")}
noisy=np.array([(r['lang'],r['book']) in NOISY for r in rows])
inset=np.isin(lang,LANGS); tzj=lang=='tzj'
MMS=np.load(B+'work/mms300m_feats.npy'); LAYERS=[0,3,6,9,12,15,18,21,24]
MF=np.load(B+'work/mfcc_feats.npy')
C=float(sys.argv[1]) if len(sys.argv)>1 else 0.1
def clf(): return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))
def mf1(y,p): return f1_score(y,p,average='macro')
def run(X_by_layer, idx, grouped, select=True, seed=0):
    y=np.array([LI[l] for l in lang[idx]]); g=book[idx]
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if grouped else StratifiedKFold(5,shuffle=True,random_state=seed)
    P=np.zeros((len(idx),6)); chosen=[]
    for tr,te in outer.split(idx,y,g):
        if select and len(X_by_layer)>1:
            inner=StratifiedGroupKFold(3,shuffle=True,random_state=1) if grouped else StratifiedKFold(3,shuffle=True,random_state=1)
            sc=[]
            for li,X in enumerate(X_by_layer):
                Xi=X[idx][tr]; s=[]
                for a,b in inner.split(Xi,y[tr],g[tr]):
                    m=clf().fit(Xi[a],y[tr][a]); s.append(mf1(y[tr][b],m.predict(Xi[b])))
                sc.append(np.mean(s))
            best=int(np.argmax(sc))
        else: best=0
        chosen.append(best)
        X=X_by_layer[best][idx]
        m=clf().fit(X[tr],y[tr]); P[te]=m.predict_proba(X[te])
    return y,P,chosen
def report(name,y,P,idx):
    p=P.argmax(1)
    rec=recall_score(y,p,average=None,labels=range(6))
    out={'macroF1':round(mf1(y,p),3),'acc':round((p==y).mean(),3),'recall':{LANGS[i]:round(rec[i],3) for i in range(6)}}
    cm=confusion_matrix(y,p,labels=range(6))
    maya=y!=0; spa=y==0
    out['binary']={'spa_misroute(Mayan->spa)':round((p[maya]==0).mean(),3),'spa_false_mayan':round((p[spa]!=0).mean(),3),
                   'binary_acc':round(((p==0)==(y==0)).mean(),3),'binary_bal_acc':round(0.5*((p[maya]!=0).mean()+(p[spa]==0).mean()),3)}
    print(f'### {name}: n={len(y)}',json.dumps(out))
    print('   CM rows=true cols=pred',LANGS); [print('  ',LANGS[i].ljust(4),cm[i]) for i in range(6)]
    return out
res={}
clean=np.where(inset & ~noisy)[0]; asl=np.where(inset)[0]
print('n clean per lang',{l:int((lang[clean]==l).sum()) for l in LANGS}, 'books',{l:len(set(book[clean][lang[clean]==l])) for l in LANGS})
print('n as-labeled per lang',{l:int((lang[asl]==l).sum()) for l in LANGS})
y=np.array([LI[l] for l in lang[clean]])
print('always-spa: acc',round((y==0).mean(),3),'macroF1',round(mf1(y,np.zeros_like(y)),3),'misroute',1.0,'false-mayan',0.0,'binary acc',round((y==0).mean(),3))
MMSL=[MMS[:,j,:] for j in range(len(LAYERS))]
t0=time.time()
# MFCC
for grouped in [False,True]:
    yy,P,_=run([MF],clean,grouped,select=False); res[f'mfcc_{"grp" if grouped else "rnd"}']=report(f'MFCC LR {"book-grouped" if grouped else "random"}',yy,P,clean)
    np.save(B+f'work/P_mfcc_{"grp" if grouped else "rnd"}.npy',P)
# duration-only
yy,P,_=run([np.log(np.minimum(dur,10.0))[:,None]],clean,True,select=False); report('DURATION-only LR book-grouped',yy,P,clean)
# MMS
for grouped in [False,True]:
    yy,P,ch=run(MMSL,clean,grouped,select=True)
    tag="grp" if grouped else "rnd"
    print('chosen layers per fold',[LAYERS[c] for c in ch], round(time.time()-t0,1),'s')
    res[f'mms_{tag}']=report(f'MMS300m LR {"book-grouped" if grouped else "random"}',yy,P,clean)
    np.save(B+f'work/P_mms_{tag}.npy',P); np.save(B+f'work/chosen_{tag}.npy',np.array(ch))
# per-layer book-grouped (fixed layer, no selection) for reference
print('per-layer book-grouped macroF1:',{LAYERS[j]:round(mf1(*(lambda r:(r[0],r[1].argmax(1)))(run([MMSL[j]],clean,True,select=False))),3) for j in range(len(LAYERS))})
json.dump(res,open(B+f'work/res_C{C}.json','w'))
