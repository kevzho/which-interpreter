import numpy as np, collections, warnings
warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/work/evaluate.py').read().split("res={}")[0])
H=np.load(B+'work/lid_hidden.npy'); clean=np.where(inset & ~noisy)[0]; y=np.array([LI[l] for l in lang[clean]])
HL=[H[:,j,:] for j in range(4)]+[H[:,4,:1024]]
MMSL=[MMS[:,j,:] for j in range(len(LAYERS))]
for seed in [1,2]:
    yy,P,ch=run(HL,clean,True,select=True,seed=seed); p=P.argmax(1)
    print(f'lidLR grouped seed{seed} macroF1 {mf1(yy,p):.3f} recall',dict(zip(LANGS,np.round(recall_score(yy,p,average=None),2))),'misroute %.3f falseMayan %.3f'%((p[yy!=0]==0).mean(),(p[yy==0]!=0).mean()))
# concat lid L24 + mms L15
Xc=np.hstack([HL[1],MMSL[LAYERS.index(15)]])
yy,P,_=run([Xc],clean,True,select=False); p=P.argmax(1); print('concat lidL24+mmsL15 grouped macroF1 %.3f'%mf1(yy,p), dict(zip(LANGS,np.round(recall_score(yy,p,average=None),2))))
# dedicated binary spa-vs-Mayan LR, grouped
from sklearn.model_selection import StratifiedGroupKFold
for nm,X in [('mmsL15',MMSL[LAYERS.index(15)]),('lidL24',HL[1]),('mfcc',MF)]:
    yb=(y!=0).astype(int); g=book[clean]; pb=np.zeros(len(yb))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X[clean],y,g):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000,class_weight='balanced')).fit(X[clean][tr],yb[tr]); pb[te]=m.predict_proba(X[clean][te])[:,1]
    pr=(pb>0.5).astype(int)
    print(f'BINARY LR {nm} grouped: misroute(Mayan->spa) {(pr[yb==1]==0).mean():.3f} falseMayan {(pr[yb==0]==1).mean():.3f} acc {(pr==yb).mean():.3f} AUROC {roc_auc_score(yb,pb):.3f}')
# book-level bootstrap CI for macro-F1 of grouped OOF predictions
rng=np.random.default_rng(0); bk=book[clean]
for nm,f in [('mms',B+'work/P_mms_grp.npy'),('lidLR',B+'work/P_lidLR_grp.npy'),('mfcc',B+'work/P_mfcc_grp.npy')]:
    p=np.load(f).argmax(1); s=[]
    byL={L:sorted(set(bk[lang[clean]==L])) for L in LANGS}
    idx_by_book={b:np.where(bk==b)[0] for b in set(bk)}
    for _ in range(500):
        ii=np.concatenate([idx_by_book[b] for L in LANGS for b in rng.choice(byL[L],len(byL[L]))])
        s.append(mf1(y[ii],p[ii]))
    print(f'{nm} grouped macroF1 book-bootstrap 95% CI [{np.percentile(s,2.5):.3f}, {np.percentile(s,97.5):.3f}]')
