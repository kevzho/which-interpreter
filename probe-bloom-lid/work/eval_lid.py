import numpy as np, collections, json, warnings
warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/work/evaluate.py').read().split("res={}")[0])
from transformers import AutoConfig
id2l=AutoConfig.from_pretrained('facebook/mms-lid-4017').id2label; l2id={v:int(k) for k,v in id2l.items()}
LG=np.load(B+'work/lid_logits.npy'); H=np.load(B+'work/lid_hidden.npy')
clean=np.where(inset & ~noisy)[0]; y=np.array([LI[l] for l in lang[clean]])
ids=[l2id[l] for l in LANGS]
# zero-shot restricted
pz=LG[clean][:,ids].argmax(1)
rec=recall_score(y,pz,average=None)
print('ZERO-SHOT lid-4017 restricted-6 on all clean in-set n=%d: acc %.3f macroF1 %.3f'%(len(y),(pz==y).mean(),mf1(y,pz)),'recall',dict(zip(LANGS,np.round(rec,3))))
print('   misroute(Mayan->spa) %.3f false-Mayan %.3f binary acc %.3f'%((pz[y!=0]==0).mean(),(pz[y==0]!=0).mean(),((pz==0)==(y==0)).mean()))
cm=confusion_matrix(y,pz); [print('  ',LANGS[i].ljust(4),cm[i]) for i in range(6)]
top1=LG.argmax(1)
for L in LANGS+['tzj']:
    mk=np.where(lang==L)[0] if L=='tzj' else clean[lang[clean]==L]
    c=collections.Counter(id2l[int(t)] for t in top1[mk])
    print(f'  unrestricted {L}: top1==true {np.mean([id2l[int(t)]==L for t in top1[mk]]):.3f}  top-dist {c.most_common(5)}')
# zero-shot open-set: tzj top1 & restricted
ti=np.where(tzj)[0]
print('  tzj restricted-6 dist',dict(collections.Counter(LANGS[q] for q in LG[ti][:,ids].argmax(1))))
# per-book zero-shot on the problematic books
for b in ["mam|Pujb il Yol Mam","mam|Tipumal xnaq'tzb'il1 si","quc|Ixim, aj, jal","quc|Kik'aslemal ri ak'alab' pa tijob'al"]:
    mk=book[clean]==b; print('  zero-shot per-book',b,mk.sum(),'restricted acc',round((pz[mk]==y[mk]).mean(),2))
# noisy books zero-shot
for b in sorted(set(book[noisy])):
    mk=book==b; print('  zero-shot NOISY',b,mk.sum(),'restricted',dict(collections.Counter(LANGS[q] for q in LG[mk][:,ids].argmax(1))),'top1',collections.Counter(id2l[int(t)] for t in top1[mk]).most_common(3))
# LR on lid features
names=['L12','L24','L36','L48','proj']
HL=[H[:,j,:] for j in range(4)]+[H[:,4,:1024]]
ls=LG-np.log(np.exp(LG-LG.max(1,keepdims=True)).sum(1,keepdims=True))-LG.max(1,keepdims=True)
for grouped in [False,True]:
    yy,P,ch=run(HL,clean,grouped,select=True); p=P.argmax(1)
    tag='grp' if grouped else 'rnd'; np.save(B+f'work/P_lidLR_{tag}.npy',P)
    print(f'\nLR on lid-4017 hidden ({tag}) layers chosen',[names[c] for c in ch]); report(f'lidLR {tag}',yy,P,clean)
    if grouped:
        Pg=P; chg=ch
        for b in ["mam|Pujb il Yol Mam","mam|Tipumal xnaq'tzb'il1 si","quc|Ixim, aj, jal","quc|Kik'aslemal ri ak'alab' pa tijob'al","quc|Choltzij K iche (1) (2)"]:
            mk=book[clean]==b; print('  lidLR per-book',b,mk.sum(),'acc',round((p[mk]==y[mk]).mean(),2))
print('per-layer lidLR grouped macroF1',{names[j]:round(mf1(*(lambda r:(r[0],r[1].argmax(1)))(run([HL[j]],clean,True,select=False))),3) for j in range(5)})
yy,P,_=run([ls[:,ids]],clean,True,select=False); print('LR on 6 restricted log-probs grouped macroF1',round(mf1(yy,P.argmax(1)),3))
# open-set with lidLR (layer = most chosen)
best=collections.Counter(chg).most_common(1)[0][0]
m=clf().fit(HL[best][clean],y); Pt=m.predict_proba(HL[best][ti])
print('lidLR open-set layer',names[best],'tzj pred dist',dict(collections.Counter(LANGS[q] for q in Pt.argmax(1))))
lab=np.r_[np.ones(len(clean)),np.zeros(len(ti))]
print('   AUROC maxprob (grouped OOF in-set vs tzj)',round(roc_auc_score(lab,np.r_[Pg.max(1),Pt.max(1)]),3))
# zero-shot "other language" score: 1 - P(6 langs) under full softmax
p6=np.exp(ls[:,ids]).sum(1)
print('   zero-shot AUROC using P(any of 6 langs) under 4017-way softmax:',round(roc_auc_score(lab,np.r_[p6[clean],p6[ti]]),3))
print('   zero-shot AUROC using max restricted-6 prob (renormalised):',round(roc_auc_score(lab,np.r_[np.exp(ls[clean][:,ids]-np.log(p6[clean])[:,None]).max(1),np.exp(ls[ti][:,ids]-np.log(p6[ti])[:,None]).max(1)]),3))
for t in [0.5,0.3,0.1]:
    print(f'   zero-shot reject if P(6)<{t}: in-set kept {np.mean(p6[clean]>=t):.2f} (restricted acc kept {np.mean((pz==y)[p6[clean]>=t]):.3f}); tzj rejected {np.mean(p6[ti]<t):.2f}')
