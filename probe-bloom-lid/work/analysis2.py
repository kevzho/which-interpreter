import csv, json, numpy as np, collections, warnings
warnings.filterwarnings('ignore')
exec(open('/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/work/evaluate.py').read().split("res={}")[0])
clean=np.where(inset & ~noisy)[0]; asl=np.where(inset)[0]
MMSL=[MMS[:,j,:] for j in range(len(LAYERS))]
L15=LAYERS.index(15); L6=LAYERS.index(6)
y=np.array([LI[l] for l in lang[clean]])
# 1) seed variance of grouped split (with inner layer selection) and C sensitivity at layer 15
for seed in [1,2]:
    yy,P,ch=run(MMSL,clean,True,select=True,seed=seed); p=P.argmax(1)
    print(f'MMS grouped seed{seed}: macroF1={mf1(yy,p):.3f} layers={[LAYERS[c] for c in ch]} recall',np.round(recall_score(yy,p,average=None),3), 'misroute',round((p[yy!=0]==0).mean(),3),'falseMayan',round((p[yy==0]!=0).mean(),3))
    yy,P,_=run([MF],clean,True,select=False,seed=seed); print(f'MFCC grouped seed{seed}: macroF1={mf1(yy,P.argmax(1)):.3f}')
    yy,P,_=run(MMSL,clean,False,select=True,seed=seed); print(f'MMS random seed{seed}: macroF1={mf1(yy,P.argmax(1)):.3f}')
for Cc in [0.01,1.0]:
    C=Cc
    yy,P,_=run([MMSL[L15]],clean,True,select=False); print(f'C={Cc} layer15 grouped macroF1={mf1(yy,P.argmax(1)):.3f}')
C=0.1
# 2) per-book accuracy under grouped CV (seed 0 predictions)
P=np.load(B+'work/P_mms_grp.npy'); p=P.argmax(1)
bk=book[clean]; acc_by_book=collections.defaultdict(list)
for b,yt,pp in zip(bk,y,p): acc_by_book[b].append(yt==pp)
print('\nper-book accuracy (MMS grouped), books with acc<0.6:')
errs=(p!=y).sum(); bad=[]
for b,v in sorted(acc_by_book.items(),key=lambda kv:np.mean(kv[1])):
    if np.mean(v)<0.6: bad.append(b); print(f'   {b[:55]:55s} n={len(v):3d} acc={np.mean(v):.2f}  preds={dict(collections.Counter(LANGS[q] for q,bb in zip(p,bk) if bb==b))}')
print('total errors',errs,'errors in books with acc<0.6:',sum((p!=y)[bk==b].sum() for b in bad),'n such books',len(bad),'of',len(acc_by_book))
for L in LANGS:
    accs=[np.mean(v) for b,v in acc_by_book.items() if b.startswith(L+'|')]
    print(f'  {L}: books={len(accs)} book-level acc median={np.median(accs):.2f} min={np.min(accs):.2f} share books >=0.8: {np.mean(np.array(accs)>=0.8):.2f}')
# same for MFCC
Pm=np.load(B+'work/P_mfcc_grp.npy'); pm=Pm.argmax(1)
print('error overlap: MMS-wrong clips', (p!=y).sum(),' MFCC-wrong',(pm!=y).sum(),' both wrong',((p!=y)&(pm!=y)).sum())
# correlation of per-book acc MMS vs MFCC
bl=sorted(acc_by_book); a1=[np.mean((p==y)[bk==b]) for b in bl]; a2=[np.mean((pm==y)[bk==b]) for b in bl]
print('per-book acc corr MMS vs MFCC r=',round(np.corrcoef(a1,a2)[0,1],3))
# 3) Train on clean (all books) layer 15, predict NOISY books and tzj; open-set AUROC
Xall=MMSL[L15]
m=clf().fit(Xall[clean],y)
for (Lb,bname) in sorted(set((lang[i],rows[i]['book']) for i in np.where(noisy)[0])):
    idx=[i for i in np.where(noisy)[0] if lang[i]==Lb and rows[i]['book']==bname]
    pr=m.predict(Xall[idx]); print(f'NOISY book [{Lb}] {bname!r} n={len(idx)} predicted: {dict(collections.Counter(LANGS[q] for q in pr))}')
ti=np.where(tzj)[0]; Pt=m.predict_proba(Xall[ti])
print('\nTZJ open-set predicted dist:',dict(collections.Counter(LANGS[q] for q in Pt.argmax(1))),' mean maxprob',round(Pt.max(1).mean(),3))
for bname in sorted(set(rows[i]['book'] for i in ti)):
    ii=[k for k,i in enumerate(ti) if rows[i]['book']==bname]; print('   tzj book',repr(bname),len(ii),dict(collections.Counter(LANGS[q] for q in Pt[ii].argmax(1))))
mp_in=P.max(1); mp_t=Pt.max(1)
lab=np.r_[np.ones(len(mp_in)),np.zeros(len(mp_t))]; sc=np.r_[mp_in,mp_t]
print('AUROC maxprob in-set(grouped OOF, layer-selected) vs tzj:',round(roc_auc_score(lab,sc),3))
# in-set OOF at fixed layer 15 for a like-for-like comparison
yy,P15,_=run([Xall],clean,True,select=False)
print('AUROC (fixed layer15 OOF):',round(roc_auc_score(lab,np.r_[P15.max(1),mp_t]),3))
# duration-matched: tzj clips <=10s only (no truncation)
short=dur[ti]<=10
print('AUROC tzj<=10s only (n=%d):'%short.sum(),round(roc_auc_score(np.r_[np.ones(len(mp_in)),np.zeros(short.sum())],np.r_[mp_in,mp_t[short]]),3))
# thresholds: at maxprob threshold t, coverage of in-set & rejection of tzj
for t in [0.5,0.7,0.9]:
    print(f'  thr {t}: in-set kept {np.mean(mp_in>=t):.2f} (acc of kept {np.mean((P.argmax(1)==y)[mp_in>=t]):.3f}), tzj rejected {np.mean(mp_t<t):.2f}')
# MFCC open-set for comparison
mm=clf().fit(MF[clean],y); Ptm=mm.predict_proba(MF[ti])
print('MFCC tzj predicted dist',dict(collections.Counter(LANGS[q] for q in Ptm.argmax(1))),'AUROC',round(roc_auc_score(lab,np.r_[Pm.max(1),Ptm.max(1)]),3))
# 4) As-labeled (noisy kek/spa kept) grouped CV with layer selection
ya,Pa,ch=run(MMSL,asl,True,select=True); pa=Pa.argmax(1)
print('\nAS-LABELED MMS grouped: macroF1',round(mf1(ya,pa),3),'recall',dict(zip(LANGS,np.round(recall_score(ya,pa,average=None),3))),'layers',[LAYERS[c] for c in ch])
na=noisy[asl]
print('   on the noisy-book clips (scored vs their wrong label): acc',round((pa[na]==ya[na]).mean(),3),' preds',dict(collections.Counter(LANGS[q] for q in pa[na])))
# 5) Spanish clips from books whose title is shared with a Mayan book (bilingual series, plausibly same reader) vs other
bi=json.load(open(B+'book_info.json')); 
shared={b for b in bi['spa'] if any(b in bi[L] for L in ['mam','quc','kek','cak','kjb'])}
sp=np.array([lang[i]=='spa' for i in clean]); sh=np.array([rows[i]['book'] in shared for i in clean])
print('spa recall: shared-title books n=%d %.3f ; other spa n=%d %.3f'%((sp&sh).sum(),(p[sp&sh]==0).mean(),(sp&~sh).sum(),(p[sp&~sh]==0).mean()))
# 6) publisher-as-proxy: within-language accuracy by publisher
pubs=np.array([bi[lang[i]][rows[i]['book']]['copyright'][:30] for i in clean])
for L in ['mam','kjb','spa','quc']:
    for pb in set(pubs[lang[clean]==L]):
        mk=(lang[clean]==L)&(pubs==pb)
        if mk.sum()>=10: print(f'  {L} publisher {pb!r} n={mk.sum()} recall={np.mean(p[mk]==LI[L]):.2f}')
