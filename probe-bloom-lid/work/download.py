import csv, os, subprocess, urllib.request, numpy as np, time
from concurrent.futures import ThreadPoolExecutor
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
A=B+'audio/'; W=B+'work/wav16k/'; os.makedirs(W,exist_ok=True)
rows=list(csv.DictReader(open(B+'work/manifest.csv')))
def job(r):
    p=A+r['fn']; out=W+r['fn'].replace('.mp3','.npy')
    try:
        if not os.path.exists(p) or os.path.getsize(p)==0:
            for t in range(3):
                try:
                    urllib.request.urlretrieve(r['url'],p); break
                except Exception as e:
                    if t==2: raise
                    time.sleep(1)
        if not os.path.exists(out):
            raw=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',p,'-ac','1','-ar','16000','-f','f32le','-'],capture_output=True,check=True).stdout
            x=np.frombuffer(raw,dtype=np.float32)
            np.save(out,x)
        else:
            x=np.load(out,mmap_mode='r')
        return r['fn'], len(x)/16000, None
    except Exception as e:
        return r['fn'], None, repr(e)[:200]
t0=time.time()
with ThreadPoolExecutor(24) as ex: res=list(ex.map(job,rows))
errs=[x for x in res if x[2]]
print('done',len(res),'errors',len(errs),'secs',round(time.time()-t0,1))
for e in errs[:10]: print(e)
with open(B+'work/durations.csv','w') as f:
    f.write('fn,dur\n')
    for fn,d,e in res:
        if d is not None: f.write(f'{fn},{d}\n')
