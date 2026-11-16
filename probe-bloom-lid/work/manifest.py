import json, random, collections, csv
B='/private/tmp/claude-501/-Volumes-KZ-Big-hackathons-veritas-ai/affd5d58-427a-4611-9d58-42f1f63fe19e/scratchpad/bloom/'
d=json.load(open(B+'bloom_speech.json')); bi=json.load(open(B+'book_info.json'))
LANGS=['spa','mam','quc','kek','cak','kjb']
random.seed(0)
rows=[]
for L in LANGS+['tzj']:
    info=bi[L]
    clips=d[L]
    print(f'== {L}: {len(clips)} clips, {len(info)} books in book_info')
    excl=[]
    for bk,v in info.items():
        if v['quarantine'] or v['license']=='cc-by-nc-nd': excl.append((bk,v['license'],v['quarantine']))
    print('  excluded books:',excl)
    keep=[c for c in clips if c['book'] in info and not info[c['book']]['quarantine'] and info[c['book']]['license']!='cc-by-nc-nd' and not c['containsNumber']]
    missing=[c for c in clips if c['book'] not in info]
    print('  clips w/o book_info:',len(missing),' containsNumber:',sum(c['containsNumber'] for c in clips),' kept:',len(keep))
    tags=collections.Counter(c['bloomLanguageTag'] for c in keep); print('  tags',dict(tags))
    bybook=collections.defaultdict(list)
    for c in keep: bybook[c['book']].append(c)
    print('  books kept:',len(bybook), sorted([len(v) for v in bybook.values()],reverse=True))
    if L=='tzj':
        sel=keep if keep else [c for c in clips]
    else:
        # cap 35/book, then sample up to 250 overall round-robin-ish
        pools={bk:random.sample(v,min(35,len(v))) for bk,v in bybook.items()}
        cand=[c for v in pools.values() for c in v]
        sel=random.sample(cand,min(250,len(cand))) if len(cand)>250 else cand
    nb=collections.Counter(c['book'] for c in sel)
    print('  selected',len(sel),'from',len(nb),'books; per-book',sorted(nb.values(),reverse=True))
    for c in sel:
        rows.append(dict(lang=L,book=c['book'],url=c['url'],text=c['text'],fn=c['url'].split('/')[-1]))
with open(B+'work/manifest.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('total',len(rows))
# titles that mention other languages
for L in LANGS+['tzj']:
    for bk in bi[L]:
        if any(s in bk.lower() for s in ["q'anjob","qanjob","anjob","mam","k'iche","kiche","quiche","q'eqchi","qeqchi","kaqchikel","kakchiquel","tz'utujil","tzutujil","espa","spanish","ingl","english","achi","ixil","poqom","chuj","akateko","jakalteko","popti"]):
            print('TITLE-FLAG',L,repr(bk),bi[L][bk]['bloomLanguageTag'])
