import json, os, sys

H=lambda g:[r[::-1]for r in g];V=lambda g:g[::-1];T=lambda g:[list(r)for r in zip(*g)]
def C(g):
 p=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v]
 return [r[min(j for _,j in p):max(j for _,j in p)+1]for r in g[min(i for i,_ in p):max(i for i,_ in p)+1]] if p else [[0]]
OPS={"identity":lambda g:g,"flip_h":H,"flip_v":V,"transpose":T,"crop_support":C}
BIN={"chain":lambda a,b:b(a),"overlay":lambda a,b:[[y or x for x,y in zip(r,s)]for r,s in zip(a,b)],"hcat":lambda a,b:[r+s for r,s in zip(a,b)],"vcat":lambda a,b:a+b}
ARC="1cf80156 67a3c6ac 68b16354 74dd1130 9dfd6313 28bf18c6 4c4377d9 6d0aefbc 6fa7a44f 7468f01a".split()

def run(p,g):
 return OPS[p](g) if type(p)==str else run(p[2],run(p[1],g)) if p[0]=="chain" else BIN[p[0]](run(p[1],g),run(p[2],g))

def show(p): return p if type(p)==str else f"({p[0]} {show(p[1])} {show(p[2])})"
def size(p): return 1 if type(p)==str else 1+size(p[1])+size(p[2])
def subs(p): return [] if type(p)==str else [p,*subs(p[1]),*subs(p[2])]
def acc(a,b): return 0 if len(a)!=len(b) or len(a[0])!=len(b[0]) else sum(x==y for r,s in zip(a,b) for x,y in zip(r,s))/sum(len(r) for r in b)
def score(p,t):
 z=[(run(p,e["input"]),e["output"])for e in t["train"]+t["test"]]
 return all(a==b for a,b in z),sum(acc(a,b)for a,b in z)/len(z),size(p)

def synth():
 ex=[[[1,0],[0,0]],[[0,2,0],[0,0,0]],[[0,0,3],[0,0,0]],[[0,1,0],[0,2,3],[0,0,0]],[[0,4,0],[0,0,0]],[[0,0,5],[0,0,0]]]
 ps="flip_h flip_v transpose (chain crop_support flip_h) (vcat flip_v crop_support) (hcat crop_support flip_h)".split(" | ") if 0 else ["flip_h","flip_v","transpose",("chain","crop_support","flip_h"),("vcat","flip_v","crop_support"),("hcat","crop_support","flip_h")]
 return {f"s{i+1}":{"train":[{"input":g,"output":run(p,g)}],"test":[{"input":H(g),"output":run(p,H(g))}]} for i,(g,p) in enumerate(zip(ex,ps))}

def arc():
 b=os.path.dirname(os.path.abspath(__file__)); xs=[os.getenv("ARC_AGI_1_TRAIN_DIR") or "(unset) ARC_AGI_1_TRAIN_DIR",f"{b}/data/ARC-AGI/data/training",f"{b}/../agi-core/data/ARC-AGI/data/training"]; d=next((x for x in xs if os.path.isdir(x)),None)
 if not d: print("missing ARC dataset; checked:\n- "+"\n- ".join(xs)); raise SystemExit(1)
 return {k:json.load(open(f"{d}/{k}.json")) for k in ARC}

def round2(lib):
 pool=list(OPS)+lib; out=[]
 for a in pool:
  for b in pool:
   if a in lib or b in lib:
    out += [("chain",a,b),("overlay",a,b),("hcat",a,b),("vcat",a,b)]
 return list(OPS)+out

def solve(name,tasks):
 lib=[]; solved=set()
 for r in (1,2):
  cand=list(OPS) if r==1 else round2(lib); best={}
  for k,t in tasks.items():
   ss=sorted(((score(p,t),p) for p in cand), key=lambda x:(-x[0][0],-x[0][1],x[0][2],show(x[1])))[0]
   if ss[0][0]: best[k]=ss[1]
  new=sorted(k for k in best if k not in solved); solved|=set(new)
  for k in new:
   for p in [best[k],*subs(best[k])]:
    if all(show(p)!=show(q) for q in lib) and (type(p)!=str or p in best.values()): lib.append(p)
  print(f"{name} round {r}: {len(solved)}/{len(tasks)} new={new} lib={[show(p) if type(p)!=str else p for p in lib]}")

if __name__=="__main__":
 m=sys.argv[1] if len(sys.argv)>1 else "both"
 if m in ("synthetic","both"): solve("synthetic",synth())
 if m in ("arc","both"): solve("arc",arc())
