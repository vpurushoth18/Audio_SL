"""The discriminating classification test: 476 speakers, not 7 or 12.

The cross-corpus classification check saturated, because OpenSLR-30 has 12
speakers and Omnilingual 7, and a linear probe separates that many classes from
almost any layer. OpenSLR-52 has 476 speakers and the benchmark's final-layer
accuracy there was 0.776, so there is real headroom between layers.

This is a within-corpus test: OpenSLR-52 utterances are split into a selection
half and a scoring half, disjoint by utterance but covering the same speaker
inventory, which is what speaker classification means. The layer is chosen on
the selection half only.
"""
import os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

ROOT="/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DEV="cuda" if torch.cuda.is_available() else "cpu"
NAMES={"xlsr300m":"facebook/wav2vec2-xls-r-300m","mms300m":"facebook/mms-300m",
 "w2vbert2":"facebook/w2v-bert-2.0","whisper":"openai/whisper-large-v3",
 "wavlm":"microsoft/wavlm-large","hubert":"facebook/hubert-large-ls960-ft"}
clean=lambda a: np.nan_to_num(a.astype(np.float32),nan=0.,posinf=0.,neginf=0.)

def fit(Xtr,ytr,Xte,yte,k,l2=1.0):
    Xtr=torch.as_tensor(Xtr,device=DEV);Xte=torch.as_tensor(Xte,device=DEV)
    ytr=torch.as_tensor(ytr,device=DEV,dtype=torch.long)
    yte=torch.as_tensor(yte,device=DEV,dtype=torch.long)
    W=torch.zeros(Xtr.shape[1],k,device=DEV,requires_grad=True)
    b=torch.zeros(k,device=DEV,requires_grad=True)
    opt=torch.optim.LBFGS([W,b],max_iter=200,history_size=10,
                          line_search_fn="strong_wolfe")
    lf=torch.nn.CrossEntropyLoss()
    def cl():
        opt.zero_grad()
        l=lf(Xtr@W+b,ytr)+l2/(2*len(Xtr))*(W*W).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): a=((Xte@W+b).argmax(1)==yte).float().mean().item()
    del Xtr,Xte,ytr,yte,W,b; return a

def probe(X,y,seeds=(0,1,2)):
    keep=pd.Series(y).groupby(y).transform("size").values>=4
    X,y=X[keep],np.asarray(y)[keep]
    codes,uniq=pd.factorize(y)
    o=[]
    for s in seeds:
        Xa,Xb,ya,yb=train_test_split(X,codes,test_size=0.4,random_state=s,
                                     stratify=codes)
        # standardisation fitted on TRAIN ONLY, applied to test
        mu,sd=Xa.mean(0),Xa.std(0)+1e-9
        Xa=((Xa-mu)/sd).astype(np.float32); Xb=((Xb-mu)/sd).astype(np.float32)
        o.append(fit(Xa,ya,Xb,yb,len(uniq)))
    return float(np.mean(o)),float(np.std(o)),int(len(y)),int(len(uniq))

sub=pd.read_csv(f"{ROOT}/eval/subset.tsv",sep="\t",quoting=3,
                dtype={"speaker_id":str}).set_index("utt_id")
J=pd.read_csv(f"{ROOT}/eval/analysis/J_layer_transfer.tsv",sep="\t")
cl_pick={r.tag:int(r.picked_layer) for _,r in J[J.capability=="speaker"].iterrows()}
rng=np.random.default_rng(0)
rows=[]
for tag in sorted(NAMES):
    z=np.load(f"{ROOT}/eval/emb/{tag}.npz",allow_pickle=True)
    emb,utt=z["emb"],z["utt"].astype(str); meta=sub.loc[utt]
    idx=np.flatnonzero(meta.dataset.values=="openslr52")
    # split utterances into a selection half and a scoring half
    half=rng.permutation(idx); sel_i,sco_i=half[:len(half)//2],half[len(half)//2:]
    y_sel=meta.speaker_id.values[sel_i]; y_sco=meta.speaker_id.values[sco_i]
    nl=emb.shape[0]; final=nl-1
    sweep=[probe(clean(emb[li])[sel_i],y_sel,seeds=(0,))[0] for li in range(nl)]
    pick=int(np.argmax(sweep))
    s_m,s_sd,n,k=probe(clean(emb[pick])[sco_i],y_sco)
    f_m=probe(clean(emb[final])[sco_i],y_sco)[0]
    u_m=probe(clean(emb.mean(axis=0))[sco_i],y_sco)[0]
    c_m=probe(clean(emb[cl_pick[tag]])[sco_i],y_sco)[0]
    rows.append(dict(model=NAMES[tag],n_layers=nl,n=n,n_speakers=k,
        clf_picked_layer=pick,clustering_picked_layer=cl_pick[tag],
        sel_acc=s_m,sel_sd=s_sd,final_acc=f_m,uniform_acc=u_m,
        clustering_layer_acc=c_m,gain_vs_final=s_m-f_m,
        gain_vs_uniform=s_m-u_m,cost_of_clustering_layer=s_m-c_m))
    print(f"  {NAMES[tag]:34s} L{pick:>2d} (clust L{cl_pick[tag]:>2d})  "
          f"sel {s_m:.3f}±{s_sd:.3f}  final {f_m:.3f}  uniform {u_m:.3f}  "
          f"clust-layer {c_m:.3f}",flush=True)
    del emb,z
d=pd.DataFrame(rows).sort_values("gain_vs_final",ascending=False)
out=f"{ROOT}/eval/analysis/U_classification_476_speakers.tsv"
d.to_csv(out,sep="\t",index=False)
print(f"\n## U. Speaker classification, OpenSLR-52, {d.n_speakers.iloc[0]} speakers, "
      f"{d.n.iloc[0]} utts\n   layer chosen on a disjoint half of the utterances\n")
print(d[["model","clf_picked_layer","clustering_picked_layer","sel_acc","sel_sd",
         "final_acc","uniform_acc","clustering_layer_acc","gain_vs_final",
         "gain_vs_uniform","cost_of_clustering_layer"]]
      .to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print(f"\n  beats final:          {int((d.gain_vs_final>0).sum())}/{len(d)}  median {d.gain_vs_final.median():+.3f}")
print(f"  beats all-layer mean: {int((d.gain_vs_uniform>0).sum())}/{len(d)}  median {d.gain_vs_uniform.median():+.3f}")
print(f"  cost of reusing the clustering layer: median {d.cost_of_clustering_layer.median():+.3f}, max {d.cost_of_clustering_layer.max():+.3f}")
print(f"\nwrote {out}")
