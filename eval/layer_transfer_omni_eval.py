"""Evaluate the ALREADY-FROZEN layer choices on Omnilingual.

The dev-side sweep lives in J_layer_transfer.tsv: layers were chosen on
OpenSLR-52 speaker AMI and frozen. Re-running that sweep to test a second
corpus is wasted work, since the choice does not depend on the test set. This
reuses those choices and only evaluates them on Omnilingual Sinhala, a corpus
collected separately from the OpenSLR pair.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

ROOT="/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SEEDS=(0,1,2,3,4)
l2=lambda X: X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-9)
clean=lambda a: np.nan_to_num(a.astype(np.float32),nan=0.,posinf=0.,neginf=0.)

def ami(X,y,seeds=SEEDS):
    yy=pd.factorize(y)[0]; k=len(np.unique(yy))
    o=[adjusted_mutual_info_score(yy,MiniBatchKMeans(n_clusters=k,random_state=s,
        n_init=5,batch_size=512).fit_predict(l2(X))) for s in seeds]
    return float(np.mean(o)), float(np.std(o))

te=pd.read_csv(f"{ROOT}/eval/subset_omni.tsv",sep="\t",quoting=3,
               dtype={"speaker_id":str}).set_index("utt_id")
J=pd.read_csv(f"{ROOT}/eval/analysis/J_layer_transfer.tsv",sep="\t")
spk={r.tag:(int(r.picked_layer),r.model,int(r.n_layers)) for _,r in
     J[J.capability=="speaker"].iterrows()}

rows=[]
for tag,(pick,name,nl) in sorted(spk.items()):
    f=f"{ROOT}/eval/emb_omni/{tag}.npz"
    if not os.path.exists(f): continue
    z=np.load(f,allow_pickle=True); utt=z["utt"].astype(str); E=z["emb"]
    meta=te.loc[utt]; y=meta.speaker_id.values
    final=E.shape[0]-1
    sm,ss=ami(clean(E[pick]),y)
    fm,fs=ami(clean(E[final]),y)
    um,us=ami(clean(E.mean(axis=0)),y)
    per=[ami(clean(E[li]),y,seeds=(0,))[0] for li in range(E.shape[0])]
    best=int(np.argmax(per))
    rows.append(dict(model=name,tag=tag,n_layers=E.shape[0],
        n_test=len(utt),n_speakers=int(meta.speaker_id.nunique()),
        picked_layer=pick,picked_depth=pick/max(final,1),
        sel_mean=sm,sel_sd=ss,final_mean=fm,uniform_mean=um,
        best_test_layer=best,best_test_mean=per[best],
        gain_vs_final=sm-fm,gain_vs_uniform=sm-um))
    print(f"  {name:34s} L{pick:>2d}  sel {sm:.3f}±{ss:.3f}  final {fm:.3f}  "
          f"uniform {um:.3f}  oracle {per[best]:.3f}(L{best})",flush=True)
    del z,E
d=pd.DataFrame(rows).sort_values("gain_vs_final",ascending=False)
out=f"{ROOT}/eval/analysis/O_layer_transfer_omnilingual.tsv"
d.to_csv(out,sep="\t",index=False)
print("\n## O. Layer transfer, SECOND test corpus (Omnilingual Sinhala)")
print(f"   layer chosen on OpenSLR-52 and frozen; test = {d.n_test.iloc[0]} utts, "
      f"{d.n_speakers.iloc[0]} speakers; 5 seeds\n")
print(d[["model","picked_layer","picked_depth","sel_mean","sel_sd","final_mean",
         "uniform_mean","best_test_mean","gain_vs_final","gain_vs_uniform"]]
      .to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print(f"\n  beats final layer:  {int((d.gain_vs_final>0).sum())}/{len(d)}  "
      f"median {d.gain_vs_final.median():+.3f}")
print(f"  beats all-layer mean: {int((d.gain_vs_uniform>0).sum())}/{len(d)}  "
      f"median {d.gain_vs_uniform.median():+.3f}")
print(f"\nwrote {out}")
