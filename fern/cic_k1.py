"""CIC-IoT-2023 non-chain transfer check.

CIC-IoT-2023 CSV has 46 flow features + label but NO src/dst IPs, so attacker-victim chains
cannot be built. Instead we run the K1 DIAGNOSTIC at the per-flow level: train several
architectures as multiclass kill-chain-stage predictors on CIC's own features, then show that
models matched on detection F1 diverge on FORENSIC per-flow fidelity = MACRO stage-recall
(mean over malicious stages of recall). This demonstrates "detection accuracy != forensic
stage identification" is not an Edge-IIoTset construction. Clearly labeled non-chain evidence.
"""
import argparse, json
import numpy as np
import pandas as pd
from stage_mapping import map_attack, UNKNOWN
from stage_clf import STAGE_CLASSES

CIDX = {c: i for i, c in enumerate(STAGE_CLASSES)}
STAGES = STAGE_CLASSES[1:]   # exclude Normal


def load_cic(path, max_rows, seed):
    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    label = df["label"].astype(str).values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([CIDX.get(map_attack(l, "ciciot")[0] if map_attack(l, "ciciot")[0] not in (None, UNKNOWN) else "Normal", 0)
                  for l in label], dtype=np.int64)
    return X, y


def det_f1(yt_idx, yp_idx):
    yt = (yt_idx != 0).astype(int); yp = (yp_idx != 0).astype(int)
    tp = int(((yp==1)&(yt==1)).sum()); fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); return 2*p*r/max(1e-9,p+r)


def macro_stage_recall(yt_idx, yp_idx):
    """Forensic per-flow fidelity proxy: mean recall over malicious stages present in truth."""
    recs = []
    for s in STAGES:
        si = CIDX[s]
        m = (yt_idx == si)
        if m.sum() == 0:
            continue
        recs.append(float((yp_idx[m] == si).mean()))
    return float(np.mean(recs)) if recs else 0.0


def models(Xtr, ytr, Xva, Xte, dev):
    import lightgbm as lgb, torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {}
    nc = len(STAGE_CLASSES)
    Xeval = np.vstack([Xva, Xte]); nv = len(Xva)
    # lgbm
    m = lgb.train({"objective":"multiclass","num_class":nc,"verbose":-1,"num_leaves":31,"learning_rate":0.1},
                  lgb.Dataset(Xtr,label=ytr),80)
    p = np.argmax(m.predict(Xeval),1); out["lgbm"]=(p[:nv],p[nv:])
    # logreg
    sc=StandardScaler().fit(Xtr); lr=LogisticRegression(max_iter=300).fit(sc.transform(Xtr),ytr)
    p=lr.predict(sc.transform(Xeval)); out["logreg"]=(p[:nv],p[nv:])
    # mlp + tinyml
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
    def train_net(hidden):
        Xt=torch.tensor((Xtr-mu)/sd,device=dev,dtype=torch.float32); yt=torch.tensor(ytr,device=dev)
        layers=[]; d=Xtr.shape[1]
        for h in hidden: layers+=[nn.Linear(d,h),nn.ReLU()]; d=h
        layers+=[nn.Linear(d,nc)]; net=nn.Sequential(*layers).to(dev)
        opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); bs=8192
        for _ in range(12):
            pm=torch.randperm(len(Xt),device=dev)
            for i in range(0,len(Xt),bs):
                idx=pm[i:i+bs]; opt.zero_grad(); lf(net(Xt[idx]),yt[idx]).backward(); opt.step()
        with torch.no_grad():
            pe=net(torch.tensor((Xeval-mu)/sd,device=dev,dtype=torch.float32)).argmax(1).cpu().numpy()
        return pe
    p=train_net([64,32]); out["mlp"]=(p[:nv],p[nv:])
    p=train_net([8]); out["tinyml"]=(p[:nv],p[nv:])
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--max-rows", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/cic_k1.json")
    args=ap.parse_args()
    Xtr_all,ytr_all=load_cic(args.train,args.max_rows,args.seed)
    Xte,yte=load_cic(args.test,args.max_rows//2,args.seed)
    # carve a val slice from train for any calibration; here just split
    n=len(Xtr_all); cut=int(0.85*n)
    Xtr,ytr=Xtr_all[:cut],ytr_all[:cut]; Xva,yva=Xtr_all[cut:],ytr_all[cut:]
    import torch; dev="cuda" if torch.cuda.is_available() else "cpu"
    print(f"CIC train {len(Xtr)} val {len(Xva)} test {len(Xte)} | feats {Xtr.shape[1]} | "
          f"stage dist test {np.bincount(yte, minlength=len(STAGE_CLASSES))}")
    preds=models(Xtr,ytr,Xva,Xte,dev)
    rows={}
    for name,(pv,pt) in preds.items():
        f1=det_f1(yte,pt); msr=macro_stage_recall(yte,pt)
        rows[name]={"det_f1":f1,"macro_stage_recall":msr}
        print(f"[{name:8s}] detF1={f1:.4f}  macro_stage_recall={msr:.3f}")
    f1s=np.array([rows[m]["det_f1"] for m in rows]); msrs=np.array([rows[m]["macro_stage_recall"] for m in rows])
    summary={"det_f1_spread":float(f1s.max()-f1s.min()),"macro_stage_recall_spread":float(msrs.max()-msrs.min()),
             "k1_transfers": bool((f1s.max()-f1s.min())<0.06 and (msrs.max()-msrs.min())>0.10)}
    rows["_summary"]=summary
    print(f"\n=== CIC K1 (non-chain) === detF1 spread={summary['det_f1_spread']:.4f} "
          f"macro_stage_recall spread={summary['macro_stage_recall_spread']:.3f} "
          f"-> diagnostic transfers: {summary['k1_transfers']}")
    json.dump(rows,open(args.out,"w"),indent=2); print("wrote",args.out)


if __name__=="__main__":
    main()



