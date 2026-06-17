"""Quick per-stage recall for the headline FERN selector (value-per-byte greedy), Edge + IoT-23,
budgets 1/5/10%, 3 seeds. Uses the fixed per_stage_recall from evaluation_suite."""
import json, os
import numpy as np
from collections import defaultdict
from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth
from stage_clf import train_stage_clf, predict_stages
from evaluation_suite import strat_split, per_stage_recall, STAGE_ORDER
import iot23_pipeline as I23

BUD = [0.01, 0.05, 0.10]

def edge(seeds=(0,1,2)):
    import torch; dev="cuda" if torch.cuda.is_available() else "cpu"
    flows=load_flows("data/processed/edge_flows_full.jsonl"); X,_=make_xy(flows,"edge-iiot")
    acc={b:{s:[] for s in STAGE_ORDER} for b in BUD}
    for sd in seeds:
        rng=np.random.default_rng(sd); tr,te=strat_split([f["attack"] for f in flows],rng)
        ftr=[flows[i] for i in tr]; fte=[flows[i] for i in te]; Xtr,Xte=X[tr],X[te]
        ce=np.array([float(byte_cost(f)) for f in fte]); full=float(ce.sum())
        gt=build_ground_truth(fte,"edge-iiot")
        v=forensic_value_labels(ftr,"edge-iiot",n_rare_stages=3,anchor_focused=True)
        s_te,_=train_fern_scorer(Xtr,v,Xte,dev=dev)
        sm=train_stage_clf(Xtr,ftr,"edge-iiot"); pst=predict_stages(sm,Xte)
        for b in BUD:
            mask=keep_under_budget(np.argsort(-(s_te/np.sqrt(ce))),ce,b*full)
            ps=per_stage_recall(fte,mask,pst,gt,"edge-iiot")
            for s in STAGE_ORDER:
                if ps[s] is not None: acc[b][s].append(ps[s])
    return {f"{b*100:g}":{s:(float(np.mean(acc[b][s])) if acc[b][s] else None) for s in STAGE_ORDER} for b in BUD}

def iot23(seeds=(0,1,2)):
    import torch; dev="cuda" if torch.cuda.is_available() else "cpu"
    from hardening_suite import iot23_value_labels
    flows=[]
    for fn in sorted(os.listdir("data/raw/iot23")):
        if not fn.endswith(".csv"): continue
        fl=I23.parse_csv(os.path.join("data/raw/iot23",fn),300000)
        tag=fn.replace("dataset","s").replace(".csv","")
        for f in fl: f["src"]=f"{tag}:{f['src']}"; f["dst"]=f"{tag}:{f['dst']}"
        flows+=fl
    X=I23.featmat(flows); ys=I23.stage_y(flows)
    acc={b:{s:[] for s in STAGE_ORDER} for b in BUD}
    for sd in seeds:
        rng=np.random.default_rng(sd); scen=[f["src"].split(":")[0] for f in flows]
        tr,te=strat_split([f"{s}|{y}" for s,y in zip(scen,ys)],rng)
        ftr=[flows[i] for i in tr]; fte=[flows[i] for i in te]; Xtr,Xte=X[tr],X[te]
        ce=np.array([max(1.0,f["bytes_tot"]) for f in fte]); full=float(ce.sum())
        gt=I23.host_ground_truth(fte)
        if not gt: continue
        _,anc_tr=iot23_value_labels(ftr)
        s_te,_=train_fern_scorer(Xtr,anc_tr,Xte,dev=dev)
        import lightgbm as lgb
        sm=lgb.train({"objective":"multiclass","num_class":len(I23.CLASSES),"verbose":-1,"num_leaves":31,"learning_rate":0.1},lgb.Dataset(Xtr,label=ys[tr]),80)
        pst=np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xte),1)],dtype=object)
        for b in BUD:
            mask=keep_under_budget(np.argsort(-(s_te/np.sqrt(ce))),ce,b*full)
            ps=per_stage_recall(fte,mask,pst,gt,"iot23",host=True)
            for s in STAGE_ORDER:
                if ps[s] is not None: acc[b][s].append(ps[s])
    return {f"{b*100:g}":{s:(float(np.mean(acc[b][s])) if acc[b][s] else None) for s in STAGE_ORDER} for b in BUD}

if __name__=="__main__":
    out={"edge":edge(),"iot23":iot23()}
    json.dump(out,open("outputs/perstage.json","w"),indent=2)
    print(json.dumps(out,indent=1))



