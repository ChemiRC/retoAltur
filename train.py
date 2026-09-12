import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, json, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from features import extract_all, extract_from_arrays, load_stereo_from_bytes
from augment import augment_stereo

def compute_eer(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])

def _extract_one(args):
    path, turns_path = args
    with open(path, "rb") as f:
        caller, agent, sr = load_stereo_from_bytes(f.read())
    turns_list = None
    if turns_path and os.path.exists(turns_path):
        with open(turns_path) as tf:
            turns_list = json.load(tf)["turns"]
    return extract_from_arrays(caller, agent, sr, turns_list=turns_list)

def _extract_augmented(path, turns_path, n_aug=2):
    with open(path, "rb") as f:
        caller, agent, sr = load_stereo_from_bytes(f.read())
    turns_list = None
    if turns_path and os.path.exists(turns_path):
        with open(turns_path) as tf:
            turns_list = json.load(tf)["turns"]
    out = [extract_from_arrays(caller, agent, sr, turns_list=turns_list)[0]]
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    for _ in range(n_aug):
        c, a = augment_stereo(caller, agent, rng)
        out.append(extract_from_arrays(c, a, sr, turns_list=turns_list)[0])
    return out

def build_dataset(data_dir, turns_dir=None, cache="features_cache_v2.npz"):
    df = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    df["path"] = df["anon_id"].apply(
        lambda a: os.path.join(data_dir, "audio", a + ".wav"))
    df["turns_path"] = df["anon_id"].apply(
        lambda a: os.path.join(turns_dir, a + ".json") if turns_dir else None)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    if os.path.exists(cache):
        print("   (usando cache " + cache + ")")
        d = np.load(cache, allow_pickle=True)
        return d["X"], df, list(d["names"])
    using_turns = turns_dir is not None and os.path.isdir(turns_dir)
    src = "turns JSON" if using_turns else "solo VAD"
    print("   extrayendo features de " + str(len(df)) + " audios (" + src + ", secuencial)...")
    names_ref, X = None, []
    for i, row in df.iterrows():
        tp = row["turns_path"] if using_turns else None
        vec, names = _extract_one((row["path"], tp))
        X.append(vec)
        names_ref = names
        if (i + 1) % 50 == 0:
            print("     " + str(i+1) + "/" + str(len(df)))
    X = np.array(X)
    np.savez(cache, X=X, names=np.array(names_ref, dtype=object))
    return X, df, names_ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--turns_dir", default=None)
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    print("1. Cargando dataset...")
    X, df, names = build_dataset(args.data_dir, args.turns_dir)
    y = (df["label"] == "synthetic").astype(int).values
    is_train = (df["split"] == "train").values
    is_val = (df["split"] == "val").values
    n_feats = str(X.shape[1])
    print("   " + str(len(X)) + " llamadas | " + n_feats + " features | train=" + str(is_train.sum()) + " val=" + str(is_val.sum()))

    Xtr, ytr = X[is_train], y[is_train]
    if args.augment:
        print("   Aumentando set de entrenamiento...")
        rows_train = df[is_train].reset_index(drop=True)
        aug_X, aug_y = [], []
        for i, (_, row) in enumerate(rows_train.iterrows()):
            tp = row["turns_path"] if args.turns_dir else None
            vecs = _extract_augmented(row["path"], tp)
            aug_X.extend(vecs)
            aug_y.extend([ytr[i]] * len(vecs))
            if (i + 1) % 50 == 0:
                print("     augment " + str(i+1) + "/" + str(len(rows_train)))
        Xtr, ytr = np.array(aug_X), np.array(aug_y)
        print("   train aumentado: " + str(len(Xtr)) + " muestras")

    print("2. Entrenando...")
    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.04,
        l2_regularization=1.0, max_depth=6, random_state=42)
    clf.fit(Xtr, ytr)
    val_scores = clf.predict_proba(X[is_val])[:, 1]
    eer, thr = compute_eer(y[is_val], val_scores)
    auc = roc_auc_score(y[is_val], val_scores)
    acc = accuracy_score(y[is_val], (val_scores >= 0.5).astype(int))
    print("   >> VAL  AUC=" + str(round(auc,4)) + "  EER=" + str(round(eer*100,2)) + "%  acc@0.5=" + str(round(acc*100,1)) + "%")

    print("3. Modelo final calibrado...")
    Xfinal = np.vstack([Xtr, X[is_val]]) if args.augment else X
    yfinal = np.concatenate([ytr, y[is_val]]) if args.augment else y
    base = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.04,
        l2_regularization=1.0, max_depth=6, random_state=42)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(Xfinal, yfinal)
    with open(args.out, "wb") as f:
        pickle.dump({"model": model, "feature_names": names}, f)
    with open("threshold.json", "w") as f:
        json.dump({"threshold": 0.5, "val_eer": eer, "val_auc": auc}, f, indent=2)
    print("4. Guardado " + args.out + " + threshold.json")

if __name__ == "__main__":
    main()