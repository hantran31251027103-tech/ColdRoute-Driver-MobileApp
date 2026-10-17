#!/usr/bin/env python
"""
Chuyển coldchain_early_warning_FINAL.pkl -> ONNX (3 cửa sổ) + metadata cho frontend.

Cài đặt (Linux/WSL/Colab, Python 3.12):
  pip install numpy==2.4.4 pandas==3.0.2 scikit-learn==1.8.0 xgboost==3.4.1 \
              onnx onnxmltools onnxconverter-common onnxruntime

Chạy:
  python convert_to_onnx.py --pkl coldchain_early_warning_FINAL.pkl \
      --data-src ./csv --inference-py ./coldchain_inference.py
"""
import argparse, copy, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

HISTORY_COLS = ["Batch_ID", "Timestamp", "Shipment_Mode", "calibrated_temp", "Humidity"]
COPY_FILES = ["fefo_ranking.csv", "batches.csv", "products.csv", "model1_alerts_long.csv"]


def patch_base_score(model):
    """xgboost>=3 lưu base_score dạng '[5E-1]' -> onnxmltools có thể không parse được."""
    try:
        booster = model.get_booster()
        cfg = json.loads(booster.save_config())
        p = cfg["learner"]["learner_model_param"]
        p["base_score"] = repr(float(str(p["base_score"]).strip("[]")))
        booster.load_config(json.dumps(cfg))
    except Exception as e:  # không chặn: parity check sẽ báo nếu sai
        print(f"  [warn] không patch được base_score: {e}")


def to_onnx(model, n_features):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    model = copy.deepcopy(model)
    booster = model.get_booster()
    try:  # early stopping: chỉ giữ các cây tới best_iteration
        best = model.best_iteration
        if best is not None and best + 1 < booster.num_boosted_rounds():
            model._Booster = booster[: best + 1]
            print(f"  cắt booster: {booster.num_boosted_rounds()} -> {best + 1} cây")
    except AttributeError:
        pass
    model.get_booster().feature_names = None  # onnxmltools cần f0..fN
    patch_base_score(model)
    return convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, n_features]))]
    )


def onnx_probs(sess, X):
    out = sess.run(None, {sess.get_inputs()[0].name: X})
    names = [o.name for o in sess.get_outputs()]
    p = out[names.index("probabilities")] if "probabilities" in names else out[1]
    if isinstance(p, list):  # ZipMap
        p = np.array([[d[0], d[1]] for d in p], dtype=np.float32)
    return np.asarray(p)[:, 1]


def parity(ref_model, onnx_path, n_features, tag):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 60, (3000, n_features)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan  # model xử lý missing
    diff = np.abs(ref_model.predict_proba(X)[:, 1] - onnx_probs(sess, X)).max()
    print(f"  [{tag}] parity max|Δ| = {diff:.2e}")
    if diff > 1e-4:
        sys.exit(f"[{tag}] ONNX KHÔNG khớp XGBoost (Δ={diff}). Xem phần Troubleshooting.")


def make_golden(bundle, inference_py, history_csv, n_batches, out_path):
    """Dùng chính coldchain_inference.py để sinh feature/xác suất chuẩn cho kiểm thử TS."""
    sys.path.insert(0, str(Path(inference_py).resolve().parent))
    import coldchain_inference as ci
    df = pd.read_csv(history_csv, usecols=HISTORY_COLS, parse_dates=["Timestamp"])
    ids = sorted(df.Batch_ID.unique())[:n_batches]
    sub = df[df.Batch_ID.isin(ids)]
    pred = ci.predict(sub, bundle, operating_point="precision50")
    ci.SM.clear(); ci.SM.update(bundle["shipment_mode_map"])
    d, F = ci.build_features(ci.prepare(sub))
    last = d.groupby("Batch_ID").tail(1).index
    golden = {"batches": d.loc[last, "Batch_ID"].tolist(), "tasks": {}}
    for k, mb in bundle["models"].items():
        spec = mb["candidates"][mb["best"]]
        X = F.loc[last, bundle["t_features"]].copy()
        X["target_hour"] = (d.loc[last, "Timestamp"].dt.hour.to_numpy() + mb["b"]) % 24
        X = X[spec["features"]].astype(np.float32)
        p = pred[pred.task == k].set_index("Batch_ID").loc[golden["batches"]]
        nn = lambda a: [None if pd.isna(v) else float(v) for v in a]
        golden["tasks"][k] = {
            "features": spec["features"],
            "X": [[None if pd.isna(v) else float(v) for v in r] for r in X.to_numpy()],
            "p_cal": nn(p.p_cal), "alert": [bool(a) for a in p.alert],
            "status": p.status.tolist(),
        }
    out_path.write_text(json.dumps(golden), encoding="utf-8")
    print(f"  golden.json: {len(ids)} lô")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--frontend", default=".", help="thư mục gốc frontend")
    ap.add_argument("--data-src", help="thư mục chứa các CSV nguồn")
    ap.add_argument("--inference-py", help="coldchain_inference.py (để tạo golden.json)")
    ap.add_argument("--op", default="precision50")
    a = ap.parse_args()

    root = Path(a.frontend)
    mdir, ddir = root / "public/models", root / "public/data"
    mdir.mkdir(parents=True, exist_ok=True); ddir.mkdir(parents=True, exist_ok=True)

    with open(a.pkl, "rb") as f:
        bundle = pickle.load(f)
    print("bundle version:", bundle.get("version"), "| tasks:", sorted(bundle["models"]))

    meta = {"version": bundle.get("version"), "operatingPoint": a.op,
            "shipmentModeMap": bundle["shipment_mode_map"], "tasks": {}}
    for k, mb in sorted(bundle["models"].items()):
        assert mb["best"] == "t_only", f"{k}: bundle dùng '{mb['best']}' (cần SARIMA) – không hỗ trợ"
        spec = mb["candidates"][mb["best"]]
        feats = list(spec["features"]); n = len(feats)
        print(f"[{k}] {n} features")
        onx = to_onnx(spec["model"], n)
        path = mdir / f"model_{k}.onnx"
        path.write_bytes(onx.SerializeToString())
        parity(spec["model"], path, n, k)
        cal = spec["calibrator"]
        meta["tasks"][k] = {
            "onnx": path.name, "a": mb["a"], "b": mb["b"], "window": f"t+{mb['a']}..t+{mb['b']}",
            "features": feats, "threshold": spec["thresholds"].get(a.op),
            "calibrator": {"x": [float(v) for v in cal.X_thresholds_],
                           "y": [float(v) for v in cal.y_thresholds_]},
        }
    (mdir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if a.data_src:
        src = Path(a.data_src)
        for fn in COPY_FILES:
            if (src / fn).exists(): shutil.copy(src / fn, ddir / fn)
        fe = src / "features_engineered.csv"
        if fe.exists():  # file slim (~20MB) chỉ gồm 5 cột model cần
            slim = ddir / "history_for_model.csv"
            pd.read_csv(fe, usecols=HISTORY_COLS).to_csv(slim, index=False)
            print("history_for_model.csv:", round(slim.stat().st_size / 1e6, 1), "MB")
            if a.inference_py:
                make_golden(bundle, a.inference_py, slim, 25, ddir / "golden.json")
    print("Xong ->", mdir)


if __name__ == "__main__":
    main()