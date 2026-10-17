"""coldchain_inference.py — Early-warning temperature-excursion predictor.
Models: XGB-T (thermal) và XGB-T + SARIMA; mỗi cửa sổ xuất model tốt nhất theo validation.
Definition: anomaly = actual_temp < 2 or > 8 degC (2.0 and 8.0 are NORMAL).  Only NORMAL origins (A_t == 0) are scored:
P(anomaly at t+H | A_t = 0, history up to t).  Input = reading history per batch (hourly):
columns Batch_ID, Timestamp, Shipment_Mode, calibrated_temp (or actual_temp), Humidity.

    from coldchain_inference import load_bundle, predict
    bundle = load_bundle("coldchain_early_warning_bundle.pkl")
    out = predict(history_df, bundle, operating_point="precision50")   # latest origin of every batch
"""
import pickle
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

TEMP_LOWER, TEMP_UPPER = 2.0, 8.0
CAUTION_LO, CAUTION_HI = 2.5, 7.5
LAGS = [1, 2, 3, 6, 12, 24]; ROLLS = [6, 12, 24]
TARGET_COL, ANOMALY_FLAG_COL = "actual_temp", "actual_temp_anomaly_flag"
SM = {}

def load_bundle(path):
    with open(path, "rb") as f: return pickle.load(f)

def _runlen(s, batch):
    rid = (s.ne(s.shift()) | batch.ne(batch.shift())).cumsum()
    return s.groupby(rid).cumcount().add(1) * s.astype(int)

def build_features(d):
    d = d.sort_values(["Batch_ID", "Timestamp"]).reset_index(drop=True)
    g = d.groupby("Batch_ID", sort=False); F = pd.DataFrame(index=d.index); T = d[TARGET_COL]; b = d["Batch_ID"]
    F["temp_now"] = T
    for k in LAGS: F[f"lag_{k}"] = g[TARGET_COL].shift(k)
    for w in ROLLS:
        r = g[TARGET_COL].rolling(w, min_periods=1)
        F[f"rolling_mean_{w}"] = r.mean().reset_index(level=0, drop=True); F[f"rolling_std_{w}"] = r.std().reset_index(level=0, drop=True)
        F[f"rolling_min_{w}"] = r.min().reset_index(level=0, drop=True);   F[f"rolling_max_{w}"] = r.max().reset_index(level=0, drop=True)
    F["diff_1"] = F["temp_now"] - F["lag_1"]; F["dev_from_24h"] = F["temp_now"] - F["rolling_mean_24"]
    F["batch_mean_sofar"] = g[TARGET_COL].expanding().mean().reset_index(level=0, drop=True)
    order = g.cumcount(); F["hours_since_start"] = order
    F["humidity"] = d["Humidity"]; F["humidity_rmean6"] = g["Humidity"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    F["transport_mode"] = d["Shipment_Mode"].map(SM)
    F["hour"] = d["Timestamp"].dt.hour; F["day_of_week"] = d["Timestamp"].dt.dayofweek
    F["hour_sin"] = np.sin(2 * np.pi * F["hour"] / 24); F["hour_cos"] = np.cos(2 * np.pi * F["hour"] / 24)
    # --- B: lịch sử anomaly (<= t) ---
    a = d[ANOMALY_FLAG_COL]; ga = a.groupby(b)
    F["n_anom_sofar"] = ga.cumsum()
    last = pd.Series(np.where(a == 1, order, np.nan), index=d.index).groupby(b).ffill(); F["hours_since_anom"] = order - last
    F["anom_last6"] = ga.rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    # --- T: THERMAL LOAD ---
    outside_core = ((T < CAUTION_LO) | (T > CAUTION_HI)); caution = outside_core & (a == 0)
    exc = np.maximum(T - TEMP_UPPER, 0) + np.maximum(TEMP_LOWER - T, 0)                   # °C·giờ vượt ngưỡng thật
    cload = np.maximum(T - CAUTION_HI, 0) + np.maximum(CAUTION_LO - T, 0)                  # °C·giờ ngoài lõi [2.5, 7.5]
    margin = np.minimum(T - TEMP_LOWER, TEMP_UPPER - T)                                    # >0 trong khoảng an toàn
    F["margin_now"] = margin
    for w in ROLLS: F[f"margin_min_{w}"] = margin.groupby(b).rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
    F["margin_slope_6"] = margin - margin.groupby(b).shift(6)
    F["consec_outside_core_h"] = _runlen(outside_core, b); F["consec_caution_h"] = _runlen(caution, b)
    for w in ROLLS:
        F[f"caution_hours_{w}"] = caution.astype(int).groupby(b).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        F[f"aulc_caution_{w}"] = cload.groupby(b).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
    F["aulc_excursion_24"] = exc.groupby(b).rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
    F["aulc_caution_cum"] = cload.groupby(b).cumsum(); F["aulc_excursion_cum"] = exc.groupby(b).cumsum()
    F["caution_hours_cum"] = caution.astype(int).groupby(b).cumsum()
    lastc = pd.Series(np.where(outside_core, order, np.nan), index=d.index).groupby(b).ffill(); F["hours_since_outside_core"] = order - lastc
    return d, F


def _two(a): a = np.asarray(a); return a[:, :, 0] if a.ndim == 3 else a
def _mats(ra):
    ssm = ra.model.ssm; T = _two(ssm["transition"]); Z = _two(ssm["design"]).reshape(-1); c = np.asarray(ssm["state_intercept"]); d = np.asarray(ssm["obs_intercept"])
    cc = (c[:, -1] if c.ndim == 2 and c.shape[1] > 1 else c.reshape(c.shape[0], -1)[:, 0]).reshape(-1); dd = float(np.asarray(d).reshape(-1)[-1]) if np.asarray(d).size else 0.0
    return T, Z, cc, dd

def sarima_pass(y, spec, params, h_list):
    """Lọc Kalman một lô với tham số cố định; trả innovation và (forecast h, path max, path min) cho mọi t. Chỉ dùng y_{<=t}."""
    mod = SARIMAX(y, order=tuple(spec["order"]), seasonal_order=tuple(spec["seasonal_order"]), trend="c", enforce_stationarity=True, enforce_invertibility=True)
    ra = mod.filter(np.asarray(params)); T, Z, cc, dd = _mats(ra); A = ra.filter_results.filtered_state.copy(); n = len(y)
    out = {}; rmax = np.full(n, -np.inf); rmin = np.full(n, np.inf)
    for h in range(1, max(h_list) + 1):
        A = T @ A + cc[:, None]; yh = Z @ A + dd; rmax = np.maximum(rmax, yh); rmin = np.minimum(rmin, yh)
        if h in h_list: out[h] = (yh.astype(np.float32), rmax.astype(np.float32), rmin.astype(np.float32))      # float32 như lúc huấn luyện
    return np.asarray(ra.filter_results.forecasts_error[0]).astype(np.float32), out

def sarima_frame(d, bundle, h_list):
    """DataFrame (index như d, đã sort theo Batch_ID, Timestamp) với các cột sar_* cho từng horizon."""
    spec, params = bundle["sarima"]["spec"], bundle["sarima"]["params"]; Tarr = d[TARGET_COL].to_numpy(float); n = len(d); h_all = sorted(set(h_list) | {1})        # luôn có h=1 (sar_fc1)
    inn = np.full(n, np.nan, np.float32); FC = {h: np.full(n, np.nan, np.float32) for h in h_all}; PX = {h: np.full(n, np.nan, np.float32) for h in h_all}; PN = {h: np.full(n, np.nan, np.float32) for h in h_all}
    codes = d["Batch_ID"].astype("category").cat.codes.to_numpy(); st = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]; en = np.r_[st[1:], n]
    for s, e in zip(st, en):
        i_, o = sarima_pass(Tarr[s:e], spec, params, h_all); inn[s:e] = i_
        for h, (f, mx, mn) in o.items(): FC[h][s:e] = f; PX[h][s:e] = mx; PN[h][s:e] = mn
    INN = pd.Series(inn.astype(float), index=d.index); ABS = INN.abs(); gb = d["Batch_ID"]
    common = pd.DataFrame({"sar_resid_1": INN, "sar_abs_resid_1": ABS,
        "sar_abs_resid_mean6": ABS.groupby(gb).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True),
        "sar_abs_resid_mean24": ABS.groupby(gb).rolling(24, min_periods=1).mean().reset_index(level=0, drop=True),
        "sar_abs_resid_max24": ABS.groupby(gb).rolling(24, min_periods=1).max().reset_index(level=0, drop=True),
        "sar_resid_mean6": INN.groupby(gb).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)})
    per_h = {}
    for h in h_list:
        f = FC[h].astype(float); per_h[h] = pd.DataFrame({"sar_fc1": FC[1], "sar_fc": f, "sar_fc_dev": f - Tarr, "sar_fc_margin": np.minimum(f - TEMP_LOWER, TEMP_UPPER - f),
                                            "sar_path_max": PX[h], "sar_path_min": PN[h], "sar_path_margin": np.minimum(PN[h] - TEMP_LOWER, TEMP_UPPER - PX[h])}, index=d.index)
    return common, per_h

def prepare(df):
    d = df.copy()
    if TARGET_COL not in d.columns: d[TARGET_COL] = pd.to_numeric(d["calibrated_temp"], errors="coerce")
    d["Timestamp"] = pd.to_datetime(d["Timestamp"]); d = d.sort_values(["Batch_ID", "Timestamp"]).reset_index(drop=True)
    assert d[TARGET_COL].notna().all(), "actual_temp thiếu/không phải số"
    d[ANOMALY_FLAG_COL] = ((d[TARGET_COL] < TEMP_LOWER) | (d[TARGET_COL] > TEMP_UPPER)).astype(int)     # [2,8] inclusive = normal
    for b, g in d.groupby("Batch_ID"): assert (g["Timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all(), f"lô {b}: reading không đều 1 giờ"
    return d

def predict(history, bundle, tasks=None, operating_point="precision50", model=None, latest_only=True):
    """Early-warning EXCURSION KÉO DÀI: P(episode anomaly >= 3 giờ BẮT ĐẦU trong cửa sổ t+a..t+b | origin normal, lịch sử <= t).
    tasks: danh sách khóa như 'w7_24' (alert >= 6 giờ trước onset) hoặc 'w1_12'...; model: 't_only' | 'hybrid' (mặc định = tốt nhất theo validation).
    status = 'NORMAL_ORIGIN' (được chấm điểm) hoặc 'ALREADY_EXCURSION' (đã vi phạm: không thuộc bài toán early-warning)."""
    SM.clear(); SM.update(bundle["shipment_mode_map"]); d = prepare(history); keys = tasks or sorted(bundle["models"]); d, F = build_features(d)
    hs = sorted({bundle["models"][k]["b"] for k in keys}); need_sar = any((model or bundle["models"][k]["best"]) == "hybrid" for k in keys)
    common, per_h = sarima_frame(d, bundle, hs) if need_sar else (None, None)
    last = d.groupby("Batch_ID").tail(1).index if latest_only else d.index; rows = []
    for k in keys:
        mb = bundle["models"][k]; h = mb["b"]; which = model or mb["best"]; spec = mb["candidates"][which]
        X = F.loc[last, bundle["t_features"]].copy(); X["target_hour"] = (d.loc[last, "Timestamp"].dt.hour.to_numpy() + h) % 24
        if which == "hybrid": X = pd.concat([X, common.loc[last, bundle["sarima"]["residual_cols"]], per_h[h].loc[last, bundle["sarima"]["forecast_cols"]]], axis=1)
        X = X[spec["features"]].astype(np.float32); p_raw = spec["model"].predict_proba(X)[:, 1]; p_cal = spec["calibrator"].transform(p_raw)
        thr = spec["thresholds"][operating_point]; normal = d.loc[last, ANOMALY_FLAG_COL].to_numpy() == 0
        for j, ix in enumerate(last):
            rows.append(dict(Batch_ID=d.at[ix, "Batch_ID"], origin_time=d.at[ix, "Timestamp"], status="NORMAL_ORIGIN" if normal[j] else "ALREADY_EXCURSION", task=k, window=f"t+{mb['a']}..t+{mb['b']}", model=which,
                             p_raw=float(p_raw[j]) if normal[j] else np.nan, p_cal=float(p_cal[j]) if normal[j] else np.nan, threshold=thr, alert=bool(normal[j] and thr is not None and p_cal[j] >= thr)))
    return pd.DataFrame(rows)