"""
SIGNAL·WATCH — Team Diamonds Backend v3.0
Merges the improved detection engine (main.py) into Flask.

Key fixes:
  - Adaptive percentile threshold (top 5%) — fixes everything-CRITICAL noise
  - Full feature engineering pipeline matching training notebook exactly  
  - Architecture auto-detected from checkpoint weight shapes
  - SHAP per anomaly window (graceful fallback)
  - Groq LLM advisory → Anthropic fallback → rule-based fallback
  - /anomalies returns flat list (dashboard compatible)
  - /stats and /health endpoints added

Run:
    python app.py
Requires .env with GROQ_API_KEY=gsk_...
"""

import os, io, json, logging, traceback, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "hai_models"
API_KEY    = os.environ.get("API_SECRET", "Team_Dimonds2026_xyzdaiict")
SEQ_LEN    = 60

# ── Model Architectures ───────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=128, latent=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, hidden, n_layers,
                                    batch_first=True, dropout=dropout)
        self.bottleneck   = nn.Sequential(nn.Linear(hidden, latent), nn.ReLU())
        self.expand       = nn.Linear(latent, hidden)
        self.decoder_lstm = nn.LSTM(hidden, n_features, n_layers,
                                    batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        z   = self.bottleneck(h[-1])
        h2  = self.expand(z).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.decoder_lstm(h2)
        return out, z

    def reconstruction_error(self, x):
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)


class TransformerAE(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        return self.output_proj(self.encoder(self.input_proj(x)))

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)


# ── PCL Graph ─────────────────────────────────────────────────────────────────
PCL_GRAPH = {
    "P1-PC": {"sp":"P1_B2016",  "pv":"P1_PIT01", "cv":["P1_PCV01D","P1_PCV02D"],
               "downstream":["P4_ST_PT01","P4_ST_PO"],  "description":"Boiler pressure control"},
    "P1-LC": {"sp":"P1_B3004",  "pv":"P1_LIT01", "cv":["P1_LCV01D"],
               "downstream":["P1_FT01Z","P1_FT03Z"],    "description":"Boiler water level control"},
    "P1-FC": {"sp":"P1_B3005",  "pv":"P1_FT03",  "cv":["P1_FCV03D"],
               "downstream":["P1_LIT01"],               "description":"Boiler flow rate control"},
    "P1-TC": {"sp":"P1_B4022",  "pv":"P1_TIT01", "cv":["P1_FCV01D","P1_FCV02D"],
               "downstream":["P4_ST_TT01"],             "description":"Boiler temperature control"},
    "P1-CC": {"sp":"P1_PP04SP", "pv":"P1_TIT03", "cv":["P1_PP04"],
               "downstream":[],                         "description":"Boiler cooling control"},
    "P2-SC": {"sp":"P2_AutoSD", "pv":"P2_SIT01", "cv":["P2_SCO"],
               "downstream":["P4_ST_PO"],               "description":"Turbine speed control"},
    "P3-LC": {"sp":"P3_LH01",   "pv":"P3_LIT01", "cv":["P3_LCV01D","P3_LCP01D"],
               "downstream":["P4_HT_PO"],               "description":"Water treatment level control"},
}

SENSOR_ROLE_MAP = {}
for _loop, _cfg in PCL_GRAPH.items():
    SENSOR_ROLE_MAP[_cfg["sp"]] = f"SP ({_loop})"
    SENSOR_ROLE_MAP[_cfg["pv"]] = f"PV ({_loop})"
    for _cv in _cfg["cv"]:              SENSOR_ROLE_MAP[_cv] = f"CV ({_loop})"
    for _ds in _cfg.get("downstream",[]): SENSOR_ROLE_MAP[_ds] = f"downstream ({_loop})"


# ── Global state ──────────────────────────────────────────────────────────────
class State:
    lstm:           Optional[nn.Module] = None
    transformer:    Optional[nn.Module] = None
    iso_forest:     Optional[object]    = None
    scalers:        Optional[dict]      = None
    feature_cols:   list                = []
    n_features:     int                 = 0
    is_ready:       bool                = False
    last_anomalies: list                = []
    last_stats:     dict                = {}

S = State()


def load_models():
    try:
        log.info("Loading from %s ...", MODELS_DIR)

        # Feature columns
        for path in [MODELS_DIR/"feature_cols.json", MODELS_DIR/"metadata.json"]:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                S.feature_cols = data if isinstance(data, list) else data["feature_cols"]
                break
        if not S.feature_cols:
            raise FileNotFoundError("No feature_cols.json or metadata.json found")
        S.n_features = len(S.feature_cols)
        log.info("  [OK] %d features", S.n_features)

        # Scalers
        raw = joblib.load(MODELS_DIR / "scalers.pkl")
        S.scalers = raw if isinstance(raw, dict) else {"main": raw}
        log.info("  [OK] scaler")

        # IsoForest
        S.iso_forest = joblib.load(MODELS_DIR / "iso_forest.pkl")
        log.info("  [OK] IsolationForest")

        # LSTM — auto-detect dims from checkpoint
        ckpt = torch.load(MODELS_DIR/"lstm_ae_full.pt", map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "n_features" in ckpt:
            nf = ckpt["n_features"]; hid = ckpt.get("hidden",128)
            lat = ckpt.get("latent",32); nl = ckpt.get("n_layers",2)
            sd = ckpt["model_state"]
        else:
            sd = ckpt.get("model_state", ckpt)
            hid = sd["encoder_lstm.weight_ih_l0"].shape[0] // 4
            nf  = sd["encoder_lstm.weight_ih_l0"].shape[1]
            bkey = "bottleneck.0.weight" if "bottleneck.0.weight" in sd else "bottleneck.weight"
            lat = sd[bkey].shape[0]
            nl  = 2
        S.lstm = LSTMAutoencoder(n_features=nf, hidden=hid, latent=lat, n_layers=nl)
        S.lstm.load_state_dict(sd, strict=False); S.lstm.eval()
        log.info("  [OK] LSTM-AE (n=%d h=%d l=%d)", nf, hid, lat)

        # Transformer — auto-detect
        tckpt = torch.load(MODELS_DIR/"transformer_ae_full.pt", map_location="cpu", weights_only=False)
        tsd   = tckpt.get("model_state", tckpt) if isinstance(tckpt, dict) else tckpt
        dm    = tsd["input_proj.weight"].shape[0]
        nft   = tsd["input_proj.weight"].shape[1]
        nl_t  = sum(1 for k in tsd if k.startswith("encoder.layers.") and k.endswith("norm1.weight"))
        nh    = tckpt.get("nhead", 4) if isinstance(tckpt, dict) else 4
        while dm % nh != 0 and nh > 1: nh -= 1
        dff   = tsd["encoder.layers.0.linear1.weight"].shape[0]
        S.transformer = TransformerAE(n_features=nft, d_model=dm, nhead=nh,
                                       num_layers=nl_t, dim_feedforward=dff)
        S.transformer.load_state_dict(tsd, strict=False); S.transformer.eval()
        log.info("  [OK] Transformer-AE (dm=%d nh=%d nl=%d ff=%d)", dm, nh, nl_t, dff)

        S.is_ready = True
        log.info("All models ready.")
    except Exception as e:
        log.error("Load failed: %s", e); traceback.print_exc()


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    cols = df.columns.tolist()
    nc   = {}

    def sd(a, b):
        return (df[a]-df[b]) if a in cols and b in cols else pd.Series(0., index=df.index)
    def sad(a, b): return sd(a, b).abs()

    nc["feat_P1PC_err"]    = sd("P1_B2016",  "P1_PIT01")
    nc["feat_P1PC_cv_dev"] = sad("P1_PCV01D","P1_PCV01Z")
    nc["feat_P1LC_err"]    = sd("P1_B3004",  "P1_LIT01")
    nc["feat_P1LC_cv_dev"] = sad("P1_LCV01D","P1_LCV01Z")
    nc["feat_P1FC_err"]    = sd("P1_B3005",  "P1_FT03Z")
    nc["feat_P1FC_cv_dev"] = sad("P1_FCV03D","P1_FCV03Z")
    nc["feat_P1TC_err"]    = sd("P1_B4022",  "P1_TIT01")
    nc["feat_P2SC_err"]    = sd("P2_AutoSD", "P2_SIT01")
    nc["feat_P2SC_cv_dev"] = (df["P2_SCO"]-df["P2_SIT01"]).abs()         if "P2_SCO" in cols and "P2_SIT01" in cols else pd.Series(0., index=df.index)

    roc_pfx = ("P1_B","P2_AutoSD","P1_PIT","P1_LIT","P1_FT","P2_SIT")
    for col in [c for c in cols if c.startswith(roc_pfx)]:
        nc[f"feat_roc_{col}"] = df[col].diff(1).fillna(0)

    key_s = [c for c in cols if c.startswith(("P1_PIT","P1_LIT","P1_FT","P2_SIT","P1_TIT","P3_LIT"))][:12]
    for col in key_s:
        for w in [10,30,60]:
            r = df[col].rolling(w, min_periods=1)
            nc[f"feat_{col}_mean{w}"] = r.mean()
            nc[f"feat_{col}_std{w}"]  = r.std().fillna(0)
            nc[f"feat_{col}_max{w}"]  = r.max()

    for dc in [c for c in cols if c.endswith("D") and c[:-1]+"Z" in cols]:
        nc[f"feat_valve_{dc}"] = (df[dc]-df[dc[:-1]+"Z"]).abs()

    return pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1).fillna(0)


def preprocess(df):
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.interpolate(method="linear").ffill().bfill()

    main_scaler = S.scalers.get("main") or next(iter(S.scalers.values()))
    scaler_cols = main_scaler.feature_names_in_.tolist()         if hasattr(main_scaler, "feature_names_in_") else         [c for c in S.feature_cols if c in df.columns]

    shared = [c for c in scaler_cols if c in df.columns]
    if not shared: return None

    df[shared] = main_scaler.transform(df[shared])
    df = engineer_features(df)
    df = df.reindex(columns=S.feature_cols, fill_value=0.0)
    return df.values.astype(np.float32)


# ── Scoring helpers ───────────────────────────────────────────────────────────
def normalize_01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def severity_label(score):
    if score > 0.85: return "CRITICAL"
    if score > 0.70: return "HIGH"
    if score > 0.50: return "MEDIUM"
    return "LOW"

def find_top_loop(sensor_names):
    sc = {loop: len(set(sensor_names) & set([cfg["sp"],cfg["pv"]]+cfg["cv"]+cfg.get("downstream",[])))
          for loop, cfg in PCL_GRAPH.items()}
    return max(sc, key=sc.get)

def classify_attack(sensor_names, cfg):
    sp = cfg["sp"] in sensor_names
    pv = cfg["pv"] in sensor_names
    cv = any(c in sensor_names for c in cfg["cv"])
    if sp and pv and cv: return "compound_attack"
    if sp and pv:        return "SP_manipulation_with_PV_spoofing"
    if sp:               return "SP_manipulation"
    if pv and cv:        return "CV_injection_with_PV_spoofing"
    if cv:               return "CV_injection"
    if pv:               return "PV_spoofing"
    return "indirect_effect"


# ── LLM Advisory ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert ICS security analyst for the HAI testbed (steam-turbine + pumped-storage). "
    "Loops: P1-PC=Boiler pressure, P1-LC=Boiler level, P1-FC=Boiler flow, P1-TC=Boiler temp, "
    "P2-SC=Turbine speed, P3-LC=Water treatment. D=Demand, Z=Position. Large D-Z = CV injection. "
    "Write a 4-sentence operator advisory: what happened, which subsystem, attack scenario, immediate action."
)

def llm_advisory(record):
    sensor_names = [s["name"] if isinstance(s, dict) else s for s in record.get("top_sensors", [])]

    # Try Groq first
    if os.environ.get("GROQ_API_KEY"):
        try:
            from groq import Groq
            sensors_str = ", ".join(sensor_names[:5])
            shap_str    = "; ".join(
                f"{s['feature']}={s['value']:+.4f}" for s in record.get("shap",[])[:4]
            ) or "n/a"
            prompt = (
                f"Loop: {record['top_loop']} ({record.get('loop_description','')}), "
                f"Attack: {record['attack_type']}, Score: {record['anomaly_score']:.4f}, "
                f"Severity: {record['severity']}\n"
                f"Top sensors: {sensors_str}\nSHAP: {shap_str}\n"
                f"Write a 4-sentence operator advisory."
            )
            res = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
                model="llama-3.1-8b-instant", max_tokens=500, temperature=0.2,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":prompt}]
            )
            log.info("Advisory via Groq")
            return res.choices[0].message.content
        except Exception as e:
            log.warning("Groq failed: %s", e)

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            prompt = (
                f"ICS anomaly: Loop={record['top_loop']}, Attack={record['attack_type']}, "
                f"Score={record['anomaly_score']:.4f}, Severity={record['severity']}, "
                f"Sensors={', '.join(sensor_names[:5])}. 3-4 sentence operator advisory."
            )
            msg = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=400,
                messages=[{"role":"user","content":prompt}]
            )
            log.info("Advisory via Anthropic")
            return msg.content[0].text
        except Exception as e:
            log.warning("Anthropic failed: %s", e)

    return _rule_based_advisory(record, sensor_names)


def _rule_based_advisory(record, sensor_names=None):
    if sensor_names is None:
        sensor_names = [s["name"] if isinstance(s,dict) else s
                        for s in record.get("top_sensors",[])]
    loop     = record["top_loop"]
    loop_cfg = PCL_GRAPH.get(loop, {})
    desc     = loop_cfg.get("description", f"control loop {loop}")
    attack   = record["attack_type"].replace("_", " ")
    severity = record["severity"]
    score    = record["anomaly_score"]
    sensors  = ", ".join(sensor_names[:3]) or "unidentified sensors"
    ds       = ", ".join(loop_cfg.get("downstream", []) or ["adjacent subsystems"])
    action   = {
        "CRITICAL": "Initiate emergency shutdown and alert shift supervisor immediately.",
        "HIGH":     "Isolate affected subsystem and switch to manual control.",
        "MEDIUM":   "Increase monitoring frequency and verify sensor calibration.",
        "LOW":      "Log the event and review at next scheduled inspection.",
    }.get(severity, "Review sensor readings against historical baselines.")
    return (
        f"The ensemble models detected a {severity.lower()} anomaly (score {score:.3f}) "
        f"in the {desc}, consistent with a {attack} pattern. "
        f"Primary contributing sensors are {sensors}, showing significant deviation from "
        f"learned normal operating envelopes. "
        f"The {loop} loop is at risk — downstream effects may propagate to {ds}. "
        f"{action}"
    )


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":"*"}})

def check_key():
    if request.headers.get("x-api-key","") != API_KEY:
        return jsonify({"error":"Unauthorized"}), 401

@app.route("/")
def serve_dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if S.is_ready else "not ready",
        "models": ["LSTM-AE","Transformer-AE","IsolationForest"],
        "n_features": S.n_features,
        "seq_len": SEQ_LEN,
        "anomalies_stored": len(S.last_anomalies),
    })

@app.route("/upload", methods=["POST"])
def upload():
    err = check_key()
    if err: return err
    if not S.is_ready:
        return jsonify({"error":"Models not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error":"No file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error":"CSV only"}), 400

    try:
        df = pd.read_csv(io.StringIO(f.stream.read().decode("utf-8", errors="replace")))
    except Exception as e:
        return jsonify({"error":f"Cannot parse CSV: {e}"}), 400

    log.info("CSV: %d rows x %d cols", *df.shape)
    missing = [c for c in ["P1_FCV01D","P1_LIT01","P1_PIT01"] if c not in df.columns]
    if missing:
        return jsonify({"error":f"Missing required columns: {missing}"}), 400

    arr = preprocess(df)
    if arr is None:
        return jsonify({"error":"Preprocessing failed — no matching sensor columns"}), 400

    n_rows = arr.shape[0]
    if n_rows < SEQ_LEN + 1:
        return jsonify({"error":f"CSV too short: {n_rows} rows, need >= {SEQ_LEN+1}"}), 400

    # Build windows (step=1 for dense coverage)
    wins_np = np.stack([arr[i:i+SEQ_LEN] for i in range(n_rows - SEQ_LEN)])
    wins    = torch.tensor(wins_np, dtype=torch.float32)
    n_wins  = len(wins)
    log.info("Windows: %d", n_wins)

    # Score all models in chunks
    CHUNK = 512
    lstm_e, trans_e = [], []
    with torch.no_grad():
        for i in range(0, n_wins, CHUNK):
            c = wins[i:i+CHUNK]
            lstm_e.append(S.lstm.reconstruction_error(c).cpu().numpy().mean(axis=1))
            trans_e.append(S.transformer.reconstruction_error(c).cpu().numpy().mean(axis=1))

    lstm_err  = np.concatenate(lstm_e)
    trans_err = np.concatenate(trans_e)
    iso_raw   = -S.iso_forest.score_samples(arr)
    iso_err   = iso_raw[SEQ_LEN: SEQ_LEN + n_wins]

    min_len   = min(len(lstm_err), len(trans_err), len(iso_err))
    lstm_err  = lstm_err[:min_len]
    trans_err = trans_err[:min_len]
    iso_err   = iso_err[:min_len]

    # Weighted ensemble with per-model normalization
    ensemble  = (0.5 * normalize_01(lstm_err) +
                 0.3 * normalize_01(trans_err) +
                 0.2 * normalize_01(iso_err))

    # ADAPTIVE threshold — top 5% only
    threshold = float(np.percentile(ensemble, 95))
    anomaly_idx = np.where(ensemble > threshold)[0]
    log.info("Threshold=%.4f  Anomalies=%d/%d", threshold, len(anomaly_idx), min_len)

    # SHAP (optional)
    shap_values = None
    try:
        import shap as shap_lib
        explainer   = shap_lib.TreeExplainer(S.iso_forest)
        shap_input  = arr[SEQ_LEN: SEQ_LEN + min_len]
        shap_values = explainer.shap_values(shap_input)
        log.info("SHAP computed")
    except Exception as e:
        log.warning("SHAP skipped: %s", e)

    anomalies = []
    for idx in anomaly_idx:
        idx = int(idx)

        # Per-feature error from LSTM
        with torch.no_grad():
            wt       = wins[idx].unsqueeze(0)
            recon, _ = S.lstm(wt)
            recon_np = recon.squeeze(0).cpu().numpy()
        feat_err = np.mean((wins_np[idx] - recon_np)**2, axis=0)

        top_ids  = np.argsort(feat_err)[::-1][:8]
        top_sens = [S.feature_cols[i] for i in top_ids if i < S.n_features]

        shap_top = []
        if shap_values is not None and idx < len(shap_values):
            sr   = shap_values[idx]
            sidx = np.argsort(np.abs(sr))[::-1][:6]
            shap_top = [{"feature":S.feature_cols[j],"value":round(float(sr[j]),5),
                          "direction":"toward anomaly" if sr[j]>0 else "toward normal"}
                        for j in sidx if j < S.n_features]

        top_loop = find_top_loop(top_sens)
        cfg      = PCL_GRAPH[top_loop]
        attack   = classify_attack(top_sens, cfg)
        score    = float(ensemble[idx])

        ts_row = SEQ_LEN + idx
        if "timestamp" in df.columns and ts_row < len(df):
            ts = str(df.iloc[ts_row]["timestamp"])
        else:
            ts = str(ts_row)

        anomalies.append({
            "window_idx":       idx,
            "anomaly_score":    round(score, 4),
            "severity":         severity_label(score),
            "top_loop":         top_loop,
            "loop_description": cfg["description"],
            "attack_type":      attack,
            "timestamp":        ts,
            "top_sensors": [{"name":s,
                              "error":round(float(feat_err[S.feature_cols.index(s)]),6),
                              "role":SENSOR_ROLE_MAP.get(s,"general sensor")}
                            for s in top_sens],
            "shap": shap_top,
        })

    S.last_anomalies = anomalies
    sev_c = {}; loop_c = {}
    for r in anomalies:
        sev_c[r["severity"]]  = sev_c.get(r["severity"],0) + 1
        loop_c[r["top_loop"]] = loop_c.get(r["top_loop"],0) + 1
    S.last_stats = {"rows_uploaded":df.shape[0],"rows_processed":n_rows,
                    "total_windows":int(min_len),"threshold":round(threshold,4),
                    "anomalies_found":len(anomalies),"by_severity":sev_c,"by_loop":loop_c}

    log.info("Done — %d anomalies", len(anomalies))
    return jsonify({"anomalies":len(anomalies),"windows_scanned":int(min_len),
                    "threshold":round(threshold,4)})


@app.route("/anomalies", methods=["GET"])
def anomalies():
    err = check_key()
    if err: return err
    records = S.last_anomalies
    sev  = request.args.get("severity","").upper()
    loop = request.args.get("loop","").upper()
    lim  = int(request.args.get("limit",200))
    if sev:  records = [r for r in records if r["severity"] == sev]
    if loop: records = [r for r in records if r["top_loop"] == loop]
    records = sorted(records, key=lambda x: -x["anomaly_score"])[:lim]
    # Flatten top_sensors to name list for dashboard compatibility
    flat = [{**r, "top_sensors":[s["name"] if isinstance(s,dict) else s
                                  for s in r.get("top_sensors",[])]}
            for r in records]
    return jsonify(flat)


@app.route("/explain", methods=["POST"])
def explain():
    err = check_key()
    if err: return err
    body = request.get_json(force=True, silent=True) or {}
    widx = body.get("window_idx")
    record = next((r for r in S.last_anomalies if r["window_idx"]==widx), None)
    if record is None and S.last_anomalies:
        record = S.last_anomalies[0]
    if record is None:
        return jsonify({"advisory":"No anomaly data. Upload a CSV first."})
    return jsonify({
        "window_idx":  record["window_idx"],
        "severity":    record["severity"],
        "top_loop":    record["top_loop"],
        "attack_type": record["attack_type"],
        "score":       record["anomaly_score"],
        "advisory":    llm_advisory(record),
    })


@app.route("/stats", methods=["GET"])
def stats():
    err = check_key()
    if err: return err
    if not S.last_anomalies:
        return jsonify({"message":"No anomalies yet — POST CSV to /upload"})
    return jsonify(S.last_stats)


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
