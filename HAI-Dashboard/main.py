# ================================================================
# HAI ANOMALY DETECTION BACKEND — COMPLETE FIXED VERSION
# ================================================================
# Run with: uvicorn main:app --reload
# Requires: hai_models/ folder with:
#   - lstm_ae_full.pt
#   - transformer_ae_full.pt
#   - iso_forest.pkl
#   - scalers.pkl
#   - feature_cols.json
# ================================================================

import os, json, warnings, torch, joblib
import numpy as np
import pandas as pd
import shap
import torch.nn as nn
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# ================================================================
# ENV
# ================================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_SECRET   = os.getenv("API_SECRET")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")
if not API_SECRET:
    raise ValueError("API_SECRET missing in .env")

# ================================================================
# CONFIG
# ================================================================

MODEL_DIR = "hai_models"
SEQ_LEN   = 60
DEVICE    = torch.device("cpu")

# ================================================================
# LOAD SAVED FILES
# ================================================================

with open(f"{MODEL_DIR}/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

N_FEATURES = len(FEATURE_COLS)
print(f"Feature cols loaded: {N_FEATURES} features")

scalers    = joblib.load(f"{MODEL_DIR}/scalers.pkl")
iso_forest = joblib.load(f"{MODEL_DIR}/iso_forest.pkl")
print("Scalers and IsolationForest loaded.")

# ================================================================
# MODEL DEFINITIONS
# Must exactly match the architecture used during training
# ================================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=128, latent=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True, dropout=dropout)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, latent),
            nn.ReLU())
        self.expand       = nn.Linear(latent, hidden)
        self.decoder_lstm = nn.LSTM(
            hidden, n_features, n_layers,
            batch_first=True, dropout=dropout)

    def forward(self, x):
        enc_out, (h_n, _) = self.encoder_lstm(x)
        z    = self.bottleneck(h_n[-1])
        h2   = self.expand(z).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.decoder_lstm(h2)
        return out, z

    def reconstruction_error(self, x):
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)   # (batch, n_features)


class TransformerAE(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h     = self.input_proj(x)
        h     = self.encoder(h)
        return self.output_proj(h)

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)   # (batch, n_features)


# ================================================================
# LOAD MODEL WEIGHTS
# Auto-detects architecture from checkpoint weight shapes
# so it always matches what was actually trained
# ================================================================

# ── LSTM
lstm_ckpt  = torch.load(f"{MODEL_DIR}/lstm_ae_full.pt", map_location="cpu")

lstm_model = LSTMAutoencoder(
    n_features=lstm_ckpt["n_features"],
    hidden=lstm_ckpt.get("hidden", 128),
    latent=lstm_ckpt.get("latent", 32),
    n_layers=lstm_ckpt.get("n_layers", 2),
)
lstm_model.load_state_dict(lstm_ckpt["model_state"])
lstm_model.eval()
print(f"LSTM-AE loaded. n_features={lstm_ckpt['n_features']}")

# ── Transformer — detect exact architecture from saved weight shapes
trans_ckpt = torch.load(f"{MODEL_DIR}/transformer_ae_full.pt", map_location="cpu")
state      = trans_ckpt["model_state"]

# input_proj.weight shape: (d_model, n_features)
d_model_actual    = state["input_proj.weight"].shape[0]
n_features_actual = state["input_proj.weight"].shape[1]

# count encoder layers by counting norm1.weight keys
num_layers_actual = sum(
    1 for k in state
    if k.startswith("encoder.layers.") and k.endswith("norm1.weight")
)

# nhead — divide d_model by head_dim inferred from in_proj_weight
# in_proj_weight shape: (3*d_model, d_model) → 3*nhead*head_dim
# Try nhead values that divide d_model evenly, pick largest valid one
nhead_actual = trans_ckpt.get("nhead", 4)
while d_model_actual % nhead_actual != 0 and nhead_actual > 1:
    nhead_actual -= 1

# dim_feedforward from linear1.weight: (dim_ff, d_model)
dim_ff_actual = state["encoder.layers.0.linear1.weight"].shape[0]

print(f"Transformer checkpoint detected:")
print(f"  d_model={d_model_actual}, nhead={nhead_actual}, "
      f"num_layers={num_layers_actual}, dim_feedforward={dim_ff_actual}, "
      f"n_features={n_features_actual}")

transformer_model = TransformerAE(
    n_features=n_features_actual,
    d_model=d_model_actual,
    nhead=nhead_actual,
    num_layers=num_layers_actual,
    dim_feedforward=dim_ff_actual,
)
transformer_model.load_state_dict(state)
transformer_model.eval()
print("Transformer-AE loaded successfully.")

# ================================================================
# SHAP EXPLAINER
# ================================================================

explainer = shap.TreeExplainer(iso_forest)
print("SHAP TreeExplainer ready.")

# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(
    title="HAI Anomaly Detection API",
    description="Ensemble anomaly detection + SHAP + LLM advisory for HAI ICS dataset",
    version="2.0.0"
)

# ── API key middleware
@app.middleware("http")
async def check_api_key(request, call_next):
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store — populated by /upload
score_records: list = []


class ExplainRequest(BaseModel):
    window_idx: int


# ================================================================
# PCL GRAPH — directly from HAI paper (Figure 3)
# ================================================================

PCL_GRAPH = {
    "P1-PC": {
        "sp":          "P1_B2016",
        "pv":          "P1_PIT01",
        "cv":          ["P1_PCV01D", "P1_PCV02D"],
        "downstream":  ["P4_ST_PT01", "P4_ST_PO"],
        "description": "Boiler pressure control",
    },
    "P1-LC": {
        "sp":          "P1_B3004",
        "pv":          "P1_LIT01",
        "cv":          ["P1_LCV01D"],
        "downstream":  ["P1_FT01Z", "P1_FT03Z"],
        "description": "Boiler water level control",
    },
    "P1-FC": {
        "sp":          "P1_B3005",
        "pv":          "P1_FT03",
        "cv":          ["P1_FCV03D"],
        "downstream":  ["P1_LIT01"],
        "description": "Boiler flow rate control",
    },
    "P1-TC": {
        "sp":          "P1_B4022",
        "pv":          "P1_TIT01",
        "cv":          ["P1_FCV01D", "P1_FCV02D"],
        "downstream":  ["P4_ST_TT01"],
        "description": "Boiler temperature control",
    },
    "P1-CC": {
        "sp":          "P1_PP04SP",
        "pv":          "P1_TIT03",
        "cv":          ["P1_PP04"],
        "downstream":  [],
        "description": "Boiler cooling control",
    },
    "P2-SC": {
        "sp":          "P2_AutoSD",
        "pv":          "P2_SIT01",
        "cv":          ["P2_SCO"],
        "downstream":  ["P4_ST_PO"],
        "description": "Turbine speed control",
    },
    "P3-LC": {
        "sp":          "P3_LH01",
        "pv":          "P3_LIT01",
        "cv":          ["P3_LCV01D", "P3_LCP01D"],
        "downstream":  ["P4_HT_PO"],
        "description": "Water treatment level control",
    },
}

SENSOR_ROLE_MAP = {}
for loop, cfg in PCL_GRAPH.items():
    SENSOR_ROLE_MAP[cfg["sp"]] = f"SP ({loop})"
    SENSOR_ROLE_MAP[cfg["pv"]] = f"PV ({loop})"
    for cv in cfg["cv"]:
        SENSOR_ROLE_MAP[cv] = f"CV ({loop})"
    for ds in cfg.get("downstream", []):
        SENSOR_ROLE_MAP[ds] = f"downstream ({loop})"


# ================================================================
# FEATURE ENGINEERING
# Must exactly match Cell 5 in the training notebook
# ================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all engineered features.
    Uses dict + single pd.concat to avoid DataFrame fragmentation warning.
    """
    cols     = df.columns.tolist()
    new_cols = {}

    def safe_diff(a, b):
        if a in cols and b in cols:
            return df[a] - df[b]
        return pd.Series(0.0, index=df.index)

    def safe_abs_diff(a, b):
        return safe_diff(a, b).abs()

    # ── PCL setpoint-process error features
    new_cols["feat_P1PC_err"]    = safe_diff("P1_B2016",  "P1_PIT01")
    new_cols["feat_P1PC_cv_dev"] = safe_abs_diff("P1_PCV01D", "P1_PCV01Z")
    new_cols["feat_P1LC_err"]    = safe_diff("P1_B3004",  "P1_LIT01")
    new_cols["feat_P1LC_cv_dev"] = safe_abs_diff("P1_LCV01D", "P1_LCV01Z")
    new_cols["feat_P1FC_err"]    = safe_diff("P1_B3005",  "P1_FT03Z")
    new_cols["feat_P1FC_cv_dev"] = safe_abs_diff("P1_FCV03D", "P1_FCV03Z")
    new_cols["feat_P1TC_err"]    = safe_diff("P1_B4022",  "P1_TIT01")
    new_cols["feat_P2SC_err"]    = safe_diff("P2_AutoSD", "P2_SIT01")
    new_cols["feat_P2SC_cv_dev"] = (
        (df["P2_SCO"] - df["P2_SIT01"]).abs()
        if "P2_SCO" in cols and "P2_SIT01" in cols
        else pd.Series(0.0, index=df.index)
    )

    # ── Rate-of-change (first derivative)
    roc_prefixes = ("P1_B", "P2_AutoSD", "P1_PIT", "P1_LIT", "P1_FT", "P2_SIT")
    for col in [c for c in cols if c.startswith(roc_prefixes)]:
        new_cols[f"feat_roc_{col}"] = df[col].diff(1).fillna(0)

    # ── Rolling statistics (10s, 30s, 60s windows)
    stat_prefixes = ("P1_PIT", "P1_LIT", "P1_FT", "P2_SIT", "P1_TIT", "P3_LIT")
    key_sensors   = [c for c in cols if c.startswith(stat_prefixes)][:12]

    for col in key_sensors:
        for w in [10, 30, 60]:
            r = df[col].rolling(w, min_periods=1)
            new_cols[f"feat_{col}_mean{w}"] = r.mean()
            new_cols[f"feat_{col}_std{w}"]  = r.std().fillna(0)
            new_cols[f"feat_{col}_max{w}"]  = r.max()

    # ── Valve demand vs position deviation (D suffix vs Z suffix)
    for d_col in [c for c in cols if c.endswith("D") and c[:-1] + "Z" in cols]:
        z_col = d_col[:-1] + "Z"
        new_cols[f"feat_valve_{d_col}"] = (df[d_col] - df[z_col]).abs()

    # ── Single concat — no fragmentation
    result = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)],
        axis=1
    )
    return result.fillna(0)


# ================================================================
# PREPROCESS
# ================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline matching the training notebook exactly.
    Input:  raw sensor CSV DataFrame
    Output: DataFrame with exactly FEATURE_COLS columns (N_FEATURES = 221)
    """

    # Step 1 — drop non-numeric (timestamp column etc)
    df = df.select_dtypes(include=[np.number]).copy()
    print(f"[preprocess] After numeric select: {df.shape}")

    # Step 2 — interpolate gaps
    df = df.interpolate(method="linear").ffill().bfill()

    # Step 3 — scale raw sensor columns using the fitted scaler
    scaler_cols = scalers["main"].feature_names_in_.tolist()
    shared      = [c for c in scaler_cols if c in df.columns]

    if not shared:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No matching sensor columns between CSV and scaler. "
                f"CSV has {df.shape[1]} columns: {df.columns[:8].tolist()}... "
                f"Scaler expects columns like: {scaler_cols[:8]}"
            )
        )

    print(f"[preprocess] Scaling {len(shared)}/{len(scaler_cols)} sensor columns")
    df[shared] = scalers["main"].transform(df[shared])

    # Step 4 — feature engineering (expands from ~87 raw → ~221 total)
    df = engineer_features(df)
    print(f"[preprocess] After feature engineering: {df.shape[1]} columns")

    # Step 5 — align to exact training feature set
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[preprocess] Filling {len(missing)} missing features with 0: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    df = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
    print(f"[preprocess] Final shape: {df.shape}  "
          f"(expected: (N, {N_FEATURES}))")

    return df


# ================================================================
# HELPERS
# ================================================================

def normalize_01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def create_windows(arr: np.ndarray) -> torch.Tensor:
    """Create sliding windows of shape (N-SEQ_LEN, SEQ_LEN, N_FEATURES)."""

    if arr.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature shape mismatch before windowing: "
                f"got {arr.shape[1]} features, models expect {N_FEATURES}. "
                f"Feature engineering may have failed — check CSV column names."
            )
        )

    if len(arr) < SEQ_LEN + 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV too short: {len(arr)} rows after preprocessing, "
                f"need at least {SEQ_LEN + 1} rows."
            )
        )

    windows = np.stack([arr[i:i + SEQ_LEN] for i in range(len(arr) - SEQ_LEN)])
    return torch.tensor(windows, dtype=torch.float32)


# ================================================================
# ROOT CAUSE + ATTACK CLASSIFICATION
# ================================================================

def classify_attack(sensors: list, cfg: dict) -> str:
    has_sp = cfg["sp"] in sensors
    has_pv = cfg["pv"] in sensors
    has_cv = any(c in sensors for c in cfg["cv"])

    if has_sp and has_pv and has_cv: return "compound_attack"
    if has_sp and has_pv:            return "SP_manipulation_with_PV_spoofing"
    if has_sp:                       return "SP_manipulation"
    if has_pv and has_cv:            return "CV_injection_with_PV_spoofing"
    if has_cv:                       return "CV_injection"
    if has_pv:                       return "PV_spoofing"
    return "indirect_effect"


def find_top_loop(top_sensors: list) -> str:
    loop_scores = {}
    for loop, cfg in PCL_GRAPH.items():
        all_nodes = [cfg["sp"], cfg["pv"]] + cfg["cv"] + cfg.get("downstream", [])
        loop_scores[loop] = len(set(top_sensors) & set(all_nodes))
    return max(loop_scores, key=loop_scores.get)


def build_context(idx: int, score: float,
                  feature_err: np.ndarray, shap_top: list) -> dict:
    # Top sensors by reconstruction error
    top_err_idx = np.argsort(feature_err)[::-1][:8]
    top_sensors = [FEATURE_COLS[i] for i in top_err_idx if i < len(FEATURE_COLS)]

    top_loop    = find_top_loop(top_sensors)
    loop_cfg    = PCL_GRAPH[top_loop]
    attack_type = classify_attack(top_sensors, loop_cfg)

    severity = (
        "CRITICAL" if score > 0.85 else
        "HIGH"     if score > 0.70 else
        "MEDIUM"   if score > 0.50 else
        "LOW"
    )

    return {
        "window_idx":       int(idx),
        "anomaly_score":    round(float(score), 4),
        "severity":         severity,
        "top_loop":         top_loop,
        "loop_description": loop_cfg["description"],
        "attack_type":      attack_type,
        "top_sensors":      [
            {
                "name":  s,
                "error": round(float(feature_err[FEATURE_COLS.index(s)]), 6)
                         if s in FEATURE_COLS else 0.0,
                "role":  SENSOR_ROLE_MAP.get(s, "general sensor"),
            }
            for s in top_sensors
        ],
        "shap": shap_top,
    }


# ================================================================
# LLM — Groq (free)
# ================================================================

groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert ICS security analyst for the HAI testbed —
a combined steam-turbine power generation and pumped-storage hydropower system.

Process control loops (PCLs):
- P1-PC: Boiler pressure   | SP: B2016  → CV: PCV01D/02D → PV: PIT01
- P1-LC: Boiler level      | SP: B3004  → CV: LCV01D     → PV: LIT01
- P1-FC: Boiler flow       | SP: B3005  → CV: FCV03D     → PV: FT03
- P1-TC: Boiler temp       | SP: B4022  → CV: FCV01D/02D → PV: TIT01
- P1-CC: Boiler cooling    | SP: PP04SP → CV: PP04       → PV: TIT03
- P2-SC: Turbine speed     | SP: AutoSD → CV: SCO        → PV: SIT01
- P3-LC: Water level       | SP: LH01   → CV: LCV01D/LCP01D → PV: LIT01

Signal conventions: D=Demand(command), Z=Position(feedback), R=Running(status)
Large D vs Z gap = actuator tracking fault or CV_injection attack.

Attack types: SP_manipulation, CV_injection, PV_spoofing,
SP_manipulation_with_PV_spoofing, CV_injection_with_PV_spoofing,
compound_attack, indirect_effect.

Response format (always follow this exactly):
1. SUMMARY (2 sentences — what is happening and severity)
2. EVIDENCE (each sensor, its role, what its value indicates)
3. PROBABLE ATTACK SCENARIO (HAI attack type and reasoning)
4. IMMEDIATE ACTIONS (numbered, operator-actionable, exact tag names)
5. SENSORS TO VERIFY ON OWS (exact tag list)
6. RISK LEVEL: LOW / MEDIUM / HIGH / CRITICAL"""


def llm_explain(ctx: dict) -> str:
    sensors_str = "\n".join(
        f"  • {s['name']:25s} | role: {s['role']:30s} | recon_err: {s['error']:.6f}"
        for s in ctx["top_sensors"]
    )
    shap_str = "\n".join(
        f"  • {s['feature']:30s}  SHAP = {s['value']:+.5f}"
        for s in ctx.get("shap", [])
    ) or "  (SHAP not available for this window)"

    prompt = f"""ANOMALY DETECTED
================
Loop:          {ctx['top_loop']} — {ctx['loop_description']}
Attack type:   {ctx['attack_type']}
Score:         {ctx['anomaly_score']:.4f}  (severity: {ctx['severity']})

Top anomalous sensors (by reconstruction error):
{sensors_str}

SHAP feature contributions (what drove the IsolationForest decision):
{shap_str}

Provide operator advisory guidance following the required format."""

    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=800,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
        )
        return res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/health")
def health():
    return {
        "status":           "ok",
        "n_features":       N_FEATURES,
        "seq_len":          SEQ_LEN,
        "anomalies_ready":  len(score_records),
        "models":           ["LSTM-AE", "Transformer-AE", "IsolationForest"],
        "llm":              "llama-3.1-8b-instant (Groq)",
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a HAI sensor CSV.
    Runs all 3 models, computes SHAP, stores anomaly records.
    Returns count of anomalies found.
    """
    global score_records

    # ── 1. Read CSV
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    print(f"[upload] CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[upload] Columns sample: {df.columns[:8].tolist()}")

    # ── 2. Validate minimum required columns
    required_cols = ["P1_FCV01D", "P1_LIT01", "P1_PIT01"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required sensor columns: {missing_cols}. "
                   f"Make sure you are uploading a raw HAI dataset CSV."
        )

    # ── 3. Preprocess → (N, N_FEATURES)
    try:
        df_clean = preprocess(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    arr    = df_clean.values.astype(np.float32)   # (N, N_FEATURES)
    n_rows = len(arr)

    # ── 4. Create sliding windows → (N-SEQ_LEN, SEQ_LEN, N_FEATURES)
    windows  = create_windows(arr)                # validates shape internally
    n_wins   = len(windows)
    print(f"[upload] Windows created: {n_wins} windows of shape ({SEQ_LEN}, {N_FEATURES})")

    # ── 5. Score with all 3 models
    with torch.no_grad():
        # LSTM: (n_wins, N_FEATURES) → mean over features → (n_wins,)
        lstm_err  = lstm_model.reconstruction_error(windows).cpu().numpy().mean(axis=1)

        # Transformer: same
        trans_err = transformer_model.reconstruction_error(windows).cpu().numpy().mean(axis=1)

    # IsolationForest scores full array then aligns to window outputs
    iso_raw = -iso_forest.score_samples(arr)           # (n_rows,)
    iso_err = iso_raw[SEQ_LEN: SEQ_LEN + n_wins]      # (n_wins,)

    # ── 6. Clip all to same length (safety — should already match)
    min_len   = min(len(lstm_err), len(trans_err), len(iso_err))
    lstm_err  = lstm_err[:min_len]
    trans_err = trans_err[:min_len]
    iso_err   = iso_err[:min_len]

    # ── 7. Weighted ensemble score
    ensemble  = (
        0.5 * normalize_01(lstm_err)  +
        0.3 * normalize_01(trans_err) +
        0.2 * normalize_01(iso_err)
    )

    threshold = float(np.percentile(ensemble, 95))
    flags     = ensemble > threshold

    anomaly_indices = np.where(flags)[0]
    print(f"[upload] Ensemble scored. Threshold={threshold:.4f}, "
          f"Anomalies={len(anomaly_indices)}/{min_len}")

    # ── 8. SHAP — computed on the IsoForest input rows aligned to windows
    # Row i of windows corresponds to arr[i + SEQ_LEN] as the representative row
    shap_input = arr[SEQ_LEN: SEQ_LEN + min_len]      # (min_len, N_FEATURES)
    try:
        shap_values = explainer.shap_values(shap_input)   # (min_len, N_FEATURES)
        print(f"[upload] SHAP computed: {shap_values.shape}")
    except Exception as e:
        print(f"[upload] SHAP warning — falling back to zeros: {e}")
        shap_values = np.zeros((min_len, N_FEATURES), dtype=np.float32)

    # ── 9. Build score records for anomalous windows
    score_records = []

    for i in anomaly_indices:
        i = int(i)

        # Per-feature reconstruction error from LSTM for this window
        with torch.no_grad():
            win_t    = windows[i].unsqueeze(0)              # (1, SEQ_LEN, N_FEATURES)
            recon, _ = lstm_model(win_t)
            recon_np = recon.squeeze(0).cpu().numpy()       # (SEQ_LEN, N_FEATURES)

        win_np      = windows[i].cpu().numpy()              # (SEQ_LEN, N_FEATURES)
        feature_err = np.mean((win_np - recon_np) ** 2, axis=0)  # (N_FEATURES,)

        # SHAP top features for this window
        shap_row     = shap_values[i] if i < len(shap_values) else np.zeros(N_FEATURES)
        top_shap_idx = np.argsort(np.abs(shap_row))[::-1][:6]
        shap_top     = [
            {
                "feature":   FEATURE_COLS[j],
                "value":     round(float(shap_row[j]), 5),
                "direction": "toward anomaly" if shap_row[j] > 0 else "toward normal",
            }
            for j in top_shap_idx
            if j < N_FEATURES
        ]

        ctx = build_context(i, float(ensemble[i]), feature_err, shap_top)
        score_records.append(ctx)

    print(f"[upload] Done. {len(score_records)} anomaly records stored.")

    return {
        "status":          "ok",
        "rows_uploaded":   df.shape[0],
        "rows_processed":  n_rows,
        "total_windows":   int(min_len),
        "threshold":       round(threshold, 4),
        "anomalies_found": len(score_records),
    }


@app.get("/anomalies")
def get_anomalies(severity: str = None, loop: str = None, limit: int = 100):
    """
    Return all anomaly records.
    Optional filters: ?severity=HIGH  ?loop=P1-PC
    """
    records = score_records
    if severity:
        records = [r for r in records if r["severity"] == severity.upper()]
    if loop:
        records = [r for r in records if r["top_loop"] == loop.upper()]
    records = sorted(records, key=lambda x: -x["anomaly_score"])
    return {"count": len(records), "events": records[:limit]}


@app.get("/anomaly/{window_idx}")
def get_anomaly(window_idx: int):
    """Get full detail for one anomaly window by its index."""
    match = next((r for r in score_records if r["window_idx"] == window_idx), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"window_idx={window_idx} not found. "
                   f"Available: {[r['window_idx'] for r in score_records[:5]]}"
        )
    return match


@app.post("/explain")
def explain(req: ExplainRequest):
    """Call Groq LLM to generate operator advisory for one anomaly window."""
    ctx = next((r for r in score_records if r["window_idx"] == req.window_idx), None)
    if not ctx:
        raise HTTPException(
            status_code=404,
            detail=f"window_idx={req.window_idx} not found. Run /upload first."
        )
    advisory = llm_explain(ctx)
    return {
        "window_idx":  req.window_idx,
        "severity":    ctx["severity"],
        "top_loop":    ctx["top_loop"],
        "attack_type": ctx["attack_type"],
        "score":       ctx["anomaly_score"],
        "advisory":    advisory,
    }


@app.get("/stats")
def stats():
    """Summary statistics across all detected anomalies."""
    if not score_records:
        return {"message": "No anomalies yet — POST a CSV to /upload first"}

    sev_counts    = {}
    loop_counts   = {}
    attack_counts = {}

    for r in score_records:
        sev_counts[r["severity"]]       = sev_counts.get(r["severity"], 0) + 1
        loop_counts[r["top_loop"]]      = loop_counts.get(r["top_loop"], 0) + 1
        attack_counts[r["attack_type"]] = attack_counts.get(r["attack_type"], 0) + 1

    scores = [r["anomaly_score"] for r in score_records]

    return {
        "total_anomalies": len(score_records),
        "by_severity":     sev_counts,
        "by_loop":         dict(sorted(loop_counts.items(),   key=lambda x: -x[1])),
        "by_attack_type":  dict(sorted(attack_counts.items(), key=lambda x: -x[1])),
        "score_stats": {
            "min":  round(min(scores), 4),
            "max":  round(max(scores), 4),
            "mean": round(float(np.mean(scores)), 4),
        },
    }