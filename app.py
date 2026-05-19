from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DLNLP Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PHOBERT_DIR = MODEL_DIR / "phobert_base"

LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_VI = {0: "Bình thường", 1: "Công kích", 2: "Thù ghét"}
LABEL_COLOR = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}
LABEL_BG = {0: "#dcfce7", 1: "#ffedd5", 2: "#fee2e2"}
LABEL_TEXT = {0: "#14532d", 1: "#7c2d12", 2: "#7f1d1d"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLES = {
    "Câu tích cực": "Bài viết này rất hữu ích, cảm ơn bạn đã chia sẻ.",
    "Câu công kích nhẹ": "Đọc mà chán thật, nói năng kiểu này thì ai chịu nổi.",
    "Câu thù ghét": "Mấy đứa đó đúng là phải biến khỏi đây ngay.",
    "Câu trung tính": "Hôm nay trời khá mát, mình vừa đi học về.",
}


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.emb_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        mask = (input_ids != 0).unsqueeze(-1)
        output_masked = output.masked_fill(~mask, -1e9)
        max_pool = output_masked.max(dim=1).values

        sum_pool = (output * mask).sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1)
        mean_pool = sum_pool / valid_len

        feat = torch.cat([max_pool, mean_pool], dim=1)
        logits = self.fc(self.dropout(feat))
        return logits


class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.texts = list(texts)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, length = encode_text(self.texts[idx], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }


@st.cache_resource(show_spinner=False)
def load_svm_artifacts() -> Tuple[Any, Any]:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "svm_model.joblib")
    return vectorizer, model


@st.cache_resource(show_spinner=False)
def load_bilstm_artifacts() -> Optional[Dict[str, Any]]:
    vocab_path = MODEL_DIR / "bilstm_vocab.json"
    model_path = MODEL_DIR / "bilstm_best.pt"
    metrics_path = RESULT_DIR / "bilstm_metrics.csv"

    if not vocab_path.exists() or not model_path.exists() or not metrics_path.exists():
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        return None

    row = metrics_df.iloc[0]
    max_len = int(row.get("max_len", 64))
    embed_dim = int(row.get("embed_dim", 300))
    hidden_dim = int(row.get("hidden_dim", 128))
    num_layers = int(row.get("num_layers", 2))

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        num_layers=num_layers,
        dropout=0.4,
    ).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "max_len": max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


@st.cache_resource(show_spinner=False)
def find_phobert_model_dir() -> Optional[Path]:
    if not PHOBERT_DIR.exists():
        return None

    candidates = []
    for pattern in ("**/config.json", "**/pytorch_model.bin", "**/model.safetensors"):
        for file_path in PHOBERT_DIR.glob(pattern):
            candidates.append(file_path.parent)

    if not candidates:
        return None

    unique_dirs = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(candidate)

    return max(unique_dirs, key=lambda item: item.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def load_phobert_artifacts() -> Optional[Dict[str, Any]]:
    model_dir = find_phobert_model_dir()
    if model_dir is None:
        return None

    try:
        from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "model_dir": model_dir}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def segment_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return ViTokenizer.tokenize(text)


@st.cache_data(show_spinner=False)
def tokenize_text(text: str):
    return str(text).split()


@st.cache_data(show_spinner=False)
def encode_text(text: str, vocab: Dict[str, int], max_len: int):
    tokens = tokenize_text(text)[:max_len]
    ids = [vocab.get(token, 1) for token in tokens]

    if len(ids) == 0:
        ids = [1]

    length = len(ids)
    if length < max_len:
        ids += [0] * (max_len - length)

    return ids, length


@torch.no_grad()
def predict_bilstm(text: str, artifacts: Dict[str, Any]):
    model = artifacts["model"]
    vocab = artifacts["vocab"]
    max_len = artifacts["max_len"]

    seg_text = segment_text(clean_text(text))
    input_ids, length = encode_text(seg_text, vocab, max_len)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=DEVICE),
        "length": torch.tensor([length], dtype=torch.long, device=DEVICE),
    }
    logits = model(batch["input_ids"], batch["length"])
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs


@torch.no_grad()
def predict_phobert(text: str, artifacts: Dict[str, Any]):
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    seg_text = segment_text(clean_text(text))
    encoding = tokenizer(
        [seg_text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encoding = {key: value.to(DEVICE) for key, value in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs


@st.cache_resource(show_spinner=False)
def load_models():
    svm_vectorizer, svm_model = load_svm_artifacts()
    bilstm_artifacts = load_bilstm_artifacts()
    phobert_artifacts = load_phobert_artifacts()
    return {
        "svm": (svm_vectorizer, svm_model),
        "bilstm": bilstm_artifacts,
        "phobert": phobert_artifacts,
    }


def predict_svm(text: str, artifacts: Tuple[Any, Any]):
    vectorizer, model = artifacts
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred_id = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if np.ndim(scores) == 1:
            scores = np.column_stack([-scores, scores])
        probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=1).squeeze(0).detach().cpu().numpy()
    else:
        probs = None

    return pred_id, probs


def top2_labels(probs: Optional[np.ndarray]) -> str:
    if probs is None:
        return "Top 2: N/A"

    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return "Top 2: N/A"

    order = probs.argsort()[::-1][:2]
    parts = [f"{LABEL_MAP.get(int(index), 'Unknown')} {probs[int(index)]:.1%}" for index in order]
    return f"Top 2: {' | '.join(parts)}"


def _safe_label_id(label_id: Optional[int]) -> Optional[int]:
    if label_id is None:
        return None
    try:
        label_id = int(label_id)
    except Exception:
        return None
    return label_id if label_id in LABEL_MAP else None


def _format_confidence(conf: Optional[float]) -> str:
    if conf is None:
        return "N/A"
    try:
        return f"{float(conf) * 100:.1f}%"
    except Exception:
        return "N/A"


def confidence_from_probs(pred_id: Optional[int], probs: Optional[np.ndarray]) -> Optional[float]:
    safe_id = _safe_label_id(pred_id)
    if safe_id is None or probs is None:
        return None
    try:
        probs = np.asarray(probs, dtype=float)
        if probs.size == 0:
            return None
        return float(probs[safe_id])
    except Exception:
        return None


def majority_vote(pred_ids: list[int]) -> int:
    valid = [pred for pred in pred_ids if pred in LABEL_MAP]
    if not valid:
        return 0
    return Counter(valid).most_common(1)[0][0]


def consensus_status(pred_ids: list[int]) -> bool:
    valid = [pred for pred in pred_ids if pred in LABEL_MAP]
    if len(valid) <= 1:
        return True
    return len(set(valid)) == 1


def render_model_card(model_name: str, label_id: Optional[int], confidence: Optional[float], *, available: bool):
    safe_id = _safe_label_id(label_id)
    if not available:
        label_code = "-"
        label_primary = "KHÔNG KHẢ DỤNG"
    elif safe_id is None:
        label_code = "-"
        label_primary = "CHƯA DỰ ĐOÁN"
    else:
        label_code = LABEL_MAP[safe_id]
        label_primary = label_code

    bg = "#f1f5f9" if safe_id is None else LABEL_BG[safe_id]
    border = "#94a3b8" if safe_id is None else LABEL_COLOR[safe_id]
    text = "#0f172a" if safe_id is None else LABEL_TEXT[safe_id]

    conf_text = _format_confidence(confidence) if available and safe_id is not None else "N/A"

    st.markdown(
        f"""
        <div class="model-card" style="background: {bg}; border: 1px solid rgba(148,163,184,0.35);">
            <div class="model-card__head">
                <div class="model-card__name">{model_name}</div>
                <div class="model-card__status">{'Sẵn sàng' if available else 'Thiếu artifact'}</div>
            </div>
            <div class="model-card__label" style="color: {text};">{label_primary}</div>
            <div class="model-card__meta">
                <span class="pill" style="border-color:{border}; color:{border};">{label_code}</span>
                <span class="confidence">Độ tin cậy: <b>{conf_text}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_final_conclusion(label_id: int, confidence: Optional[float] = None, note: Optional[str] = None):
    safe_id = _safe_label_id(label_id) or 0
    bg = LABEL_BG[safe_id]
    border = LABEL_COLOR[safe_id]
    text = LABEL_TEXT[safe_id]
    conf_text = _format_confidence(confidence)
    note_html = "" if not note else f"<div class='final-note'>{note}</div>"

    st.markdown(
        f"""
        <div class="final-card" style="background:{bg}; border: 1px solid rgba(148,163,184,0.35);">
            <div class="final-card__head">
                <div class="final-card__title">Kết luận cuối</div>
                <span class="pill" style="background:{border}; color:white; border-color:{border};">{LABEL_MAP[safe_id]}</span>
            </div>
            <div class="final-card__label" style="color:{text};">{LABEL_MAP[safe_id]}</div>
            <div class="final-card__meta">Độ tin cậy (tham khảo): <b>{conf_text}</b></div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_ui_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        .sidebar-card {
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            margin: 0.3rem 0;
            font-size: 0.92rem;
            color: #0f172a;
        }
        .status-row span {
            color: #475569;
        }

        .hero {
            padding: 1.15rem 1.25rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 1.85rem;
            color: #0f172a;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            color: #475569;
            font-size: 1rem;
            line-height: 1.55;
        }

        .model-card {
            padding: 1.0rem 1.05rem;
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            min-height: 140px;
        }
        .model-card__head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.8rem;
            margin-bottom: 0.55rem;
        }
        .model-card__name {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.02em;
        }
        .model-card__status {
            font-size: 0.82rem;
            color: #64748b;
        }
        .model-card__label {
            font-size: 1.25rem;
            font-weight: 900;
            margin-bottom: 0.55rem;
        }
        .model-card__meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            flex-wrap: wrap;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-weight: 800;
            font-size: 0.82rem;
            background: rgba(255, 255, 255, 0.5);
        }
        .confidence {
            color: #0f172a;
            font-size: 0.9rem;
        }

        .final-card {
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.09);
        }
        .final-card__head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.45rem;
        }
        .final-card__title {
            font-size: 1.0rem;
            font-weight: 900;
            color: #0f172a;
        }
        .final-card__label {
            font-size: 1.55rem;
            font-weight: 950;
            margin-bottom: 0.25rem;
        }
        .final-card__meta {
            color: #334155;
            font-size: 0.95rem;
        }
        .final-note {
            margin-top: 0.6rem;
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 16px;
        }

        button[data-testid="baseButton-primary"] {
            border-radius: 14px !important;
            padding: 0.55rem 1rem !important;
            border: 1px solid rgba(59, 130, 246, 0.35) !important;
            background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%) !important;
            color: white !important;
            font-weight: 800 !important;
        }
        button[data-testid="baseButton-secondary"] {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()

st.markdown(
    """
    <div class="hero">
        <h1>Phân loại bình luận độc hại tiếng Việt</h1>
        <p>Nhập một bình luận bất kỳ và so sánh dự đoán từ 3 mô hình: TF‑IDF + SVM, BiLSTM và PhoBERT.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models = None
with st.spinner("Đang tải mô hình đã lưu..."):
    try:
        models = load_models()
    except Exception as exc:
        st.error(f"Không thể tải model: {exc}")
        st.stop()

svm_available = (MODEL_DIR / "tfidf_vectorizer.joblib").exists() and (MODEL_DIR / "svm_model.joblib").exists()
bilstm_available = models["bilstm"] is not None
phobert_available = models["phobert"] is not None

with st.sidebar:
    st.markdown("### Demo")
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="status-row"><b>Thiết bị</b><span>{DEVICE}</span></div>
            <div class="status-row"><b>SVM</b><span>{'Sẵn sàng' if svm_available else 'Thiếu artifact'}</span></div>
            <div class="status-row"><b>BiLSTM</b><span>{'Sẵn sàng' if bilstm_available else 'Thiếu artifact'}</span></div>
            <div class="status-row"><b>PhoBERT</b><span>{'Sẵn sàng' if phobert_available else 'Thiếu artifact'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("\n")
    with st.expander("Văn bản mẫu", expanded=True):
        if "input_text" not in st.session_state:
            st.session_state["input_text"] = ""

        for label, sample_text in SAMPLES.items():
            if st.button(label, use_container_width=True):
                st.session_state["input_text"] = sample_text

left_col, right_col = st.columns([1.4, 1.0], gap="large")

with left_col:
    st.markdown("### Nhập nội dung")
    input_text = st.text_area(
        "Nội dung bình luận",
        key="input_text",
        height=220,
        placeholder="Nhập một câu bình luận tiếng Việt ở đây...",
        label_visibility="collapsed",
    )

    action_col, helper_col = st.columns([0.33, 0.67], gap="medium")
    with action_col:
        predict_clicked = st.button("Dự đoán", use_container_width=True, type="primary")
    with helper_col:
        if st.button("Xóa nội dung", use_container_width=True):
            st.session_state["input_text"] = ""
            st.session_state.pop("last_results", None)
            st.session_state.pop("last_vote", None)
            st.rerun()

    cleaned_preview = clean_text(input_text)
    segmented_preview = segment_text(cleaned_preview)

    with st.expander("Tiền xử lý đầu vào", expanded=False):
        st.markdown("**Văn bản sau chuẩn hóa**")
        st.code(cleaned_preview or "(trống)")
        st.markdown("**Văn bản sau tách từ**")
        st.code(segmented_preview or "(trống)")

with right_col:
    st.markdown("### Kết quả dự đoán")

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "last_vote" not in st.session_state:
        st.session_state["last_vote"] = None

    if predict_clicked and input_text.strip():
        results: Dict[str, Dict[str, Any]] = {
            "svm": {"name": "TF‑IDF + SVM", "available": svm_available, "pred": None, "probs": None},
            "bilstm": {"name": "BiLSTM", "available": bilstm_available, "pred": None, "probs": None},
            "phobert": {"name": "PhoBERT", "available": phobert_available, "pred": None, "probs": None},
        }

        if svm_available:
            pred, probs = predict_svm(input_text, models["svm"])
            results["svm"].update({"pred": pred, "probs": probs})

        if bilstm_available:
            pred, probs = predict_bilstm(input_text, models["bilstm"])
            results["bilstm"].update({"pred": pred, "probs": probs})

        if phobert_available:
            pred, probs = predict_phobert(input_text, models["phobert"])
            results["phobert"].update({"pred": pred, "probs": probs})

        preds_for_vote = [
            results["svm"]["pred"],
            results["bilstm"]["pred"],
            results["phobert"]["pred"],
        ]
        preds_for_vote = [p for p in preds_for_vote if p is not None]
        vote = majority_vote([int(p) for p in preds_for_vote]) if preds_for_vote else 0

        st.session_state["last_results"] = results
        st.session_state["last_vote"] = vote

    results = st.session_state.get("last_results")
    vote = st.session_state.get("last_vote")

    cards = st.columns(3, gap="large")
    if results is None:
        with cards[0]:
            render_model_card("TF‑IDF + SVM", None, None, available=svm_available)
        with cards[1]:
            render_model_card("BiLSTM", None, None, available=bilstm_available)
        with cards[2]:
            render_model_card("PhoBERT", None, None, available=phobert_available)
        st.info("Nhập một bình luận và bấm **Dự đoán** để xem kết quả.")
    else:
        svm_pred = results["svm"]["pred"]
        bilstm_pred = results["bilstm"]["pred"]
        phobert_pred = results["phobert"]["pred"]

        svm_conf = confidence_from_probs(svm_pred, results["svm"]["probs"])
        bilstm_conf = confidence_from_probs(bilstm_pred, results["bilstm"]["probs"])
        phobert_conf = confidence_from_probs(phobert_pred, results["phobert"]["probs"])

        with cards[0]:
            render_model_card("TF‑IDF + SVM", svm_pred, svm_conf, available=bool(results["svm"]["available"]))
        with cards[1]:
            render_model_card("BiLSTM", bilstm_pred, bilstm_conf, available=bool(results["bilstm"]["available"]))
        with cards[2]:
            render_model_card("PhoBERT", phobert_pred, phobert_conf, available=bool(results["phobert"]["available"]))

        st.caption("Diễn giải nhãn: CLEAN = Bình thường · OFFENSIVE = Công kích · HATE = Thù ghét")

        pred_ids = [
            int(svm_pred) if svm_pred is not None else -1,
            int(bilstm_pred) if bilstm_pred is not None else -1,
            int(phobert_pred) if phobert_pred is not None else -1,
        ]
        is_consensus = consensus_status(pred_ids)

        st.markdown("\n")
        if not is_consensus:
            st.warning("Đây là trường hợp khó, các mô hình đưa ra dự đoán khác nhau.")

        st.markdown("\n")
        note = "Theo nguyên tắc đa số phiếu từ các mô hình khả dụng." if vote is not None else None
        final_label = int(vote or 0)
        final_conf_candidates: list[float] = []
        for pred, conf in (
            (svm_pred, svm_conf),
            (bilstm_pred, bilstm_conf),
            (phobert_pred, phobert_conf),
        ):
            if pred is None or conf is None:
                continue
            try:
                if int(pred) == final_label:
                    final_conf_candidates.append(float(conf))
            except Exception:
                continue

        final_conf = float(np.mean(final_conf_candidates)) if final_conf_candidates else None
        render_final_conclusion(final_label, final_conf, note)

        st.caption("Độ tin cậy là xác suất lớp dự đoán (nếu mô hình có cung cấp).")
