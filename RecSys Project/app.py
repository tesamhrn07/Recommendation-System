from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ============================================================
# 0. KONFIGURASI HALAMAN DAN KONSTANTA
# ============================================================

st.set_page_config(
    page_title="Hotel Recommendation System",
    page_icon="🏨",
    layout="wide",
)

DATA_PATHS = [
    Path("data/master data hotel_200.xlsx"),
    Path("master data hotel_200.xlsx"),
]
TRAVELOKA_SHEET = "Traveloka Data"
GMAPS_SHEET = "Gmaps Review"
REQUIRED_HOTEL_COLS = {"ID Hotel", "Nama Hotel", "Harga", "Rating Traveloka", "Predikat", "Wilayah"}
REQUIRED_REVIEW_COLS = {"ID Hotel", "title", "stars", "name"}


# ============================================================
# 1. UTILITAS UMUM
# ============================================================

def make_ohe() -> OneHotEncoder:
    """Kompatibel untuk scikit-learn lama dan baru."""
    try:
        return OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=True, handle_unknown="ignore")


def rupiah(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"Rp {float(value):,.0f}".replace(",", ".")


def normalize_predikat(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    text = str(x).strip()
    return text if text else "Unknown"


def normalize_user(x: object) -> str:
    if pd.isna(x):
        return "Unknown Reviewer"
    text = str(x).strip()
    return text if text else "Unknown Reviewer"


def first_existing_data_path() -> Optional[Path]:
    for path in DATA_PATHS:
        if path.exists():
            return path
    return None


def validate_columns(df: pd.DataFrame, required: set, label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom {label} tidak lengkap. Kolom hilang: {sorted(missing)}")


def predikat_weight(x: str) -> float:
    val = str(x).strip().lower()
    if val == "istimewa":
        return 1.00
    if val == "mengesankan":
        return 0.85
    if val in {"sangat bagus", "memuaskan"}:
        return 0.70
    if val == "mengecewakan":
        return 0.30
    return 0.50


def minmax_series(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    base_cols = ["ID Hotel", "Nama Hotel", "Harga", "Rating Traveloka", "Predikat", "Wilayah"]
    score_cols = [
        c for c in [
            "Content_Score",
            "Preference_Score",
            "CF_Score",
            "Hybrid_Score",
            "RW_Score",
            "GNN_Score",
        ] if c in df.columns
    ]
    cols = [c for c in base_cols if c in df.columns] + score_cols
    out = df[cols].copy()
    if "Harga" in out.columns:
        out["Harga"] = out["Harga"].apply(rupiah)
    for col in score_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(5)
    return out


def show_selected_hotel(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ID Hotel", int(row["ID Hotel"]))
    c2.metric("Harga", rupiah(row["Harga"]))
    c3.metric("Rating Traveloka", f"{float(row['Rating Traveloka']):.1f}")
    c4.metric("Predikat", str(row["Predikat"]))
    st.caption(f"Nama hotel: **{row['Nama Hotel']}** | Wilayah: {row['Wilayah']}")


def display_metrics(metrics: dict, k: int) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Precision@{k}", f"{metrics.get('Precision@K', 0.0):.4f}")
    m2.metric(f"Recall@{k}", f"{metrics.get('Recall@K', 0.0):.4f}")
    m3.metric(f"F1@{k}", f"{metrics.get('F1@K', 0.0):.4f}")
    m4.metric("Users Evaluated", int(metrics.get("Users Evaluated", 0)))


# ============================================================
# 2. LOAD DAN CLEANING DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel_data(file_bytes: Optional[bytes], fallback_path: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_bytes is not None:
        source = file_bytes
    elif fallback_path:
        source = fallback_path
    else:
        raise FileNotFoundError("File Excel belum tersedia.")

    df_hotel = pd.read_excel(source, sheet_name=TRAVELOKA_SHEET)
    df_review = pd.read_excel(source, sheet_name=GMAPS_SHEET)

    df_hotel.columns = df_hotel.columns.astype(str).str.strip()
    df_review.columns = df_review.columns.astype(str).str.strip()

    validate_columns(df_hotel, REQUIRED_HOTEL_COLS, "hotel")
    validate_columns(df_review, REQUIRED_REVIEW_COLS, "review")

    # Cleaning master hotel
    df_hotel = df_hotel.copy()
    df_hotel["ID Hotel"] = pd.to_numeric(df_hotel["ID Hotel"], errors="coerce")
    df_hotel = df_hotel.dropna(subset=["ID Hotel"]).copy()
    df_hotel["ID Hotel"] = df_hotel["ID Hotel"].astype(int)
    df_hotel = df_hotel.drop_duplicates("ID Hotel", keep="first")

    df_hotel["Nama Hotel"] = df_hotel["Nama Hotel"].fillna("Unknown").astype(str).str.strip()
    df_hotel["Harga"] = pd.to_numeric(df_hotel["Harga"], errors="coerce")
    df_hotel["Rating Traveloka"] = pd.to_numeric(df_hotel["Rating Traveloka"], errors="coerce")
    df_hotel["Harga"] = df_hotel["Harga"].fillna(df_hotel["Harga"].median())
    df_hotel["Rating Traveloka"] = df_hotel["Rating Traveloka"].fillna(df_hotel["Rating Traveloka"].median())
    df_hotel["Predikat"] = df_hotel["Predikat"].apply(normalize_predikat)
    df_hotel["Wilayah"] = df_hotel["Wilayah"].fillna("Unknown").astype(str).str.strip()

    # Cleaning review
    df_review = df_review.copy()
    df_review["ID Hotel"] = pd.to_numeric(df_review["ID Hotel"], errors="coerce")
    df_review = df_review.dropna(subset=["ID Hotel"]).copy()
    df_review["ID Hotel"] = df_review["ID Hotel"].astype(int)
    df_review["stars"] = pd.to_numeric(df_review["stars"], errors="coerce")
    df_review = df_review.dropna(subset=["stars"]).copy()
    df_review["stars"] = df_review["stars"].astype(float)
    df_review["title"] = df_review["title"].fillna("").astype(str)
    df_review["name"] = df_review["name"].apply(normalize_user)

    valid_ids = set(df_hotel["ID Hotel"])
    df_review = df_review[df_review["ID Hotel"].isin(valid_ids)].copy()

    return df_hotel.reset_index(drop=True), df_review.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def prepare_hotel_with_reviews(df_hotel: pd.DataFrame, df_review: pd.DataFrame) -> pd.DataFrame:
    review_per_hotel = (
        df_review.groupby("ID Hotel")["title"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index(name="Combined_Reviews")
    )
    df = df_hotel.merge(review_per_hotel, on="ID Hotel", how="left")
    df["Combined_Reviews"] = df["Combined_Reviews"].fillna("")
    return df.reset_index(drop=True)


# ============================================================
# 3. SPLIT TRAIN/TEST UNTUK INTERAKSI USER-HOTEL
# ============================================================

@st.cache_data(show_spinner=False)
def create_train_test_split(
    df_review: pd.DataFrame,
    rating_threshold: float = 4.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split berbasis interaksi positif per user.
    Test tidak digunakan untuk membangun CF, graph Random Walk, atau LightGCN.
    """
    df = df_review.copy()
    df["user_key"] = df["name"].apply(normalize_user)
    df = df[df["stars"] >= rating_threshold].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = (
        df.sort_values(["user_key", "ID Hotel", "stars"], ascending=[True, True, False])
        .drop_duplicates(["user_key", "ID Hotel"], keep="first")
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(random_state)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in df.groupby("user_key", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1)
        perm = rng.permutation(idx)
        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())

    train = df.loc[sorted(train_idx)].reset_index(drop=True)
    test = df.loc[sorted(test_idx)].reset_index(drop=True)
    return train, test


@st.cache_data(show_spinner=False)
def encode_interactions(
    df_hotel: pd.DataFrame,
    train_pos: pd.DataFrame,
    test_pos: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    users = sorted(set(train_pos.get("user_key", pd.Series(dtype=str))) | set(test_pos.get("user_key", pd.Series(dtype=str))))
    hotel_ids = df_hotel["ID Hotel"].astype(int).tolist()

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {hid: i for i, hid in enumerate(hotel_ids)}
    idx_to_item = {i: hid for hid, i in item_to_idx.items()}

    def enc(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=list(df.columns) + ["user_idx", "item_idx"])
        out = df.copy()
        out["user_idx"] = out["user_key"].map(user_to_idx).astype(int)
        out["item_idx"] = out["ID Hotel"].map(item_to_idx).astype(int)
        return out.dropna(subset=["user_idx", "item_idx"]).reset_index(drop=True)

    return enc(train_pos), enc(test_pos), user_to_idx, idx_to_user, item_to_idx, idx_to_item


# ============================================================
# 4. CONTENT-BASED FEATURE ENGINEERING
# ============================================================

@st.cache_resource(show_spinner=False)
def build_content_similarity(df_hotel: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df_hotel[["Harga", "Rating Traveloka"]])
    num_features = csr_matrix(num_features)

    ohe = make_ohe()
    cat_features = ohe.fit_transform(df_hotel[["Predikat"]])

    tfidf_wilayah = TfidfVectorizer(max_features=80)
    wilayah_features = tfidf_wilayah.fit_transform(df_hotel["Wilayah"].astype(str))

    parts = [num_features, cat_features, wilayah_features]

    if "Combined_Reviews" in df_hotel.columns:
        tfidf_review = TfidfVectorizer(max_features=400, ngram_range=(1, 2), min_df=1)
        review_features = tfidf_review.fit_transform(df_hotel["Combined_Reviews"].astype(str))
        parts.append(review_features)

    feature_matrix = hstack(parts)
    return cosine_similarity(feature_matrix)


# ============================================================
# 5. CONTENT-BASED ITEM-TO-ITEM
# ============================================================

def recommend_content_item_to_item(
    df_hotel: pd.DataFrame,
    similarity_matrix: np.ndarray,
    hotel_id: int,
    top_n: int = 5,
) -> Tuple[pd.Series, pd.DataFrame]:
    idx_list = df_hotel.index[df_hotel["ID Hotel"] == hotel_id].tolist()
    if not idx_list:
        raise ValueError(f"Hotel ID {hotel_id} tidak ditemukan.")

    idx = idx_list[0]
    selected = df_hotel.loc[idx]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

    result = df_hotel.iloc[[i for i, _ in sim_scores]].copy()
    result["Content_Score"] = [float(s) for _, s in sim_scores]
    return selected, result


# ============================================================
# 6. CONTENT-BASED PREFERENCE-BASED
# ============================================================

def recommend_content_preference_based(
    df_hotel: pd.DataFrame,
    wilayah: Optional[str],
    harga_min: Optional[float],
    harga_max: Optional[float],
    rating_min: Optional[float],
    predikat: Optional[str],
    top_n: int,
    w_rating: float,
    w_harga: float,
    w_wilayah: float,
    w_predikat: float,
) -> pd.DataFrame:
    df = df_hotel.copy()

    if wilayah:
        df = df[df["Wilayah"].str.contains(wilayah, case=False, na=False)]
    if harga_min is not None:
        df = df[df["Harga"] >= harga_min]
    if harga_max is not None:
        df = df[df["Harga"] <= harga_max]
    if rating_min is not None:
        df = df[df["Rating Traveloka"] >= rating_min]
    if predikat and predikat != "Semua":
        df = df[df["Predikat"].str.lower() == predikat.lower()]

    if df.empty:
        return df

    df = df.copy()
    df["norm_rating"] = minmax_series(df["Rating Traveloka"])
    df["norm_harga_inv"] = 1.0 - minmax_series(df["Harga"])
    df["score_wilayah"] = 1.0 if wilayah else 0.5
    df["score_predikat"] = df["Predikat"].apply(predikat_weight)

    total_w = max(w_rating + w_harga + w_wilayah + w_predikat, 1e-9)
    df["Preference_Score"] = (
        w_rating * df["norm_rating"]
        + w_harga * df["norm_harga_inv"]
        + w_wilayah * df["score_wilayah"]
        + w_predikat * df["score_predikat"]
    ) / total_w

    return df.sort_values("Preference_Score", ascending=False).head(top_n)


# ============================================================
# 7. COLLABORATIVE FILTERING DAN HYBRID
# ============================================================

@st.cache_resource(show_spinner=False)
def build_item_cf_similarity(train_pos: pd.DataFrame) -> pd.DataFrame:
    if train_pos.empty:
        return pd.DataFrame()
    user_item = train_pos.pivot_table(
        index="user_key",
        columns="ID Hotel",
        values="stars",
        aggfunc="mean",
    ).fillna(0)
    if user_item.shape[0] < 1 or user_item.shape[1] < 2:
        return pd.DataFrame()
    item_sim = cosine_similarity(user_item.T)
    return pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)


def recommend_collaborative(
    df_hotel: pd.DataFrame,
    train_pos: pd.DataFrame,
    item_sim_df: pd.DataFrame,
    user_key: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if item_sim_df.empty or train_pos.empty:
        return pd.DataFrame()

    user_hist = train_pos[train_pos["user_key"] == user_key]
    if user_hist.empty:
        return pd.DataFrame()

    seen_ids = user_hist["ID Hotel"].astype(int).tolist()
    seen_ratings = user_hist.set_index("ID Hotel")["stars"].astype(float)
    seen_available = [hid for hid in seen_ids if hid in item_sim_df.columns]
    if not seen_available:
        return pd.DataFrame()

    sim_part = item_sim_df[seen_available].copy()
    rating_weights = seen_ratings.reindex(seen_available).fillna(1.0)
    scores = sim_part.dot(rating_weights) / (rating_weights.abs().sum() + 1e-9)
    scores = scores.drop(index=seen_ids, errors="ignore")
    scores = scores.sort_values(ascending=False).head(top_n)

    result = df_hotel[df_hotel["ID Hotel"].isin(scores.index)].copy()
    result["CF_Score"] = result["ID Hotel"].map(scores)
    return result.sort_values("CF_Score", ascending=False)


def recommend_hybrid_user(
    df_hotel: pd.DataFrame,
    train_pos: pd.DataFrame,
    similarity_matrix: np.ndarray,
    item_sim_df: pd.DataFrame,
    user_key: str,
    top_n: int = 5,
    alpha_content: float = 0.5,
) -> pd.DataFrame:
    user_hist = train_pos[train_pos["user_key"] == user_key]
    if user_hist.empty:
        return pd.DataFrame()

    hotel_ids = df_hotel["ID Hotel"].astype(int).tolist()
    hotel_id_to_row = {hid: i for i, hid in enumerate(hotel_ids)}
    seen_ids = [int(x) for x in user_hist["ID Hotel"].tolist() if int(x) in hotel_id_to_row]
    if not seen_ids:
        return pd.DataFrame()

    seen_rows = [hotel_id_to_row[hid] for hid in seen_ids]
    content_scores_arr = similarity_matrix[:, seen_rows].mean(axis=1)
    content_scores = pd.Series(content_scores_arr, index=hotel_ids)
    content_scores = minmax_series(content_scores)

    if item_sim_df.empty:
        cf_scores = pd.Series(0.0, index=hotel_ids)
    else:
        seen_available = [hid for hid in seen_ids if hid in item_sim_df.columns]
        if seen_available:
            cf_raw = item_sim_df[seen_available].mean(axis=1).reindex(hotel_ids).fillna(0.0)
            cf_scores = minmax_series(cf_raw)
        else:
            cf_scores = pd.Series(0.0, index=hotel_ids)

    hybrid = alpha_content * content_scores + (1.0 - alpha_content) * cf_scores
    hybrid = hybrid.drop(index=seen_ids, errors="ignore").sort_values(ascending=False).head(top_n)

    result = df_hotel[df_hotel["ID Hotel"].isin(hybrid.index)].copy()
    result["Hybrid_Score"] = result["ID Hotel"].map(hybrid)
    result["Content_Score"] = result["ID Hotel"].map(content_scores)
    result["CF_Score"] = result["ID Hotel"].map(cf_scores)
    return result.sort_values("Hybrid_Score", ascending=False)


# ============================================================
# 8. RANDOM WALK / PERSONALIZED PAGERANK
# ============================================================

@st.cache_resource(show_spinner=False)
def build_random_walk_graph(train_enc: pd.DataFrame, num_users: int, num_items: int) -> nx.Graph:
    G = nx.Graph()
    for uid in range(num_users):
        G.add_node(f"U_{uid}", bipartite=0)
    for iid in range(num_items):
        G.add_node(f"H_{iid}", bipartite=1)

    if train_enc.empty:
        return G

    for _, row in train_enc.iterrows():
        weight = float(row.get("stars", 1.0))
        G.add_edge(f"U_{int(row['user_idx'])}", f"H_{int(row['item_idx'])}", weight=weight)
    return G


def random_walk_rank_item_indices(
    G: nx.Graph,
    user_idx: int,
    train_enc: pd.DataFrame,
    num_items: int,
    top_n: int,
    alpha: float = 0.85,
) -> List[Tuple[int, float]]:
    user_node = f"U_{user_idx}"
    if user_node not in G:
        return []

    personalization = {node: 0.0 for node in G.nodes()}
    personalization[user_node] = 1.0
    try:
        pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight")
    except Exception:
        return []

    seen = set(train_enc[train_enc["user_idx"] == user_idx]["item_idx"].astype(int))
    rows = []
    for iid in range(num_items):
        if iid in seen:
            continue
        score = float(pr.get(f"H_{iid}", 0.0))
        rows.append((iid, score))
    return sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]


def recommend_random_walk(
    df_hotel: pd.DataFrame,
    G: nx.Graph,
    train_enc: pd.DataFrame,
    idx_to_item: dict,
    user_idx: int,
    top_n: int,
    alpha: float,
) -> pd.DataFrame:
    ranked = random_walk_rank_item_indices(G, user_idx, train_enc, len(idx_to_item), top_n, alpha)
    if not ranked:
        return pd.DataFrame()
    scores = pd.DataFrame({"item_idx": [i for i, _ in ranked], "RW_Score": [s for _, s in ranked]})
    scores["ID Hotel"] = scores["item_idx"].map(idx_to_item)
    result = scores.merge(df_hotel, on="ID Hotel", how="left")
    return result.sort_values("RW_Score", ascending=False)


# ============================================================
# 9. EVALUASI RANKING UMUM
# ============================================================

def evaluate_ranking(
    train_enc: pd.DataFrame,
    test_enc: pd.DataFrame,
    idx_to_item: dict,
    k: int,
    ranker: Callable[[int, int], List[Tuple[int, float]]],
) -> dict:
    if train_enc.empty or test_enc.empty:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "F1@K": 0.0, "Users Evaluated": 0}

    precisions: List[float] = []
    recalls: List[float] = []
    eligible_users = sorted(set(test_enc["user_idx"].astype(int)) & set(train_enc["user_idx"].astype(int)))

    for uid in eligible_users:
        relevant = set(test_enc[test_enc["user_idx"] == uid]["item_idx"].astype(int))
        if not relevant:
            continue
        ranked = ranker(uid, k)
        recommended = [iid for iid, _ in ranked[:k]]
        if not recommended:
            continue
        hit = len(set(recommended) & relevant)
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "F1@K": f1,
        "Users Evaluated": len(precisions),
    }


# ============================================================
# 10. LIGHTGCN DARI SCRATCH DENGAN PYTORCH
# ============================================================

def torch_available() -> Tuple[bool, Optional[str]]:
    try:
        import torch  # noqa: F401
        return True, None
    except Exception as exc:
        return False, str(exc)


def train_lightgcn_model(
    train_enc: pd.DataFrame,
    num_users: int,
    num_items: int,
    embedding_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    reg: float = 1e-4,
    batch_size: int = 512,
    random_state: int = 42,
    device_name: str = "cpu",
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if train_enc.empty:
        raise ValueError("Train interaction kosong. LightGCN tidak dapat dilatih.")

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    class LightGCN(nn.Module):
        def __init__(self, n_users: int, n_items: int, dim: int, layers: int):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.layers = layers
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.normal_(self.user_emb.weight, std=0.1)
            nn.init.normal_(self.item_emb.weight, std=0.1)

        def propagate(self, graph_data):
            # Message passing LightGCN menggunakan edge list + index_add_ agar lebih ringan di CPU.
            row, col, norm = graph_data
            all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            embs = [all_emb]
            emb = all_emb
            for _ in range(self.layers):
                out = torch.zeros_like(emb)
                out.index_add_(0, row, emb[col] * norm.unsqueeze(1))
                emb = out
                embs.append(emb)
            final_emb = torch.stack(embs, dim=0).mean(dim=0)
            user_final, item_final = torch.split(final_emb, [self.n_users, self.n_items], dim=0)
            return user_final, item_final

    def build_graph_data():
        users = train_enc["user_idx"].astype(int).to_numpy()
        items = train_enc["item_idx"].astype(int).to_numpy() + num_users
        row = np.concatenate([users, items])
        col = np.concatenate([items, users])
        values = np.ones(len(row), dtype=np.float32)

        deg = np.bincount(row, minlength=num_users + num_items).astype(np.float32)
        deg[deg == 0] = 1.0
        norm_values = values / np.sqrt(deg[row] * deg[col])

        row_t = torch.tensor(row, dtype=torch.long, device=device)
        col_t = torch.tensor(col, dtype=torch.long, device=device)
        norm_t = torch.tensor(norm_values, dtype=torch.float32, device=device)
        return row_t, col_t, norm_t

    model = LightGCN(num_users, num_items, embedding_dim, num_layers).to(device)
    graph_data = build_graph_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_pairs = train_enc[["user_idx", "item_idx"]].astype(int).to_numpy()
    user_pos: Dict[int, set] = train_enc.groupby("user_idx")["item_idx"].apply(lambda x: set(x.astype(int))).to_dict()
    all_items = np.arange(num_items)

    def sample_negative(user_id: int) -> int:
        seen = user_pos.get(user_id, set())
        if len(seen) >= num_items:
            return int(np.random.randint(0, num_items))
        while True:
            neg = int(np.random.randint(0, num_items))
            if neg not in seen:
                return neg

    losses: List[float] = []
    max_samples_per_epoch = min(len(pos_pairs), max(512, int(batch_size) * 4))
    for _ in range(epochs):
        # Untuk Streamlit lokal, training dibuat mini-batch sampling per epoch,
        # tetapi propagasi graph hanya satu kali per epoch agar tidak terlalu lambat di CPU.
        if len(pos_pairs) > max_samples_per_epoch:
            sampled_idx = np.random.choice(len(pos_pairs), size=max_samples_per_epoch, replace=False)
            batch = pos_pairs[sampled_idx]
        else:
            batch = pos_pairs.copy()
        np.random.shuffle(batch)

        users_np = batch[:, 0]
        pos_np = batch[:, 1]
        neg_np = np.array([sample_negative(int(u)) for u in users_np], dtype=np.int64)

        users = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_items = torch.tensor(pos_np, dtype=torch.long, device=device)
        neg_items = torch.tensor(neg_np, dtype=torch.long, device=device)

        user_final, item_final = model.propagate(graph_data)
        u_emb = user_final[users]
        p_emb = item_final[pos_items]
        n_emb = item_final[neg_items]

        pos_scores = torch.sum(u_emb * p_emb, dim=1)
        neg_scores = torch.sum(u_emb * n_emb, dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg_loss = reg * (
            model.user_emb(users).pow(2).sum()
            + model.item_emb(pos_items).pow(2).sum()
            + model.item_emb(neg_items).pow(2).sum()
        ) / len(users)
        loss = bpr_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        user_final, item_final = model.propagate(graph_data)
        score_matrix = torch.matmul(user_final, item_final.T).detach().cpu().tolist()

    return {
        "model": model,
        "graph_data": graph_data,
        "score_matrix": score_matrix,
        "losses": losses,
        "device": str(device),
        "num_users": num_users,
        "num_items": num_items,
    }


def lightgcn_rank_item_indices(
    gnn_artifacts: dict,
    train_enc: pd.DataFrame,
    user_idx: int,
    top_n: int,
) -> List[Tuple[int, float]]:
    import torch

    if "score_matrix" in gnn_artifacts:
        scores = np.array(gnn_artifacts["score_matrix"][user_idx], dtype=float)
    else:
        model = gnn_artifacts["model"]
        graph_data = gnn_artifacts["graph_data"]
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.propagate(graph_data)
            scores = torch.matmul(user_emb[user_idx], item_emb.T).detach().cpu().tolist()

    seen = set(train_enc[train_enc["user_idx"] == user_idx]["item_idx"].astype(int))
    for iid in seen:
        if 0 <= iid < len(scores):
            scores[iid] = -np.inf
    top_idx = np.argsort(-scores)[:top_n]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


def recommend_lightgcn(
    df_hotel: pd.DataFrame,
    gnn_artifacts: dict,
    train_enc: pd.DataFrame,
    idx_to_item: dict,
    user_idx: int,
    top_n: int,
) -> pd.DataFrame:
    ranked = lightgcn_rank_item_indices(gnn_artifacts, train_enc, user_idx, top_n)
    if not ranked:
        return pd.DataFrame()
    scores = pd.DataFrame({"item_idx": [i for i, _ in ranked], "GNN_Score": [s for _, s in ranked]})
    scores["ID Hotel"] = scores["item_idx"].map(idx_to_item)
    result = scores.merge(df_hotel, on="ID Hotel", how="left")
    return result.sort_values("GNN_Score", ascending=False)


# ============================================================
# 11. STREAMLIT UI
# ============================================================

def main() -> None:
    st.title("🏨 Hotel Recommendation System")
    st.caption(
        "Content-Based, Collaborative Filtering, Hybrid, Random Walk/Personalized PageRank, dan LightGCN. "
        "Metode berbasis interaksi menggunakan train/test split yang terpisah."
    )

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload file Excel master hotel", type=["xlsx"])
        default_path = first_existing_data_path()
        use_default = st.checkbox("Gunakan file lokal default", value=True)
        if default_path:
            st.caption(f"File lokal terdeteksi: `{default_path}`")
        else:
            st.caption("File lokal default belum ditemukan.")

        st.divider()
        st.header("Parameter Rekomendasi")
        top_n = st.slider("Jumlah rekomendasi", 3, 30, 10)

        st.divider()
        st.header("Split Train/Test")
        rating_threshold = st.slider("Positive rating threshold", 1.0, 5.0, 4.0, 0.5)
        test_size = st.slider("Test size", 0.10, 0.50, 0.20, 0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    file_bytes = uploaded.getvalue() if uploaded is not None else None
    fallback_path = str(default_path) if (use_default and default_path is not None) else None

    if file_bytes is None and fallback_path is None:
        st.warning("Upload file Excel atau letakkan file di `data/master data hotel_200.xlsx`.")
        st.stop()

    try:
        with st.spinner("Memuat dan membersihkan data..."):
            df_hotel_raw, df_review = load_excel_data(file_bytes, fallback_path)
            df_hotel = prepare_hotel_with_reviews(df_hotel_raw, df_review)
            similarity_matrix = build_content_similarity(df_hotel)
            train_pos, test_pos = create_train_test_split(df_review, rating_threshold, test_size, int(random_state))
            train_enc, test_enc, user_to_idx, idx_to_user, item_to_idx, idx_to_item = encode_interactions(
                df_hotel, train_pos, test_pos
            )
            item_cf_sim_df = build_item_cf_similarity(train_pos)
            G_train = build_random_walk_graph(train_enc, len(user_to_idx), len(item_to_idx))
    except Exception as exc:
        st.error(f"Gagal memuat data: {exc}")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hotel", f"{len(df_hotel):,}")
    c2.metric("Review Valid", f"{len(df_review):,}")
    c3.metric("User/Reviewer", f"{df_review['name'].nunique():,}")
    c4.metric("Train Positif", f"{len(train_pos):,}")
    c5.metric("Test Positif", f"{len(test_pos):,}")

    hotel_options = df_hotel[["ID Hotel", "Nama Hotel"]].copy()
    hotel_options["label"] = hotel_options["ID Hotel"].astype(str) + " — " + hotel_options["Nama Hotel"]

    user_options = pd.DataFrame({"user_key": sorted(user_to_idx.keys())})
    user_options["user_idx"] = user_options["user_key"].map(user_to_idx)
    user_options["label"] = user_options["user_idx"].astype(str) + " — " + user_options["user_key"]

    tabs = st.tabs([
        "1. CBF Item-to-Item",
        "2. CBF Preference-Based",
        "3. Collaborative Filtering",
        "4. Hybrid CBF + CF",
        "5. Random Walk / PPR",
        "6. Data & Eval Random Walk",
        "7. GNN LightGCN",
        "8. Data & Eval GNN",
    ])

    with tabs[0]:
        st.subheader("1. Content-Based Item-to-Item")
        selected_label = st.selectbox("Pilih hotel acuan", hotel_options["label"], key="cbf_item")
        selected_id = int(selected_label.split(" — ")[0])
        selected, result = recommend_content_item_to_item(df_hotel, similarity_matrix, selected_id, top_n)
        show_selected_hotel(selected)
        st.dataframe(format_result(result), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("2. Content-Based Preference-Based")
        wilayah_list = sorted(df_hotel["Wilayah"].dropna().unique().tolist())
        predikat_list = ["Semua"] + sorted(df_hotel["Predikat"].dropna().unique().tolist())

        col_a, col_b, col_c = st.columns(3)
        wilayah_mode = col_a.radio("Mode wilayah", ["Semua", "Pilih dari data", "Ketik manual"], key="wilayah_mode")
        if wilayah_mode == "Pilih dari data":
            wilayah = col_a.selectbox("Wilayah", wilayah_list)
        elif wilayah_mode == "Ketik manual":
            wilayah = col_a.text_input("Keyword wilayah", value="") or None
        else:
            wilayah = None

        min_price, max_price = int(df_hotel["Harga"].min()), int(df_hotel["Harga"].max())
        harga_min, harga_max = col_b.slider(
            "Range harga",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=10000,
        )
        rating_min = col_c.slider(
            "Rating Traveloka minimum",
            min_value=float(df_hotel["Rating Traveloka"].min()),
            max_value=float(df_hotel["Rating Traveloka"].max()),
            value=float(df_hotel["Rating Traveloka"].min()),
            step=0.1,
        )
        predikat = col_c.selectbox("Predikat", predikat_list)

        with st.expander("Atur bobot scoring", expanded=False):
            w1, w2, w3, w4 = st.columns(4)
            w_rating = w1.slider("Bobot rating", 0.0, 1.0, 0.40, 0.05)
            w_harga = w2.slider("Bobot harga murah", 0.0, 1.0, 0.30, 0.05)
            w_wilayah = w3.slider("Bobot wilayah", 0.0, 1.0, 0.20, 0.05)
            w_predikat = w4.slider("Bobot predikat", 0.0, 1.0, 0.10, 0.05)

        result = recommend_content_preference_based(
            df_hotel,
            wilayah,
            harga_min,
            harga_max,
            rating_min,
            predikat,
            top_n,
            w_rating,
            w_harga,
            w_wilayah,
            w_predikat,
        )
        if result.empty:
            st.info("Tidak ada hotel yang cocok dengan filter tersebut.")
        else:
            st.dataframe(format_result(result), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("3. Collaborative Filtering")
        st.caption("Rekomendasi user-based menggunakan item-item collaborative similarity dari data train positif.")
        if user_options.empty:
            st.warning("Tidak ada user yang dapat digunakan untuk Collaborative Filtering.")
        else:
            user_label = st.selectbox("Pilih user/reviewer", user_options["label"].tolist(), key="cf_user")
            user_key = user_label.split(" — ", 1)[1]
            hist = train_pos[train_pos["user_key"] == user_key][["ID Hotel", "title", "stars"]].drop_duplicates()
            with st.expander("Interaksi train user", expanded=False):
                st.dataframe(hist, use_container_width=True, hide_index=True)
            result = recommend_collaborative(df_hotel, train_pos, item_cf_sim_df, user_key, top_n)
            if result.empty:
                st.info("Rekomendasi CF kosong. User mungkin memiliki interaksi train terlalu sedikit.")
            else:
                st.dataframe(format_result(result), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("4. Hybrid Content-Based + Collaborative Filtering")
        st.caption("Hybrid Score = alpha × Content Score + (1 - alpha) × CF Score.")
        alpha_content = st.slider("Alpha content-based", 0.0, 1.0, 0.50, 0.05, key="hybrid_alpha")
        if user_options.empty:
            st.warning("Tidak ada user yang dapat digunakan untuk Hybrid Recommendation.")
        else:
            user_label = st.selectbox("Pilih user/reviewer", user_options["label"].tolist(), key="hybrid_user")
            user_key = user_label.split(" — ", 1)[1]
            result = recommend_hybrid_user(df_hotel, train_pos, similarity_matrix, item_cf_sim_df, user_key, top_n, alpha_content)
            if result.empty:
                st.info("Rekomendasi hybrid kosong. User mungkin memiliki interaksi train terlalu sedikit.")
            else:
                st.dataframe(format_result(result), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("5. Random Walk / Personalized PageRank")
        st.caption("Graf bipartit user-hotel dibangun hanya dari data train positif.")
        pr_alpha = st.slider("Alpha PageRank", 0.50, 0.99, 0.85, 0.01, key="rw_alpha")
        if user_options.empty:
            st.warning("Tidak ada user yang dapat digunakan untuk Random Walk.")
        else:
            user_label = st.selectbox("Pilih user/reviewer", user_options["label"].tolist(), key="rw_user")
            selected_user_idx = int(user_label.split(" — ")[0])
            user_key = user_label.split(" — ", 1)[1]
            hist = train_pos[train_pos["user_key"] == user_key][["ID Hotel", "title", "stars"]].drop_duplicates()
            with st.expander("Interaksi train user", expanded=False):
                st.dataframe(hist, use_container_width=True, hide_index=True)
            result = recommend_random_walk(df_hotel, G_train, train_enc, idx_to_item, selected_user_idx, top_n, pr_alpha)
            if result.empty:
                st.info("Rekomendasi Random Walk kosong.")
            else:
                st.dataframe(format_result(result), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("6. Preview Data dan Evaluasi Random Walk")
        t1, t2, t3, t4 = st.tabs(["Traveloka Data", "Gmaps Review", "Train/Test Split", "Evaluasi Random Walk"])
        with t1:
            st.dataframe(df_hotel_raw, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(df_review, use_container_width=True, hide_index=True)
        with t3:
            c_train, c_test = st.columns(2)
            c_train.write("**Train positif**")
            c_train.dataframe(train_pos, use_container_width=True, hide_index=True)
            c_test.write("**Test positif**")
            c_test.dataframe(test_pos, use_container_width=True, hide_index=True)
        with t4:
            k_eval = st.slider("K evaluasi Random Walk", 3, 30, 10, key="rw_eval_k")
            eval_alpha = st.slider("Alpha PageRank evaluasi", 0.50, 0.99, 0.85, 0.01, key="rw_eval_alpha")
            if st.button("Jalankan Evaluasi Random Walk"):
                with st.spinner("Menghitung Precision@K, Recall@K, dan F1@K Random Walk..."):
                    metrics = evaluate_ranking(
                        train_enc,
                        test_enc,
                        idx_to_item,
                        k_eval,
                        ranker=lambda uid, kk: random_walk_rank_item_indices(
                            G_train, uid, train_enc, len(idx_to_item), kk, eval_alpha
                        ),
                    )
                display_metrics(metrics, k_eval)

    with tabs[6]:
        st.subheader("7. GNN Recommendation System — LightGCN")
        st.caption(
            "LightGCN dilatih pada graph user-hotel dari data train positif. "
            "Test positif hanya digunakan pada tab evaluasi."
        )
        ok_torch, torch_err = torch_available()
        if not ok_torch:
            st.error(f"PyTorch belum terpasang atau gagal diimpor: {torch_err}")
            st.code("pip install torch", language="bash")
        else:
            with st.expander("Parameter training LightGCN", expanded=True):
                g1, g2, g3, g4 = st.columns(4)
                emb_dim = g1.selectbox("Embedding dim", [16, 32, 64, 128], index=0)
                num_layers = g2.slider("Jumlah layer", 1, 4, 1)
                epochs = g3.slider("Epochs", 1, 100, 5, 1)
                lr = g4.selectbox("Learning rate", [0.001, 0.003, 0.005, 0.01, 0.02], index=3)
                b1, b2, b3 = st.columns(3)
                batch_size = b1.selectbox("Batch size", [128, 256, 512, 1024], index=2)
                reg = b2.selectbox("L2 regularization", [1e-5, 1e-4, 1e-3], index=1, format_func=lambda x: f"{x:.0e}")
                device = b3.selectbox("Device", ["cpu", "cuda"], index=0)

            if st.button("Latih / Refresh Model LightGCN"):
                with st.spinner("Melatih LightGCN..."):
                    try:
                        st.session_state["gnn_artifacts"] = train_lightgcn_model(
                            train_enc,
                            len(user_to_idx),
                            len(item_to_idx),
                            embedding_dim=int(emb_dim),
                            num_layers=int(num_layers),
                            epochs=int(epochs),
                            lr=float(lr),
                            reg=float(reg),
                            batch_size=int(batch_size),
                            random_state=int(random_state),
                            device_name=str(device),
                        )
                        st.success("Model LightGCN selesai dilatih.")
                    except Exception as exc:
                        st.error(f"Training LightGCN gagal: {exc}")

            gnn_artifacts = st.session_state.get("gnn_artifacts")
            if gnn_artifacts:
                st.write(f"Device model: `{gnn_artifacts['device']}`")
                loss_df = pd.DataFrame({"epoch": np.arange(1, len(gnn_artifacts["losses"]) + 1), "loss": gnn_artifacts["losses"]})
                st.line_chart(loss_df, x="epoch", y="loss")

                user_label = st.selectbox("Pilih user/reviewer", user_options["label"].tolist(), key="gnn_user")
                selected_user_idx = int(user_label.split(" — ")[0])
                result = recommend_lightgcn(df_hotel, gnn_artifacts, train_enc, idx_to_item, selected_user_idx, top_n)
                if result.empty:
                    st.info("Rekomendasi LightGCN kosong.")
                else:
                    st.dataframe(format_result(result), use_container_width=True, hide_index=True)
            else:
                st.info("Latih model LightGCN terlebih dahulu untuk menampilkan rekomendasi GNN.")

    with tabs[7]:
        st.subheader("8. Preview Data dan Evaluasi LightGCN")
        t1, t2, t3 = st.tabs(["Train/Test Split", "Evaluasi GNN", "Catatan Model"])
        with t1:
            c_train, c_test = st.columns(2)
            c_train.write("**Train positif**")
            c_train.dataframe(train_pos, use_container_width=True, hide_index=True)
            c_test.write("**Test positif**")
            c_test.dataframe(test_pos, use_container_width=True, hide_index=True)
        with t2:
            gnn_artifacts = st.session_state.get("gnn_artifacts")
            if not gnn_artifacts:
                st.info("Latih model LightGCN pada tab 7 terlebih dahulu.")
            else:
                k_gnn = st.slider("K evaluasi GNN", 3, 30, 10, key="gnn_eval_k")
                if st.button("Jalankan Evaluasi LightGCN"):
                    with st.spinner("Menghitung Precision@K, Recall@K, dan F1@K LightGCN..."):
                        metrics = evaluate_ranking(
                            train_enc,
                            test_enc,
                            idx_to_item,
                            k_gnn,
                            ranker=lambda uid, kk: lightgcn_rank_item_indices(
                                gnn_artifacts, train_enc, uid, kk
                            ),
                        )
                    display_metrics(metrics, k_gnn)
        with t3:
            st.markdown(
                """
                **LightGCN** menggunakan propagasi embedding pada graph user-item tanpa transformasi fitur dan aktivasi non-linear.
                Dalam aplikasi ini, node user berasal dari kolom `name`, node item berasal dari `ID Hotel`, dan edge positif berasal dari `stars >= threshold`.

                Pemisahan train/test dilakukan sebelum graph dibangun. Karena itu, edge pada data test tidak masuk ke graph training maupun user-item matrix.
                """
            )


if __name__ == "__main__":
    main()
