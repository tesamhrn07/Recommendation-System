# RecSys Project — Hotel Recommendation System

Aplikasi Streamlit untuk sistem rekomendasi hotel berbasis:

1. Content-Based Item-to-Item
2. Content-Based Preference-Based
3. Collaborative Filtering
4. Hybrid Content-Based + Collaborative Filtering
5. Random Walk / Personalized PageRank
6. Evaluasi Random Walk dengan Precision@K, Recall@K, F1@K
7. GNN Recommendation berbasis LightGCN
8. Evaluasi GNN dengan Precision@K, Recall@K, F1@K

## Struktur Folder

```text
RecSys Project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── master data hotel_200.xlsx
├── models/
└── outputs/
```

## Cara Menjalankan Lokal

```bash
cd "RecSys Project"
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

File default dibaca dari:

```text
data/master data hotel_200.xlsx
```

Sheet yang digunakan:

- `Traveloka Data`: data master hotel.
- `Gmaps Review`: data review/rating user.

## Split Data

Metode Collaborative Filtering, Hybrid, Random Walk, dan LightGCN menggunakan data interaksi positif yang dipisah menjadi train dan test.
Interaksi positif default: `stars >= 4`.

- Train: dipakai untuk membangun user-item matrix, graph, dan model LightGCN.
- Test: dipakai hanya untuk evaluasi ranking Precision@K, Recall@K, dan F1@K.

Pengaturan `rating threshold`, `test size`, dan `random state` tersedia di sidebar aplikasi.
