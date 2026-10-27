"""
Dashboard Streamlit: Linear Regression vs Random Forest untuk Prediksi Penjualan UMKM
Mengikuti alur & grafik pada notebook
"Analisis_LinearRegression_vs_RandomForest_Penjualan_UMKM.ipynb"

Jalankan:  streamlit run app.py
"""
import io
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
NAMA_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
FITUR = ["is_weekend", "bulan_sin", "bulan_cos", "trend",
         "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_28"]
TARGET = "total_terjual"
DATA_DEFAULT = Path(__file__).parent / "data_penjualan_last_two_tea.xls"

st.set_page_config(page_title="Prediksi Penjualan UMKM", page_icon="🧋", layout="wide")


# =====================================================================
# 1. MEMBACA FILE (CSV / XLS / XLSX — format dideteksi dari isi file)
# =====================================================================
def muat_file(isi: bytes) -> pd.DataFrame:
    """Deteksi format dari isi file, bukan dari ekstensi.
    (File 'data_penjualan_last_two_tea.xls' ternyata teks CSV.)"""
    if isi[:2] == b"PK":                                   # .xlsx
        return pd.read_excel(io.BytesIO(isi), engine="openpyxl")
    if isi[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":      # .xls biner asli
        return pd.read_excel(io.BytesIO(isi), engine="xlrd")

    if isi[:2] in (b"\xff\xfe", b"\xfe\xff"):
        teks = isi.decode("utf-16")
    else:
        try:
            teks = isi.decode("utf-8-sig")
        except UnicodeDecodeError:
            teks = isi.decode("latin-1")
    baris1 = teks.splitlines()[0] if teks.strip() else ""
    sep = max([",", ";", "\t", "|"], key=baris1.count)
    return pd.read_csv(io.StringIO(teks), sep=sep)


# =====================================================================
# 2. PEMBERSIHAN OTOMATIS
# =====================================================================
ALIAS = {
    "tanggal": ["tanggal", "tgl", "date", "tanggal_transaksi", "waktu"],
    "nama_produk": ["nama_produk", "produk", "product", "nama_menu", "menu", "item", "nama_barang"],
    "harga_jual": ["harga_jual", "harga", "price", "harga_satuan"],
    "jumlah_terjual": ["jumlah_terjual", "jumlah", "qty", "quantity", "terjual", "jml"],
}


def _norm_kolom(c) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")


def _ke_angka(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace(r"(?i)rp\.?|\s", "", regex=True)
    titik = t.str.match(r"^-?\d{1,3}(\.\d{3})+$")      # 3.000  -> 3000
    koma = t.str.match(r"^-?\d{1,3}(,\d{3})+$")        # 3,000  -> 3000
    t = t.where(~titik, t.str.replace(".", "", regex=False))
    t = t.where(~koma, t.str.replace(",", "", regex=False))
    t = t.str.replace(",", ".", regex=False)           # 2,5 -> 2.5
    return pd.to_numeric(t, errors="coerce")


def _parse_tanggal(s: pd.Series):
    """Mengembalikan (Series datetime, keterangan format)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize(), "sudah bertipe tanggal"
    teks = s.astype(str).str.strip().str.split(r"[ T]", n=1, regex=True).str[0]

    if teks.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$").mean() > 0.9:
        return pd.to_datetime(teks, format="%Y-%m-%d", errors="coerce"), "YYYY-MM-DD (ISO)"

    teks = teks.str.replace(r"[-.]", "/", regex=True)
    bagian = teks.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$").astype(float)
    if bagian.notna().all(axis=1).mean() > 0.9:
        a, b = bagian[0].max(), bagian[1].max()
        tahun = "%Y" if bagian[2].max() > 99 else "%y"
        if a > 12:
            fmt, ket = f"%d/%m/{tahun}", "hari/bulan/tahun (bagian-1 maks > 12)"
        elif b > 12:
            fmt, ket = f"%m/%d/{tahun}", "bulan/hari/tahun (bagian-2 maks > 12)"
        else:
            fmt, ket = f"%m/%d/{tahun}", "bulan/hari/tahun (ambigu, diasumsikan sama seperti notebook)"
        return pd.to_datetime(teks, format=fmt, errors="coerce"), ket

    return pd.to_datetime(teks, errors="coerce"), "deteksi otomatis pandas"


@st.cache_data(show_spinner=False)
def bersihkan(raw: pd.DataFrame, lengkapi_kombinasi: bool):
    """Pembersihan otomatis. Mengembalikan (df_bersih, tabel_log)."""
    log = []

    def catat(langkah, hasil, diperbaiki=False):
        log.append({"Langkah": langkah, "Hasil": hasil,
                    "Status": "⚠️ Diperbaiki" if diperbaiki else "✅ OK"})

    df = raw.copy()
    catat("Data awal dibaca", f"{len(df):,} baris × {df.shape[1]} kolom")

    # --- nama kolom ---
    df.columns = [_norm_kolom(c) for c in df.columns]
    rename = {}
    for baku, daftar in ALIAS.items():
        for kolom in df.columns:
            if kolom in daftar and baku not in df.columns and kolom not in rename:
                rename[kolom] = baku
                break
    df = df.rename(columns=rename)
    hilang = [k for k in ALIAS if k not in df.columns]
    if hilang:
        raise ValueError(f"Kolom wajib tidak ditemukan: {hilang}. Kolom di file: {list(raw.columns)}")
    df = df[list(ALIAS)]
    catat("Standarisasi nama kolom", "tanggal, nama_produk, harga_jual, jumlah_terjual",
          bool(rename) or list(raw.columns) != list(df.columns))

    # --- baris kosong ---
    n = len(df)
    df = df.dropna(how="all")
    catat("Hapus baris kosong total", f"{n - len(df):,} baris dihapus", n != len(df))

    # --- nama produk ---
    asli = df["nama_produk"].astype(str)
    rapi = asli.str.strip().str.replace(r"\s+", " ", regex=True)
    # ejaan berbeda huruf besar/kecil → pakai ejaan yang paling sering muncul
    kanon = rapi.groupby(rapi.str.lower()).agg(lambda x: x.value_counts().index[0])
    df["nama_produk"] = rapi.str.lower().map(kanon)
    ubah = int((asli != df["nama_produk"]).sum())
    catat("Rapikan nama produk (spasi & huruf besar/kecil)", f"{ubah:,} nilai diseragamkan", ubah > 0)

    # --- angka ---
    for k in ["harga_jual", "jumlah_terjual"]:
        sebelum = df[k].notna().sum()
        df[k] = _ke_angka(df[k])
        gagal = int(sebelum - df[k].notna().sum())
        catat(f"Konversi `{k}` ke angka", f"{gagal:,} nilai tidak valid → kosong", gagal > 0)

    # --- tanggal ---
    sebelum = df["tanggal"].notna().sum()
    df["tanggal"], fmt = _parse_tanggal(df["tanggal"])
    gagal = int(sebelum - df["tanggal"].notna().sum())
    catat("Konversi `tanggal` ke datetime", f"format terdeteksi: {fmt}; {gagal:,} tanggal tidak valid", gagal > 0)

    # --- nilai kosong ---
    n = len(df)
    df = df.dropna(subset=["tanggal", "nama_produk", "jumlah_terjual", "harga_jual"])
    catat("Hapus baris dengan nilai kosong", f"{n - len(df):,} baris dihapus", n != len(df))

    # --- nilai negatif ---
    n = len(df)
    df = df[(df["harga_jual"] >= 0) & (df["jumlah_terjual"] >= 0)]
    catat("Hapus nilai negatif", f"{n - len(df):,} baris dihapus", n != len(df))

    # --- duplikat ---
    n = len(df)
    df = df.drop_duplicates()
    catat("Hapus baris duplikat persis", f"{n - len(df):,} baris dihapus", n != len(df))

    n = len(df)
    df = (df.groupby(["tanggal", "nama_produk"], as_index=False)
            .agg(harga_jual=("harga_jual", "max"), jumlah_terjual=("jumlah_terjual", "sum")))
    catat("Gabungkan duplikat (tanggal, produk)", f"{n - len(df):,} baris digabung (jumlah dijumlahkan)", n != len(df))

    # --- kelengkapan kombinasi tanggal x produk ---
    n_produk, n_hari = df["nama_produk"].nunique(), df["tanggal"].nunique()
    kurang = n_produk * n_hari - len(df)
    if kurang > 0 and lengkapi_kombinasi:
        modus = df.groupby("nama_produk")["harga_jual"].agg(lambda x: x.mode().iloc[0])
        idx = pd.MultiIndex.from_product([sorted(df["tanggal"].unique()), sorted(df["nama_produk"].unique())],
                                         names=["tanggal", "nama_produk"])
        df = df.set_index(["tanggal", "nama_produk"]).reindex(idx).reset_index()
        df["jumlah_terjual"] = df["jumlah_terjual"].fillna(0)
        df["harga_jual"] = df["harga_jual"].fillna(df["nama_produk"].map(modus))
        catat("Lengkapi kombinasi tanggal × produk",
              f"{kurang:,} baris yang hilang ditambahkan dengan jumlah_terjual = 0", True)
    elif kurang > 0:
        catat("Kelengkapan tanggal × produk", f"{kurang:,} kombinasi hilang (tidak diisi)", True)
    else:
        catat("Kelengkapan tanggal × produk", f"lengkap: {n_produk} produk × {n_hari} hari = {n_produk * n_hari:,} baris")

    # --- harga konsisten ---
    berubah = int((df.groupby("nama_produk")["harga_jual"].nunique() > 1).sum())
    catat("Konsistensi harga per produk", f"{berubah} produk dengan harga berubah-ubah")

    # --- tipe data akhir ---
    if np.allclose(df["jumlah_terjual"], df["jumlah_terjual"].round()):
        df["jumlah_terjual"] = df["jumlah_terjual"].round().astype(int)
    if np.allclose(df["harga_jual"], df["harga_jual"].round()):
        df["harga_jual"] = df["harga_jual"].round().astype(int)

    catat("Baris bernilai 0 (tidak laku)",
          f"{(df['jumlah_terjual'] == 0).mean():.1%} — wajar, tidak dihapus")

    # --- kolom turunan (notebook 6.3) ---
    df["omzet"] = df["harga_jual"] * df["jumlah_terjual"]
    df["ukuran"] = df["nama_produk"].str.extract(r"Cap\s*(\d+)")[0].fillna("Lainnya")
    df["ukuran"] = df["ukuran"].replace({"18": "Cap 18", "22": "Cap 22"})

    df = df.sort_values(["tanggal", "nama_produk"]).reset_index(drop=True)
    catat("Data bersih siap dianalisis", f"{len(df):,} baris")
    return df, pd.DataFrame(log)


@st.cache_data(show_spinner=False)
def agregasi_harian(df: pd.DataFrame, isi_hari_kosong: bool):
    harian = (df.groupby("tanggal")
                .agg(total_terjual=("jumlah_terjual", "sum"), omzet=("omzet", "sum"))
                .sort_index()
                .asfreq("D"))
    kosong = int(harian["total_terjual"].isna().sum())
    if isi_hari_kosong and kosong:
        harian = harian.interpolate("linear").round(0)
    return harian, kosong


def buat_fitur(harian: pd.DataFrame) -> pd.DataFrame:
    """Rekayasa fitur — identik dengan notebook bagian 8."""
    data = harian[[TARGET]].copy()
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
    data["bulan_sin"] = np.sin(2 * np.pi * data.index.month / 12)
    data["bulan_cos"] = np.cos(2 * np.pi * data.index.month / 12)
    data["trend"] = np.arange(len(data))
    for k in [1, 7, 14]:
        data[f"lag_{k}"] = data[TARGET].shift(k)
    data["rolling_mean_7"] = data[TARGET].shift(1).rolling(7).mean()      # shift(1) → tanpa data leakage
    data["rolling_mean_28"] = data[TARGET].shift(1).rolling(28).mean()
    return data.dropna()


# =====================================================================
# 3. PEMODELAN
# =====================================================================
def evaluasi(nama, y_true, y_pred):
    return {
        "Model": nama,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R²": r2_score(y_true, y_pred),
    }


@st.cache_data(show_spinner=False)
def latih_model(data: pd.DataFrame, rasio_latih: float, tuning: bool, n_splits: int):
    split = int(len(data) * rasio_latih)
    X_train, y_train = data[FITUR].iloc[:split], data[TARGET].iloc[:split]
    X_test, y_test = data[FITUR].iloc[split:], data[TARGET].iloc[split:]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    hasil = []

    # Baseline
    hasil.append(evaluasi("Baseline: Rata-rata historis", y_test, np.full(len(y_test), y_train.mean())))
    hasil.append(evaluasi("Baseline: Naif (minggu lalu)", y_test, X_test["lag_7"].values))

    # Linear Regression
    model_lr = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    hasil.append(evaluasi("Linear Regression", y_test, pred_lr))

    # Random Forest default
    rf_default = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    hasil.append(evaluasi("Random Forest (default)", y_test, rf_default.predict(X_test)))

    # Random Forest tuned
    if tuning:
        grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            param_grid={"n_estimators": [200, 400], "max_depth": [None, 5, 10],
                        "min_samples_leaf": [1, 5, 10], "max_features": [1.0, 0.5]},
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0,
        ).fit(X_train, y_train)
        model_rf, best_params, cv_mae_grid = grid.best_estimator_, grid.best_params_, -grid.best_score_
        rf_label = "Random Forest (tuned)"
    else:
        model_rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                         random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
        best_params, cv_mae_grid = {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5}, np.nan
        rf_label = "Random Forest (parameter cepat)"
    pred_rf = model_rf.predict(X_test)
    hasil.append(evaluasi(rf_label, y_test, pred_rf))
    tabel = pd.DataFrame(hasil).set_index("Model")

    # Residual
    resid = pd.DataFrame({"Linear Regression": y_test.values - pred_lr,
                          rf_label: y_test.values - pred_rf}, index=y_test.index)

    # Error hari kerja vs akhir pekan
    cek = pd.DataFrame({
        "Jenis hari": np.where(X_test["is_weekend"] == 1, "Akhir pekan", "Hari kerja"),
        "MAE Linear Regression": np.abs(resid["Linear Regression"].values),
        "MAE Random Forest": np.abs(resid[rf_label].values),
    }, index=X_test.index).groupby("Jenis hari").mean().round(2)

    # Wilcoxon
    try:
        stat, p_value = wilcoxon(np.abs(resid["Linear Regression"].values), np.abs(resid[rf_label].values))
    except ValueError:
        stat, p_value = np.nan, np.nan

    # Validasi silang
    def cv_mae(model, nama):
        skor = cross_validate(model, X_train, y_train, cv=tscv,
                              scoring=["neg_mean_absolute_error", "r2"], n_jobs=-1)
        mae = -skor["test_neg_mean_absolute_error"]
        return {"Model": nama, "MAE rata-rata": mae.mean(), "MAE std": mae.std(),
                "R² rata-rata": skor["test_r2"].mean()}
    cv_tabel = pd.DataFrame([cv_mae(model_lr, "Linear Regression"), cv_mae(model_rf, rf_label)]).set_index("Model")

    # Interpretasi fitur
    koef = pd.Series(model_lr.named_steps["reg"].coef_, index=FITUR).sort_values()
    perm = permutation_importance(model_rf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE,
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
    pentingnya = pd.Series(perm.importances_mean, index=FITUR).sort_values()

    return dict(split=split, y_train=y_train, y_test=y_test, X_test=X_test, pred_lr=pred_lr, pred_rf=pred_rf,
                tabel=tabel, resid=resid, cek=cek, stat=stat, p_value=p_value, cv_tabel=cv_tabel,
                koef=koef, pentingnya=pentingnya, best_params=best_params, cv_mae_grid=cv_mae_grid,
                rf_label=rf_label, X_train_index=X_train.index)


@st.cache_data(show_spinner=False)
def bangun_ekspor(long_bersih, harian_model, prediksi_uji):
    """Notebook bagian 16: Excel 3 sheet + ZIP berisi 3 CSV."""
    xlsx = io.BytesIO()
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        long_bersih.to_excel(xw, sheet_name="per_produk", index=False)
        harian_model.to_excel(xw, sheet_name="harian_model", index=False)
        prediksi_uji.to_excel(xw, sheet_name="prediksi_uji", index=False)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("1_data_bersih_per_produk.csv", long_bersih.to_csv(index=False))
        z.writestr("2_data_harian_untuk_model.csv", harian_model.to_csv(index=False))
        z.writestr("3_prediksi_data_uji.csv", prediksi_uji.to_csv(index=False))
    return xlsx.getvalue(), zbuf.getvalue()


def tampil(fig):
    st.pyplot(fig)
    plt.close(fig)


def gaya_tabel(tabel):
    return (tabel.style.format("{:.2f}")
            .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#c7e9c0")
            .highlight_max(subset=["R²"], color="#c7e9c0"))


# =====================================================================
# 4. ANTARMUKA
# =====================================================================
st.title("🧋 Prediksi Penjualan UMKM Es Teh")
st.caption("Perbandingan **Linear Regression** vs **Random Forest Regression** untuk memprediksi "
           "total gelas terjual per hari — mengikuti alur notebook analisis.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    berkas = st.file_uploader("Unggah data penjualan (CSV / XLS / XLSX)", type=["csv", "xls", "xlsx", "txt"])
    st.caption("Kolom wajib: `tanggal`, `nama_produk`, `harga_jual`, `jumlah_terjual`.")
    st.divider()
    st.subheader("Pembersihan")
    lengkapi = st.checkbox("Lengkapi kombinasi tanggal × produk yang hilang (jumlah = 0)", value=True)
    isi_hari = st.checkbox("Isi hari tanpa data dengan interpolasi linear", value=True)
    st.subheader("Pemodelan")
    rasio = st.slider("Porsi data latih", 0.60, 0.90, 0.80, 0.05,
                      help="Pembagian kronologis (tanpa diacak). Notebook memakai 80%.")
    tuning = st.checkbox("Tuning Random Forest dengan GridSearchCV", value=True,
                         help="Sesuai notebook (36 kombinasi × 5 lipatan). Hasil di-cache; matikan jika ingin lebih cepat.")
    n_splits = st.slider("Jumlah lipatan TimeSeriesSplit", 3, 8, 5)

# --- sumber data ---
if berkas is not None:
    isi_bytes, nama_sumber = berkas.getvalue(), berkas.name
elif DATA_DEFAULT.exists():
    isi_bytes, nama_sumber = DATA_DEFAULT.read_bytes(), DATA_DEFAULT.name + " (data bawaan)"
else:
    st.info("Unggah file data penjualan di sidebar untuk memulai.")
    st.stop()

try:
    raw = muat_file(isi_bytes)
    df, log = bersihkan(raw, lengkapi)
except Exception as e:
    st.error(f"Gagal membaca/membersihkan data: {e}")
    st.stop()

harian, hari_hilang = agregasi_harian(df, isi_hari)
data = buat_fitur(harian)
if len(data) < 90:
    st.error(f"Data harian terlalu sedikit untuk pemodelan ({len(data)} hari setelah rekayasa fitur; minimal ±90).")
    st.stop()

st.success(f"Sumber data: **{nama_sumber}** — {len(raw):,} baris mentah → {len(df):,} baris bersih, "
           f"{harian.index.min().date()} s/d {harian.index.max().date()}.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Data & Pembersihan", "📈 EDA", "🤖 Model & Evaluasi", "🔍 Interpretasi Fitur", "✅ Kesimpulan & Ekspor"])

# ---------------------------------------------------------------------
# TAB 1 — DATA & PEMBERSIHAN (notebook bagian 4–6)
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Log pembersihan otomatis")
    st.dataframe(log, hide_index=True)

    st.subheader("Pemahaman data")
    c = st.columns(4)
    c[0].metric("Baris × Kolom", f"{df.shape[0]:,} × {df.shape[1]}")
    c[1].metric("Produk unik", df["nama_produk"].nunique())
    c[2].metric("Hari unik", df["tanggal"].nunique())
    c[3].metric("Hari hilang di rentang", hari_hilang)

    st.markdown("**Contoh data bersih**")
    st.dataframe(df[["tanggal", "nama_produk", "harga_jual", "jumlah_terjual", "omzet", "ukuran"]].head(10),
                 hide_index=True)

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Tipe & kelengkapan kolom**")
        info = pd.DataFrame({"Tipe": df.dtypes.astype(str), "Non-null": df.notna().sum(),
                             "Kosong": df.isna().sum()})
        st.dataframe(info)
    with k2:
        st.markdown("**Statistik deskriptif**")
        st.dataframe(df[["harga_jual", "jumlah_terjual", "omzet"]].describe().round(2))

    st.markdown("**Total penjualan harian (target prediksi)**")
    st.dataframe(harian.head(10))
    st.caption("Data berbentuk *long format* (satu baris = satu produk pada satu hari). "
               "Diagregasi menjadi total harian karena penjualan per produk per hari berupa angka kecil dan sangat acak.")

# ---------------------------------------------------------------------
# TAB 2 — EDA (notebook bagian 7)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("7.1 Tren penjualan harian")
    fig = plt.figure(figsize=(14, 5))
    plt.plot(harian.index, harian[TARGET], alpha=0.35, label="Harian")
    plt.plot(harian.index, harian[TARGET].rolling(7).mean(), lw=2, label="Rata-rata 7 hari")
    plt.plot(harian.index, harian[TARGET].rolling(30).mean(), lw=2.5, label="Rata-rata 30 hari")
    plt.title("Total Gelas Terjual per Hari")
    plt.xlabel("Tanggal"); plt.ylabel("Gelas terjual")
    plt.legend(); plt.tight_layout()
    tampil(fig)

    st.subheader("7.2 Pola hari dalam seminggu dan pola bulanan")
    eda = harian.copy()
    eda["hari"] = eda.index.dayofweek
    rata_hari = eda.groupby("hari")[TARGET].mean().reindex(range(7))
    rata_bulan = eda[TARGET].resample("MS").mean()

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.barplot(x=NAMA_HARI, y=rata_hari.values, ax=ax[0], color="#4C72B0")
    ax[0].set_title("Rata-rata Penjualan per Hari dalam Seminggu")
    ax[0].set_ylabel("Gelas terjual")
    ax[0].set_ylim(np.nanmin(rata_hari) * 0.8, np.nanmax(rata_hari) * 1.05)
    ax[1].plot(rata_bulan.index, rata_bulan.values, marker="o")
    ax[1].set_title("Rata-rata Penjualan Harian per Bulan")
    ax[1].set_ylabel("Gelas terjual")
    plt.tight_layout()
    tampil(fig)
    st.dataframe(rata_hari.rename(dict(enumerate(NAMA_HARI))).round(1).rename("Rata-rata gelas").to_frame().T)

    akhir_pekan = rata_hari.iloc[5:].mean()
    hari_kerja = rata_hari.iloc[:5].mean()
    paruh = len(harian) // 2
    st.info(f"Rata-rata **akhir pekan {akhir_pekan:.1f}** gelas vs **hari kerja {hari_kerja:.1f}** gelas "
            f"({(akhir_pekan / hari_kerja - 1) * 100:+.1f}%). Rata-rata paruh awal periode "
            f"{harian[TARGET].iloc[:paruh].mean():.1f} → paruh akhir {harian[TARGET].iloc[paruh:].mean():.1f} gelas/hari.")

    st.subheader("7.3 Produk terlaris & komposisi ukuran gelas")
    per_produk = df.groupby("nama_produk")["jumlah_terjual"].sum().sort_values(ascending=False)
    per_ukuran = df.groupby("ukuran")["jumlah_terjual"].sum()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 1]})
    per_produk.head(10).sort_values().plot.barh(ax=ax[0], color="#55A868")
    ax[0].set_title("10 Produk Terlaris (total gelas, seluruh periode)")
    ax[0].set_xlabel("Gelas terjual"); ax[0].set_ylabel("")
    ax[1].pie(per_ukuran.values, labels=per_ukuran.index, autopct="%1.1f%%", startangle=90)
    ax[1].set_title("Proporsi Penjualan per Ukuran")
    plt.tight_layout()
    tampil(fig)

    st.subheader("7.4 Sebaran data & autokorelasi")
    lags = range(1, 31)
    acf = [harian[TARGET].autocorr(lag=k) for k in lags]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.histplot(harian[TARGET].dropna(), bins=30, kde=True, ax=ax[0], color="#C44E52")
    ax[0].set_title("Distribusi Total Penjualan Harian"); ax[0].set_xlabel("Gelas terjual")
    ax[1].bar(lags, acf, color="#8172B2")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_title("Autokorelasi Penjualan Harian (lag 1-30 hari)")
    ax[1].set_xlabel("Lag (hari)"); ax[1].set_ylabel("Korelasi")
    plt.tight_layout()
    tampil(fig)
    top_lag = sorted(zip(acf, lags), reverse=True)[:3]
    st.caption("Lag dengan autokorelasi tertinggi: " + ", ".join(f"lag {l} ({a:.2f})" for a, l in top_lag)
               + ". Lonjakan di kelipatan 7 menandakan pola mingguan — dasar pemilihan fitur lag.")

# ---------------------------------------------------------------------
# Latih model (dipakai tab 3–5)
# ---------------------------------------------------------------------
with st.spinner("Melatih Linear Regression & Random Forest"
                + (" (GridSearchCV bisa memakan waktu ±1 menit pada eksekusi pertama)…" if tuning else "…")):
    R = latih_model(data, rasio, tuning, n_splits)

tabel, y_test, resid, rf_label = R["tabel"], R["y_test"], R["resid"], R["rf_label"]
pred_lr, pred_rf = R["pred_lr"], R["pred_rf"]

# ---------------------------------------------------------------------
# TAB 3 — MODEL & EVALUASI (notebook bagian 8–13)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("8–9. Fitur & pembagian data latih/uji (kronologis)")
    c = st.columns(3)
    c[0].metric("Data siap model", f"{len(data)} hari")
    c[1].metric("Data latih", f"{R['split']} hari",
                f"{R['X_train_index'].min().date()} – {R['X_train_index'].max().date()}", delta_color="off")
    c[2].metric("Data uji", f"{len(y_test)} hari",
                f"{y_test.index.min().date()} – {y_test.index.max().date()}", delta_color="off")
    st.caption("Fitur: " + ", ".join(f"`{f}`" for f in FITUR))
    st.dataframe(data.head())

    st.subheader("Hyperparameter Random Forest")
    if tuning:
        st.write(f"Parameter terbaik GridSearchCV: `{R['best_params']}` — "
                 f"MAE validasi silang (data latih): **{R['cv_mae_grid']:.2f}** gelas")
    else:
        st.write(f"Parameter tetap (tanpa tuning): `{R['best_params']}`")

    st.subheader("13. Perbandingan model pada data uji")
    st.dataframe(gaya_tabel(tabel))
    st.caption("Semakin kecil MAE, RMSE, MAPE dan semakin besar R² → semakin baik (hijau = terbaik).")

    st.subheader("13.1 Visualisasi perbandingan metrik")
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    for a, metrik in zip(ax, ["MAE", "RMSE", "MAPE (%)", "R²"]):
        tabel[metrik].plot.barh(ax=a, color=sns.color_palette("Set2", len(tabel)))
        a.set_title(metrik); a.set_ylabel("")
        if a is not ax[0]:
            a.set_yticklabels([])
    plt.tight_layout()
    tampil(fig)

    st.subheader("13.2 Prediksi vs aktual pada periode uji")
    fig = plt.figure(figsize=(15, 5))
    plt.plot(y_test.index, y_test.values, color="black", lw=2, label="Aktual")
    plt.plot(y_test.index, pred_lr, lw=1.6, label="Linear Regression")
    plt.plot(y_test.index, pred_rf, lw=1.6, label=rf_label)
    plt.title("Prediksi vs Aktual — Data Uji")
    plt.xlabel("Tanggal"); plt.ylabel("Gelas terjual")
    plt.legend(); plt.tight_layout()
    tampil(fig)

    st.subheader("13.3 Scatter plot aktual vs prediksi")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    batas = [min(y_test.min(), pred_lr.min(), pred_rf.min()) - 3,
             max(y_test.max(), pred_lr.max(), pred_rf.max()) + 3]
    for a, pred, judul in zip(ax, [pred_lr, pred_rf], ["Linear Regression", rf_label]):
        a.scatter(y_test, pred, alpha=0.6)
        a.plot(batas, batas, "r--", label="Prediksi sempurna")
        a.set_title(judul); a.set_xlabel("Aktual"); a.set_ylabel("Prediksi"); a.legend()
    plt.tight_layout()
    tampil(fig)

    st.subheader("13.4 Analisis residual")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for kolom in resid.columns:
        sns.histplot(resid[kolom], kde=True, alpha=0.4, label=kolom, ax=ax[0])
    ax[0].axvline(0, color="black", ls="--"); ax[0].set_title("Distribusi Residual"); ax[0].legend()
    sns.boxplot(data=resid, ax=ax[1])
    ax[1].axhline(0, color="black", ls="--"); ax[1].set_title("Boxplot Residual")
    plt.tight_layout()
    tampil(fig)
    st.markdown("**Rata-rata residual (bias)** — positif = model cenderung menebak terlalu rendah")
    st.dataframe(resid.mean().round(2).rename("Rata-rata residual").to_frame().T)

    st.subheader("13.5 Error pada hari kerja vs akhir pekan")
    st.dataframe(R["cek"])

    st.subheader("13.6 Uji signifikansi (Wilcoxon signed-rank)")
    if np.isnan(R["p_value"]):
        st.warning("Uji Wilcoxon tidak dapat dihitung (selisih error kedua model nol / data terlalu sedikit).")
    else:
        sig = R["p_value"] < 0.05
        st.write(f"Statistik Wilcoxon = **{R['stat']:.1f}** | p-value = **{R['p_value']:.4f}**")
        (st.warning if sig else st.info)(
            "Perbedaan error kedua model **signifikan** (p < 0,05)." if sig
            else "Perbedaan error kedua model **tidak signifikan** (p ≥ 0,05) — akurasinya setara secara statistik.")

    st.subheader("13.7 Validasi silang pada data latih (kestabilan)")
    st.dataframe(R["cv_tabel"].style.format("{:.2f}"))
    st.caption("Parameter Random Forest dipilih dari lipatan yang sama, sehingga skornya sedikit optimistis. "
               "Skor final yang tidak bias tetap tabel data uji.")

# ---------------------------------------------------------------------
# TAB 4 — INTERPRETASI FITUR (notebook bagian 14)
# ---------------------------------------------------------------------
with tab4:
    st.subheader("14.1 Fitur apa yang paling berpengaruh?")
    koef, pentingnya = R["koef"], R["pentingnya"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    koef.plot.barh(ax=ax[0], color=np.where(koef > 0, "#55A868", "#C44E52"))
    ax[0].set_title("Linear Regression — Koefisien (terstandarisasi)")
    pentingnya.plot.barh(ax=ax[1], color="#4C72B0")
    ax[1].set_title("Random Forest — Permutation Importance\n(kenaikan MAE jika fitur diacak)")
    plt.tight_layout()
    tampil(fig)
    st.markdown(
        f"- **Linear Regression:** fitur paling berpengaruh adalah `{koef.abs().idxmax()}` "
        f"(koefisien {koef[koef.abs().idxmax()]:+.2f}). Tanda `+` menaikkan penjualan, `−` menurunkan.\n"
        f"- **Random Forest:** fitur paling penting adalah `{pentingnya.idxmax()}` "
        f"(MAE naik {pentingnya.max():.2f} gelas jika diacak).")

# ---------------------------------------------------------------------
# TAB 5 — KESIMPULAN & EKSPOR (notebook bagian 15–17)
# ---------------------------------------------------------------------
with tab5:
    st.subheader("Kesimpulan otomatis dari hasil eksekusi")
    model_ml = tabel.loc[["Linear Regression", rf_label]]
    terbaik = model_ml["MAE"].idxmin()
    lr, rf = model_ml.loc["Linear Regression"], model_ml.loc[rf_label]
    base = tabel.loc["Baseline: Naif (minggu lalu)"]
    base2 = tabel.loc["Baseline: Rata-rata historis"]
    selisih = (lr["MAE"] - rf["MAE"]) / lr["MAE"] * 100

    st.success(f"Model terbaik berdasarkan MAE: **{terbaik}**")
    c = st.columns(3)
    c[0].metric("Linear Regression — MAE", f"{lr['MAE']:.2f}", f"R² {lr['R²']:.3f}", delta_color="off")
    c[1].metric(f"{rf_label} — MAE", f"{rf['MAE']:.2f}", f"R² {rf['R²']:.3f}", delta_color="off")
    c[2].metric("Baseline naif — MAE", f"{base['MAE']:.2f}")

    poin = [
        f"MAPE: Linear Regression **{lr['MAPE (%)']:.2f}%**, Random Forest **{rf['MAPE (%)']:.2f}%**; "
        f"RMSE: **{lr['RMSE']:.2f}** vs **{rf['RMSE']:.2f}**.",
        f"MAE Random Forest **{'lebih rendah' if selisih > 0 else 'lebih tinggi'} {abs(selisih):.1f}%** "
        f"dibanding Linear Regression.",
    ]
    if max(lr["MAE"], rf["MAE"]) < min(base["MAE"], base2["MAE"]):
        poin.append("Kedua model **mengalahkan kedua baseline** — fitur kalender, lag, dan rata-rata bergerak membantu.")
    elif min(lr["MAE"], rf["MAE"]) < min(base["MAE"], base2["MAE"]):
        poin.append("Hanya sebagian model yang mengalahkan baseline; periksa kembali fitur atau parameter.")
    else:
        poin.append("Kedua model **belum mengalahkan baseline** — pola data mungkin terlalu acak.")
    if not np.isnan(R["p_value"]):
        poin.append(f"Uji Wilcoxon p-value = **{R['p_value']:.4f}** → perbedaan "
                    f"**{'signifikan' if R['p_value'] < 0.05 else 'tidak signifikan'}** secara statistik.")
    bias = resid.mean()
    poin.append(f"Bias rata-rata residual: Linear Regression **{bias['Linear Regression']:+.2f}**, "
                f"Random Forest **{bias[rf_label]:+.2f}** gelas. Random Forest tidak bisa *extrapolate* — "
                "jika tren naik, prediksinya cenderung terlalu rendah (residual positif).")
    poin.append(f"R² moderat ({lr['R²']:.2f} / {rf['R²']:.2f}) wajar: penjualan harian UMKM dipengaruhi faktor acak "
                "(cuaca, promo, kondisi sekitar) yang tidak ada di data.")
    for p in poin:
        st.markdown(f"- {p}")

    st.markdown("**Saran pengembangan:** tambahkan fitur hari libur/tanggal gajian, promo, dan cuaca; "
                "latih ulang model berkala; coba Gradient Boosting atau SARIMA/Prophet sebagai pembanding.")

    st.divider()
    st.subheader("16. Ekspor data bersih (SPSS / Excel)")
    long_bersih = df[["tanggal", "nama_produk", "ukuran", "harga_jual", "jumlah_terjual", "omzet"]].copy()
    long_bersih["tanggal"] = long_bersih["tanggal"].dt.strftime("%Y-%m-%d")

    harian_model = data.copy()
    harian_model.insert(0, "hari", harian_model.index.dayofweek.map(dict(enumerate(NAMA_HARI))))
    harian_model["set_data"] = np.where(np.arange(len(harian_model)) < R["split"], "Latih", "Uji")
    harian_model["set_kode"] = (harian_model["set_data"] == "Uji").astype(int)
    harian_model = harian_model.reset_index()
    harian_model["tanggal"] = harian_model["tanggal"].dt.strftime("%Y-%m-%d")

    prediksi_uji = pd.DataFrame({
        "tanggal": y_test.index.strftime("%Y-%m-%d"),
        "aktual": y_test.values,
        "pred_linear_regression": pred_lr,
        "pred_random_forest": pred_rf,
    })
    prediksi_uji["residual_lr"] = prediksi_uji["aktual"] - prediksi_uji["pred_linear_regression"]
    prediksi_uji["residual_rf"] = prediksi_uji["aktual"] - prediksi_uji["pred_random_forest"]
    prediksi_uji = prediksi_uji.round(4)

    xlsx_bytes, zip_bytes = bangun_ekspor(long_bersih, harian_model, prediksi_uji)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh Excel (3 sheet)", xlsx_bytes, "data_bersih_UMKM.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇️ Unduh ZIP (3 file CSV)", zip_bytes, "data_bersih_UMKM.zip", "application/zip")
    with st.expander("Pratinjau data harian + fitur (untuk SPSS)"):
        st.dataframe(harian_model.head(10), hide_index=True)
