# mini_tool_lms.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import shap
import seaborn as sns

# =========================
# 0. KONFIGURASI AWAL
# =========================
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan LMS",
    layout="wide"
)
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("Mini Tool Analisis Kepuasan LMS")
st.write(
    "Aplikasi ini digunakan untuk menganalisis data survei LMS, "
    "melihat distribusi kepuasan pengguna, membangun model klasifikasi "
    "dengan Random Forest, dan menjelaskan model menggunakan SHAP (XAI)."
)

# =========================
# 1. UPLOAD DATA
# =========================
uploaded_file = st.file_uploader(
    "Upload file survei dalam format CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

# Baca data
df = pd.read_csv(uploaded_file)

st.subheader("Preview Data")
st.dataframe(df.head())

st.write(f"Jumlah baris: **{df.shape[0]}**, jumlah kolom: **{df.shape[1]}**")

# =========================
# 2. PILIH KOLOM TARGET
# =========================
st.subheader("Pengaturan Target Kepuasan")

kolom_kandidat_target = df.columns.tolist()
target_col = st.selectbox(
    "Pilih kolom sebagai *Target Kepuasan* (misal: Target_Kepuasan / Overall Satisfaction)",
    options=kolom_kandidat_target
)

if target_col is None:
    st.warning("Silakan pilih kolom target terlebih dahulu.")
    st.stop()

# Cek tipe dan jumlah kelas
target_values = df[target_col].dropna()
unique_classes = target_values.unique()

st.write(f"Nilai unik pada target: `{list(unique_classes)}`")

if len(unique_classes) < 2:
    st.error(
        "Kolom target harus memiliki minimal 2 kelas "
        "(misalnya: Puas dan Tidak Puas)."
    )
    st.stop()

# =========================
# 3. PERSIAPAN DATA
# =========================
# Fitur = semua kolom selain target
X = df.drop(columns=[target_col])
y_raw = df[target_col]

# Buang kolom non-numerik di X (untuk memudahkan; bisa dikembangkan lagi nanti)
X_numeric = X.select_dtypes(include=[np.number])

if X_numeric.shape[1] == 0:
    st.error(
        "Tidak ada fitur numerik yang dapat digunakan. "
        "Pastikan kuesioner Anda berisi skor angka (Likert 1–5)."
    )
    st.stop()

st.write(f"Jumlah fitur numerik yang digunakan untuk pemodelan: **{X_numeric.shape[1]} kolom**")

# Encode target jika masih teks
le = LabelEncoder()
y = le.fit_transform(y_raw)

class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
st.write("Mapping label target (teks → angka):")
st.json(class_mapping)

# =========================
# 4. RINGKASAN DATA & VISUALISASI DASAR
# =========================
st.subheader("Ringkasan Data dan Distribusi")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Distribusi Kelas Target")
    target_counts = pd.Series(y_raw).value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        ax=ax
    )
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Responden")
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + 0.1, str(v), ha="center")
    st.pyplot(fig)

with col2:
    st.markdown("Rata-rata Skor Fitur")
    mean_scores = X_numeric.mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    mean_scores.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Rata-rata Skor")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

st.markdown("---")

# =========================
# 5. PEMBAGIAN DATA TRAIN / TEST
# =========================
test_size = st.slider(
    "Proporsi Data Uji",
    min_value=0.2,
    max_value=0.4,
    step=0.05,
    value=0.3
)

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

st.write(
    f"Data dibagi menjadi **{X_train.shape[0]}** baris training dan "
    f"**{X_test.shape[0]}** baris testing (test size = {test_size})."
)

# =========================
# 6. TRAINING MODEL RANDOM FOREST
# =========================
st.subheader("Pemodelan Klasifikasi: Random Forest")

n_estimators = st.slider(
    "Jumlah Trees (n_estimators)",
    min_value=50,
    max_value=300,
    step=50,
    value=100
)

max_depth = st.selectbox(
    "Maksimum Kedalaman Tree (max_depth)",
    options=[None, 3, 5, 10],
    index=0
)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"**Akurasi pada data uji: {acc:.4f}**")

# Classification report (teks)
report_text = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
)
st.text("Classification Report:")
st.text(report_text)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax_cm
)
ax_cm.set_xlabel("Prediksi")
ax_cm.set_ylabel("Aktual")
ax_cm.set_title("Confusion Matrix - Random Forest")
st.pyplot(fig_cm)

st.markdown("---")

# =========================
# 7. FEATURE IMPORTANCE
# =========================
st.subheader("Pentingnya Fitur (Feature Importance) - Random Forest")

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X_numeric.columns)
feat_imp_sorted = feat_imp.sort_values(ascending=True)

fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
feat_imp_sorted.plot(kind="barh", ax=ax_fi)
ax_fi.set_xlabel("Importance")
ax_fi.set_title("Feature Importance Random Forest")
st.pyplot(fig_fi)

# Tampilkan Top 5 & insight singkat
top_5 = feat_imp.sort_values(ascending=False).head(5)
st.write("**Top 5 fitur paling berpengaruh:**")
st.table(top_5.to_frame("importance"))

# Insight teks otomatis
dominant_class = target_counts.idxmax()
st.markdown(
    f"""
    **Insight Otomatis:**

    - Kelas yang paling banyak muncul adalah **{dominant_class}** \
    dengan jumlah **{target_counts.max()}** responden.
    - Fitur yang paling berpengaruh menurut Random Forest adalah \
    **{top_5.index[0]}**, diikuti oleh **{top_5.index[1]}** dan \
    **{top_5.index[2]}**.
    """
)

st.markdown("---")

# =========================
# 8. EXPLAINABLE AI (SHAP)
# =========================
st.subheader("Explainable AI (SHAP)")

with st.expander("Tampilkan Analisis SHAP (mungkin sedikit lebih lambat)"):
    # Untuk RandomForestClassifier, TreeExplainer akan otomatis dipakai
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Jika 2 kelas → shap_values berbentuk list [class0, class1]
    if isinstance(shap_values, list):
        # Fokus pada kelas "positif" (misalnya kelas dengan indeks 1)
        shap_class_idx = 1 if len(shap_values) > 1 else 0
        shap_for_plot = shap_values[shap_class_idx]
    else:
        shap_for_plot = shap_values

    st.write(
        "Plot berikut menunjukkan pengaruh setiap fitur "
        "terhadap probabilitas prediksi kelas tertentu "
        "(misalnya kelas 'Puas')."
    )

    # Summary plot (bar)
    st.markdown("**SHAP Bar Plot (Rata-rata dampak absolut tiap fitur)**")
    fig_shap_bar = plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_for_plot,
        X_test,
        feature_names=X_numeric.columns,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig_shap_bar)

    # Summary plot (dot)
    st.markdown("**SHAP Summary Plot (Detail sebaran dampak fitur)**")
    fig_shap_dot = plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_for_plot,
        X_test,
        feature_names=X_numeric.columns,
        show=False
    )
    st.pyplot(fig_shap_dot)

st.success("Analisis selesai. Mini tool siap digunakan untuk laporan KP dan eksperimen lanjutan.")
