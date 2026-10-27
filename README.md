# Dashboard Prediksi Penjualan UMKM (Streamlit)

    pip install -r requirements.txt
    streamlit run app.py

- Data bawaan: `data_penjualan_last_two_tea.xls` (sebenarnya berformat CSV; terdeteksi otomatis).
- Bisa juga unggah CSV / XLS / XLSX lain lewat sidebar.
- Eksekusi pertama dengan GridSearchCV ±1 menit; setelah itu hasil di-cache.
# python -m streamlit run app.py