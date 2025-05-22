import streamlit as st

st.set_page_config(page_title="Iris Dashboard App", layout="centered")
st.sidebar.header("Dashboard")

st.title("Selamat datang di Aplikasi Streamlit Sederhana")
st.write("""
Aplikasi ini dirancang untuk menampilkan visualisasi dan prediksi data **Iris**.
Gunakan menu di sidebar untuk menavigasi berbagai fitur yang tersedia, seperti:
- Melihat data Iris secara langsung,
- Visualisasi distribusi data,
- Memprediksi jenis bunga berdasarkan input fitur.

Silakan jelajahi dan nikmati tampilan interaktif dari aplikasi ini! 🚀
""")
