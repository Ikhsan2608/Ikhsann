import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediction", page_icon="🔍")
st.header("📈 Evaluasi Performa Model")

# --- Load Dataset ---
try:
    df = pd.read_csv("model/iris.csv")
except:
    df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398eaa3cba4f7d537619d0e07d5ae3/iris.csv")

# --- Tampilkan Dataset ---
st.subheader("📊 Iris Dataset")
st.dataframe(df)

# --- Preprocessing ---
X = df.drop("variety", axis=1)
y = df["variety"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# --- Load Model atau Latih Model Baru ---
try:
    model = joblib.load("model/iris_model.pkl")
    st.success("✅ Model berhasil dimuat dari file iris_model.pkl")
except Exception as e:
    st.warning(f"⚠️ Model tidak ditemukan: {str(e)}. Melatih model baru...")
    
    # Latih dan simpan model baru
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Simpan model yang baru dilatih
    joblib.dump(model, "model/iris_model.pkl")
    st.success("🚀 Model baru berhasil disimpan!")

# --- Evaluasi Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.subheader("🔍 Hasil Evaluasi")
st.write(f"**Akurasi Model:** {accuracy:.2f}")
st.text("Classification Report:")
st.code(report)
