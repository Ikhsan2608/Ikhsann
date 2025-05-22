import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ======================
# KONFIGURASI AWAL
# ======================

# Buat folder 'model' jika belum ada
if not os.path.exists("model"):
    os.makedirs("model")

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================

def load_data():
    """Memuat dataset dari file lokal atau URL"""
    try:
        df = pd.read_csv("model/iris.csv")
        st.success("Dataset berhasil dimuat dari file lokal")
    except:
        df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398eaa3cba4f7d537619d0e07d5ae3/iris.csv")
        st.warning("Menggunakan dataset dari URL karena file lokal tidak ditemukan")
    return df

# ======================
# TRAIN/LOAD MODEL
# ======================

def get_model():
    """Memuat model yang ada atau melatih model baru"""
    try:
        model = joblib.load("model/iris_model.pkl")
        st.success("Model berhasil dimuat dari file")
        return model
    except Exception as e:
        st.warning(f"Model tidak ditemukan: {str(e)}. Melatih model baru...")
        
        # Train new model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Save the new model
        joblib.dump(model, "model/iris_model.pkl")
        st.success("Model baru berhasil dilatih dan disimpan!")
        return model

# ======================
# TAMPILAN UI
# ======================

st.title("🌸 Aplikasi Klasifikasi Iris")
st.markdown("""
Aplikasi ini menunjukkan klasifikasi dataset Iris menggunakan Random Forest.
""")

# Load data
df = load_data()

# Tampilkan data
with st.expander("📊 Lihat Dataset"):
    st.dataframe(df, height=200)

# ======================
# PREPROCESSING
# ======================

X = df.drop("variety", axis=1)
y = df["variety"]
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# ======================
# MODEL OPERATION
# ======================

model = get_model()

# ======================
# EVALUASI
# ======================

st.subheader("📈 Evaluasi Model")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi Model", f"{accuracy:.2%}")

with col2:
    st.write("**Classification Report:**")
    st.code(classification_report(y_test, y_pred))

# ======================
# PREDIKSI MANUAL
# ======================

st.subheader("🔮 Prediksi Manual")

with st.form("input_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)
    
    if st.form_submit_button("Prediksi"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()
        
        st.success(f"Hasil Prediksi: **{prediction}** (Confidence: {probability:.2%})")
