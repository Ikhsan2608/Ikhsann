import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediction", page_icon="🔍")
st.header("🔍 Prediksi Jenis Bunga Iris")

# --- Load Dataset ---
try:
    df = pd.read_csv("model/iris.csv")  # file lokal
except:
    df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398eaa3cba4f7d537619d0e07d5ae3/iris.csv")

st.subheader("📊 Iris Dataset")
st.dataframe(df)

# --- Preprocessing ---
X = df.drop("variety", axis=1)
y = df["variety"]

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Training Model ---
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Evaluasi Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Evaluasi Model")
st.write(f"**Akurasi Model:** {accuracy:.2f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred))

# --- Prediksi Berdasarkan Input Pengguna ---
st.subheader("🔍 Prediksi Bunga dari Input Manual")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal.length", "sepal.width", "petal.length", "petal.width"])

if st.button("Prediksi Sekarang"):
    prediction = model.predict(input_data)
    st.success(f"🌼 Model memprediksi bunga tersebut adalah: **{prediction[0]}**")
