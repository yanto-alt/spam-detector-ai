import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Spam Detector AI")

text = st.text_area("Masukkan pesan:")

if st.button("Cek"):
    data = vectorizer.transform([user_input])
    pred = model.predict(data)[0]

    if pred == "spam":
        st.error("🚨 Ini adalah SPAM!")
    else:
        st.success("✅ Ini BUKAN spam (ham)")
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


