import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

st.title("Spam Detector AI")

text = st.text_area("Masukkan pesan:")

if st.button("Cek"):
    if text.strip() == "":
        st.warning("Pesan tidak boleh kosong")
    else:
        data = vectorizer.transform([text])
        pred = model.predict(data)[0]

        # Numeric label handling
        if pred == 1:
            st.error("🚨 Ini adalah SPAM!")
        else:
            st.success("✅ Ini BUKAN spam (ham)")
