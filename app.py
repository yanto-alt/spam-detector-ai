import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Spam Detector AI")

text = st.text_area("Masukkan pesan:")

if st.button("Cek"):
    if text.strip() == "":
        st.warning("Pesan tidak boleh kosong")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == "spam":
            st.error("🚨 SPAM")
        else:
            st.success("✅ BUKAN SPAM")
