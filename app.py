import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam Detector AI")

# INPUT USER (INI YANG KEMARIN KURANG)
user_input = st.text_area("Masukkan pesan:")

if st.button("Cek"):
    if user_input.strip() == "":
        st.warning("Pesan tidak boleh kosong")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ BUKAN SPAM")
