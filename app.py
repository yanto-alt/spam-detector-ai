import streamlit as st
import pickle
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# ===== Page config =====
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🚨",
    layout="centered"
)

# ===== Title =====
st.title("🚨 Spam Detector AI")
st.write("Cek apakah pesan termasuk **SPAM** atau **BUKAN SPAM**")

# ===== Contoh pesan =====
with st.expander("📌 Contoh Pesan"):
    st.write("🔴 SPAM:")
    st.code("Selamat! Anda memenangkan hadiah 100 juta. Klik link berikut.")
    st.write("🟢 BUKAN SPAM:")
    st.code("Halo, jangan lupa meeting jam 10 pagi ya.")

# ===== Input =====
text = st.text_area("✉️ Masukkan pesan:", height=150)

# ===== Button =====
if st.button("🔍 Cek Pesan"):
    if text.strip() == "":
        st.warning("⚠️ Pesan tidak boleh kosong!")
    else:
        # Transform text
        text_vec = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec).max() * 100

        st.divider()

        # Result
        if prediction == 1:
            st.error(f"🚨 **SPAM** ({probability:.2f}%)")
        else:
            st.success(f"✅ **BUKAN SPAM** ({probability:.2f}%)")

        # Confidence bar
        st.write("📊 Tingkat Keyakinan Model")
        st.progress(int(probability))

