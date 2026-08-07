import streamlit as st
import numpy as np
import pickle
from PIL import Image

st.set_page_config(
    page_title="Insects Classification AI",
    page_icon="🐞",
    layout="centered"
)

st.title("🐞 โปรแกรมวัดและจำแนกประเภทแมลง")
st.write("อัปโหลดภาพแมลงเพื่อจำแนกประเภท และวัดค่าความถูกต้องด้วย AI (Random Forest)")
st.write("---")

@st.cache_resource
def load_model():
    try:
        with open('insect_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

model = load_model()
classes = ['แมลงศัตรูพืช (Pest Insect / Bad)', 'แมลงดี (Beneficial Insect / Good)']

def extract_features_from_pil(image):
    img = image.resize((64, 64))
    img_array = np.array(img)
    return (img_array.flatten() / 255.0).reshape(1, -1)

if model is None:
    st.warning("⚠️ ยังไม่พบไฟล์โมเดล กรุณารันคำสั่ง `python train.py` ใน Terminal ก่อนครับ")
else:
    uploaded_file = st.file_uploader("📂 เลือกไฟล์รูปภาพแมลงที่ต้องการวัดค่า...", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # แก้เป็น use_container_width=True ตรงนี้ครับ
        st.image(image, caption="ภาพที่ต้องการตรวจสอบ", use_container_width=True)
        
        with st.spinner("⏳ กำลังวิเคราะห์และวัดค่า..."):
            features = extract_features_from_pil(image)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            score = probabilities[prediction] * 100
        
        st.write("---")
        st.subheader("📊 ผลการวิเคราะห์")
        
        if prediction == 1:
            st.success(f"✅ **ผลการวัด:** {classes[1]}")
            st.metric(label="ค่าความมั่นใจ (Confidence Score)", value=f"{score:.2f} %")
        else:
            st.error(f"⚠️ **ผลการวัด:** {classes[0]}")
            st.metric(label="ค่าความมั่นใจ (Confidence Score)", value=f"{score:.2f} %")
