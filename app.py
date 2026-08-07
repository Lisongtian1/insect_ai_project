import streamlit as st
import numpy as np
import pickle
from PIL import Image

# ==========================================
# 1. ตั้งค่าหน้าเพจ & ธีม (Wide Layout)
# ==========================================
st.set_page_config(
    page_title="AI Insect Classifier Dashboard",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ตกแต่งด้วย CSS ให้ดูเป็นแอป AI มืออาชีพ
# ==========================================
st.markdown("""
    <style>
    /* ปรับแต่งกล่องส่วนหัว (Hero Banner) */
    .hero-box {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 25px 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 16px;
        margin-top: 5px;
        opacity: 0.9;
        color: #E0F7FA;
    }
    /* กล่องการ์ดแสดงผลลัพธ์ */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 15px;
    }
    </style>
    
    <div class="hero-box">
        <p class="hero-title">🐞 INSECT AI DASHBOARD</p>
        <p class="hero-subtitle">ระบบปัญญาประดิษฐ์ตรวจวัดและจำแนกประเภทแมลงทางการเกษตร (Random Forest Classifier)</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. แถบเมนูด้านซ้าย (Sidebar - ข้อมูลโครงงาน)
# ==========================================
with st.sidebar:
    st.markdown("### 🔬 ข้อมูลทางเทคนิค")
    st.info("""
    **AI Architecture:**  
    - Machine Learning: `Random Forest`  
    - Feature Extraction: `RGB Color Histogram (64x64)`  
    - Classification: Binary (Pest vs Beneficial)
    """)
    
    st.divider()
    st.markdown("### 💡 คำแนะนำการใช้งาน")
    st.markdown("""
    1. อัปโหลดภาพแมลงที่ต้องการวิเคราะห์
    2. รองรับไฟล์รูปภาพ `.JPG`, `.PNG`, `.WEBP`
    3. ควรเป็นภาพที่เห็นลำตัวและสีสันแมลงชัดเจน
    """)
    
    st.divider()
    st.caption("© 2026 Agricultural AI Research Project")

# ==========================================
# 4. โหลดโมเดล AI
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('insect_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

model = load_model()
classes = ['⚠️ แมลงศัตรูพืช (Pest Insect / Bad)', '✅ แมลงดี มีประโยชน์ (Beneficial Insect / Good)']

# ==========================================
# 5. พื้นที่การทำงานหลัก (แบ่ง 2 คอลัมน์ ซ้าย-ขวา)
# ==========================================
if model is None:
    st.error("🚨 ไม่พบไฟล์โมเดล (`insect_model.pkl`) กรุณาตรวจสอบการตั้งค่าไฟล์ใน GitHub ครับ")
else:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    # ----- ฝั่งซ้าย: อัปโหลดรูปภาพ -----
    with col_left:
        st.subheader("📁 1. อัปโหลดรูปภาพแมลง")
        uploaded_file = st.file_uploader("เลือกภาพจากคอมพิวเตอร์หรือมือถือของคุณ", type=["jpg", "png", "jpeg", "webp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="📷 ภาพถ่ายที่ต้องการวิเคราะห์", use_container_width=True)
        else:
            st.info("👈 กรุณาอัปโหลดรูปภาพเพื่อเริ่มต้นการวิเคราะห์")

    # ----- ฝั่งขวา: ผลการวิเคราะห์จาก AI -----
    with col_right:
        st.subheader("🎯 2. ผลการวิเคราะห์และวัดค่าความถูกต้อง")
        
        if uploaded_file is not None:
            with st.spinner("⏳ AI กำลังสกัดจุดเด่นของภาพและคำนวณความน่าจะเป็น..."):
                # ประมวลผลภาพ
                img = image.resize((64, 64))
                img_array = np.array(img)
                features = (img_array.flatten() / 255.0).reshape(1, -1)
                
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                score = probabilities[prediction] * 100

            # แสดงผลลัพธ์
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if prediction == 1:
                st.success(f"### ผลการตรวจสอบ: **{classes[1]}**")
                st.write("🌿 **บทบาทในระบบนิเวศ:** เป็นแมลงที่มีประโยชน์ต่อพืช เช่น ช่วยผสมเกสร หรือเป็นตัวห้ำ/ตัวเบียนคอยกินแมลงศัตรูพืช ไม่ควรทำลาย")
            else:
                st.error(f"### ผลการตรวจสอบ: **{classes[0]}**")
                st.write("🚨 **ความเสี่ยงต่อการเกษตร:** เป็นแมลงที่อาจกัดกินผลผลิต ดูดน้ำเลี้ยง หรือเป็นพาหะนำโรคพืช ควรเฝ้าระวังและควบคุมปริมาณ")
            
            st.divider()
            
            # เกจวัดความมั่นใจ
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="ระดับความมั่นใจของ AI (Confidence Score)", value=f"{score:.2f} %")
            with col_m2:
                status_text = "มีความแม่นยำสูงมาก" if score >= 80 else "มีความแม่นยำปานกลาง"
                st.metric(label="ความน่าเชื่อถือของผลลัพธ์", value=status_text)
                
            st.caption("เกจความแม่นยำ (Confidence Level):")
            st.progress(int(score))
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⏳ รอการอัปโหลดรูปภาพจากช่องด้านซ้าย...")
