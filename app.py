"""
เว็บแอปทำนายการรอดชีวิตบนเรือไททานิก (Streamlit + scikit-learn)

วิธีรัน (ใน cmd ที่ activate venv แล้ว และอยู่ในโฟลเดอร์เดียวกับไฟล์นี้):
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="ทำนายการรอดชีวิตบนเรือไททานิก", page_icon="🚢")

# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------

# แอปจะหาไฟล์โมเดลตามลำดับนี้ (ไฟล์ต้องอยู่โฟลเดอร์เดียวกับ app.py)
MODEL_FILES = ["titanic_model.joblib", "titanic_tree.joblib"]

# โมเดลนี้ถูกเทรนด้วยค่าที่ "แปลงแล้ว" (ดูจากค่า threshold ในต้นไม้)
# ค่าด้านล่างต้องตรงกับที่ใช้ตอนเทรนใน Colab ถ้าไม่ตรง ผลทำนายจะเพี้ยน
AGE_MIN, AGE_MAX = 0.42, 80.0          # Age ถูกทำ Min-Max scaling ให้อยู่ช่วง 0-1
# Fare: ประมาณการ (Robust scaling) เดาจากโครงสร้างต้นไม้ ไม่ใช่ค่าที่ยืนยันแล้ว
# มีผลแค่ความน่าจะเป็นของผู้หญิงชั้น 1-2 ไม่เปลี่ยนคำตอบ รอด/ไม่รอด
# ถ้าใน Colab ใช้ scaler ตัวอื่น ให้แก้สองบรรทัดนี้
FARE_MEDIAN, FARE_IQR = 14.4542, 23.0896


# ---------------------------------------------------------------------------
# ฟังก์ชันช่วย
# ---------------------------------------------------------------------------

def find_model_path():
    """หาไฟล์โมเดลในโฟลเดอร์เดียวกับ app.py"""
    folder = Path(__file__).parent
    for name in MODEL_FILES:
        path = folder / name
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model(path):
    """โหลดโมเดลครั้งเดียว แล้วจำไว้ (ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม)"""
    return joblib.load(path)


def build_features(model, pclass, sex, age, sibsp, parch, fare):
    """แปลงค่าที่ผู้ใช้กรอก ให้เป็นแถวข้อมูลที่โมเดลต้องการ"""
    row = {
        "Pclass": pclass - 1,                                  # ชั้น 1,2,3 -> 0,1,2
        "Sex_female": 1 if sex == "หญิง" else 0,
        "Age": min(max((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0), 1),
        "Fare": (fare - FARE_MEDIAN) / FARE_IQR,
        "FamilySize": sibsp + parch + 1,                       # นับตัวเองด้วย
    }
    # เรียงคอลัมน์ให้ตรงกับที่โมเดลเห็นตอนเทรน
    return pd.DataFrame([row])[list(model.feature_names_in_)]


# ===== หน้าเว็บ =====

st.title("🚢 ทำนายการรอดชีวิตบนเรือไททานิก")
st.write("กรอกข้อมูลผู้โดยสาร แล้วกด “ทำนาย” เพื่อดูว่าโมเดลคาดว่ารอดหรือไม่")

model_path = find_model_path()
if model_path is None:
    st.error(
        "ไม่พบไฟล์โมเดล กรุณาวางไฟล์ `titanic_model.joblib` "
        "ไว้ในโฟลเดอร์เดียวกับ `app.py` แล้วรีเฟรชหน้านี้"
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(
        f"โหลดโมเดลไม่สำเร็จ: {e}\n\n"
        "ลองติดตั้ง scikit-learn เวอร์ชันเดียวกับที่ใช้เทรน เช่น "
        "`pip install scikit-learn==1.6.1`"
    )
    st.stop()

with st.form("passenger_form"):
    left, right = st.columns(2)

    with left:
        pclass = st.selectbox(
            "ชั้นตั๋ว",
            options=[1, 2, 3],
            format_func=lambda x: f"ชั้น {x}",
            index=2,
        )
        sex = st.radio("เพศ", ["ชาย", "หญิง"], horizontal=True)
        age = st.number_input("อายุ (ปี)", min_value=0.42, max_value=80.0, value=30.0, step=1.0)

    with right:
        sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่มาด้วย", min_value=0, max_value=8, value=0, step=1)
        parch = st.number_input("จำนวนพ่อแม่/ลูกที่มาด้วย", min_value=0, max_value=6, value=0, step=1)
        fare = st.number_input("ค่าตั๋ว (ปอนด์)", min_value=0.0, max_value=520.0, value=30.0, step=1.0)

    submitted = st.form_submit_button("ทำนาย")

if submitted:
    X = build_features(model, pclass, sex, age, sibsp, parch, fare)
    pred = int(model.predict(X)[0])
    survive_idx = list(model.classes_).index(1)
    p_survive = float(model.predict_proba(X)[0][survive_idx])

    st.subheader("ผลการทำนาย")
    if pred == 1:
        st.success("โมเดลคาดว่า **รอดชีวิต**")
    else:
        st.error("โมเดลคาดว่า **ไม่รอดชีวิต**")

    st.metric("ความน่าจะเป็นที่จะรอดชีวิต", f"{p_survive:.1%}")
    st.progress(p_survive)

    with st.expander("ดูค่าที่ส่งเข้าโมเดล (ไว้ตรวจเทียบกับ Colab)"):
        st.dataframe(X)
        st.caption(
            f"scikit-learn ที่ใช้รันแอป: {sklearn.__version__} | "
            f"ไฟล์โมเดล: {model_path.name}"
        )
