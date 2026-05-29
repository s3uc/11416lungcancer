import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 한글 폰트 설정
# =========================
plt.rc('font', family='Malgun Gothic')   # 윈도우
plt.rcParams['axes.unicode_minus'] = False

# Mac 사용 시
# plt.rc('font', family='AppleGothic')

# Linux 사용 시 (나눔폰트 설치 필요)
# plt.rc('font', family='NanumGothic')

# =========================
# 모델 불러오기
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# =========================
# Streamlit 화면
# =========================
st.set_page_config(
    page_title="환자 군집 예측",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 환자 군집 예측 시스템")
st.write("환자의 정보를 입력하세요.")

# =========================
# 입력 영역
# =========================
age = st.number_input("나이", min_value=0.0, step=1.0)
tbc = st.number_input("흡연", min_value=0.0, step=1.0)
alc = st.number_input("알코올", min_value=0.0, step=1.0)

# =========================
# 예측 버튼
# =========================
if st.button("군집 예측"):

    # 데이터프레임 생성
    new_patient = pd.DataFrame(
        [[age, tbc, alc]],
        columns=['나이', '흡연', '알코올']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # =========================
    # 간단한 시각화
    # =========================
    fig, ax = plt.subplots(figsize=(5, 3))

    labels = ['나이', '흡연', '알코올']
    values = [age, tbc, alc]

    ax.bar(labels, values)
    ax.set_title("입력 환자 정보")

    st.pyplot(fig)
