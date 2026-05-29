import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# =========================
# 운영체제별 한글 폰트 설정
# =========================
system_name = platform.system()

if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')

elif system_name == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')

elif system_name == 'Linux':
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# =========================
# Streamlit 설정
# =========================
st.set_page_config(
    page_title="환자 군집 예측",
    page_icon="🩺",
    layout="centered"
)

# =========================
# 모델 불러오기
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("lung_model.pkl")

# =========================
# 제목
# =========================
st.title("🩺 환자 군집 예측 시스템")
st.write("환자의 정보를 입력하세요.")

# =========================
# 입력창
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
    # 그래프 출력
    # =========================
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ['나이', '흡연', '알코올']
    values = [age, tbc, alc]

    bars = ax.bar(labels, values, color=['skyblue', 'orange', 'green'])

    ax.set_title("환자 입력 정보", fontsize=16)
    ax.set_ylabel("값", fontsize=12)

    # 막대 위 숫자 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom'
        )

    st.pyplot(fig)
