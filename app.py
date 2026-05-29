import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 한글 폰트 설정
# =========================

# Windows: 맑은 고딕
plt.rc('font', family='Malgun Gothic')

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Streamlit 한글 CSS 적용
# =========================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Malgun Gothic', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Malgun Gothic', sans-serif;
    font-weight: bold;
}

.stButton>button {
    font-family: 'Malgun Gothic', sans-serif;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)
# =========================
# 제목
# =========================
st.title("🩺 환자 군집 예측")

st.write("환자 정보를 입력하세요.")

# =========================
# 입력
# =========================
age = st.number_input("나이 입력", min_value=0.0, step=1.0)
tbc = st.number_input("흡연 입력", min_value=0.0, step=1.0)
alc = st.number_input("알코올 입력", min_value=0.0, step=1.0)

# =========================
# 버튼
# =========================
if st.button("군집 예측"):

    # 새로운 환자 데이터 생성
    new_patient = pd.DataFrame(
        [[age, tbc, alc]],
        columns=['나이', '흡연', '알코올']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # =========================
    # 산점도
    # =========================
    st.subheader("📊 군집 시각화")

    fig, ax = plt.subplots(figsize=(8, 6))

    # 기존 데이터
    ax.scatter(
        df['나이'],
        df['흡연'],
        c=df['cluster'],
        alpha=0.5
    )

    # 새 환자 표시
    ax.scatter(
        age,
        tbc,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel("나이")
    ax.set_ylabel("흡연 여부")

    ax.legend()

    st.pyplot(fig)
