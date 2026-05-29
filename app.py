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
# 한글 폰트 설정 (간단 버전)
# =========================
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')  # Windows 기준
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
# 입력
# =========================
age = st.number_input("나이", min_value=0.0, step=1.0)
tbc = st.number_input("흡연", min_value=0.0, step=1.0)
alc = st.number_input("알코올", min_value=0.0, step=1.0)

# =========================
# 예측 버튼
# =========================
if st.button("군집 예측"):

    # 데이터 생성
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
    # 막대그래프
    # =========================
    st.subheader("📊 입력값 막대그래프")

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    labels = ['나이', '흡연', '알코올']
    values = [age, tbc, alc]

    bars = ax1.bar(
        labels,
        values,
        color=['skyblue', 'orange', 'green']
    )

    ax1.set_title(
        "환자 입력 정보",
        fontproperties=fontprop,
        fontsize=18
    )

    ax1.set_ylabel(
        "값",
        fontproperties=fontprop
    )

    # x축 폰트 적용
    for label in ax1.get_xticklabels():
        label.set_fontproperties(fontprop)

    # y축 폰트 적용
    for label in ax1.get_yticklabels():
        label.set_fontproperties(fontprop)

    # 값 표시
    for bar in bars:
        height = bar.get_height()

        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=11
        )

    st.pyplot(fig1)

    # =========================
    # 산점도
    # =========================
    st.subheader("📍 입력값 산점도")

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    x = [1, 2, 3]
    y = [age, tbc, alc]

    labels = ['나이', '흡연', '알코올']
    colors = ['blue', 'red', 'green']

    ax2.scatter(
        x,
        y,
        c=colors,
        s=200
    )

    # 점 이름 표시
    for i in range(len(labels)):
        ax2.text(
            x[i],
            y[i] + 0.5,
            labels[i],
            fontproperties=fontprop,
            ha='center',
            fontsize=12
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontproperties=fontprop)

    ax2.set_ylabel(
        "값",
        fontproperties=fontprop
    )

    ax2.set_title(
        "환자 데이터 산점도",
        fontproperties=fontprop,
        fontsize=18
    )

    st.pyplot(fig2)
