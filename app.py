import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# 모델 & 스케일러 불러오기
# -----------------------------
model = joblib.load("lung_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# 데이터 불러오기
# -----------------------------
df = pd.read_csv("lung_csv")

# -----------------------------
# 기존 데이터 군집 생성
# -----------------------------
X = df[['나이', '흡연', '알코올']]
X_scaled = scaler.transform(X)

df['cluster'] = model.predict(X_scaled)

# -----------------------------
# Streamlit 화면
# -----------------------------
st.title("환자 군집 예측")

st.write("환자의 정보를 입력하세요.")

# 입력창
age = st.number_input("나이", min_value=0.0, step=1.0)
tbc = st.number_input("흡연", min_value=0.0, step=1.0)
alc = st.number_input("알코올", min_value=0.0, step=1.0)

# 버튼
if st.button("군집 예측"):

    # 새 환자 데이터프레임 생성
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

    # -----------------------------
    # 그래프 생성
    # -----------------------------
    fig = px.scatter(
        df,
        x='나이',
        y='흡연',
        color='cluster',
        opacity=0.6,
        title='환자 군집 시각화'
    )

    # 새 환자 표시
    fig.add_scatter(
        x=[age],
        y=[tbc],
        mode='markers',
        marker=dict(
            color='black',
            size=20,
            symbol='x'
        ),
        name='새 환자'
    )

    # 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
