import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="와인 분석기",
    page_icon="🍷",
    layout="centered"
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    text-align: center;
    margin-bottom: 15px;
}

div.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    background-color: #f9f9f9;
    color: #333;
    font-size: 14px;
    padding: 6px 10px;
    transition: background-color 0.2s;
}
div.stButton > button:hover {
    background-color: #f0e6f0;
    border-color: #c084c8;
}
</style>
""", unsafe_allow_html=True)

# 1. 모델과 데이터 로드 (경로 수정 버전)
try:
    red_model = joblib.load('model/red_model.pkl')
    white_model = joblib.load('model/white_model.pkl')

    red_df = pd.read_csv('dataset/winequality-red.csv')

    try:
        # 화이트 와인 데이터 경로 및 구분자 처리
        white_df = pd.read_csv('dataset/winequality-white.csv', sep=';')
        if 'quality' not in white_df.columns:
            white_df = pd.read_csv('dataset/winequality-white.csv')
    except:
        white_df = pd.read_csv('dataset/winequality-white.csv')

except FileNotFoundError:
    st.error("""
        🚨 **파일을 찾을 수 없습니다!**
        1. GitHub에 'model' 폴더와 'dataset' 폴더가 잘 올라갔는지 확인하세요.
        2. 파일명이 정확한지(대소문자 포함) 확인하세요.
        - 필요한 파일: model/red_model.pkl, dataset/winequality-red.csv 등
    """)
    st.stop()
except Exception as e:
    st.error(f"파일 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------
# 🎯 페이지 선택
# -----------------------------
wine_type = st.sidebar.selectbox("🍷 와인 종류 선택", ["레드 와인", "화이트 와인"])
page = st.sidebar.selectbox("📂 기능 선택", ["품질 예측", "취향 매치"])
st.markdown("""
<h1 style='text-align: center;'>
🍷 <span style='color:red;'>레드 와인</span> & 
<span style='color:#bbb;'>화이트 와인</span> 분석기
</h1>
""", unsafe_allow_html=True)

st.caption("레드 와인과 화이트 와인의 특성을 반영한 머신러닝 기반 분석 서비스")

if wine_type == "레드 와인":
    model = red_model
    df = red_df
    wine_label = "레드 와인"
else:
    model = white_model
    df = white_df
    wine_label = "화이트 와인"

# -----------------------------
# ✅ 1. 품질 예측 (ML 기반)
# -----------------------------
if page == "품질 예측":

    st.write(f"머신러닝 기반으로 {wine_label}의 고급 패턴인지 분석합니다.")

    st.sidebar.header("🔍 상세 수치 입력")

    def user_input_features():
        st.sidebar.markdown("### 🔍 상세 성분 수치 입력")
        st.sidebar.info("💡 와인 라벨에 수치가 적혀 있지 않다면, 기본값 그대로 분석해 보세요!")
        
        if wine_type == "레드 와인":
            alcohol = st.sidebar.number_input('알코올 도수 (%)', 8.0, 18.0, 13.5, 0.1, 
                                            help="와인 라벨에 기재된 도수를 입력하세요.")
            sugar_mg = st.sidebar.number_input('잔당 (mg/L)', 500, 30000, 2500, 100,
                                            help="발효 후 남은 설탕 양입니다. 보통 라벨에는 '드라이/스위트'로 표시되며 수치는 생략되는 경우가 많습니다.")
            acidity = st.sidebar.number_input('휘발성 산도 (g/L)', 0.1, 2.0, 0.6, 0.01,
                                            help="와인의 '식초 같은 신맛' 정도를 나타냅니다. 너무 높으면 불쾌한 냄새가 날 수 있습니다.")
            sulphates = st.sidebar.number_input('황산염 (g/L)', 0.2, 2.0, 0.7, 0.01,
                                                help="와인의 산화를 방지하고 신선도를 유지하는 보존제 역할 성분입니다.")
            
            data = {
                'fixed acidity': 8.3, 'volatile acidity': acidity, 'citric acid': 0.27,
                'residual sugar': sugar_mg / 1000, 'chlorides': 0.08, 'free sulfur dioxide': 15.0,
                'total sulfur dioxide': 46.0, 'density': 0.99, 'pH': 3.3,
                'sulphates': sulphates, 'alcohol': alcohol
            }
        else:
            alcohol = st.sidebar.number_input('알코올 도수 (%)', 7.0, 16.0, 12.0, 0.1,
                                            help="화이트 와인의 알코올 도수를 입력하세요. 보통 10~13% 사이가 많습니다.")
            sugar_mg = st.sidebar.number_input('잔당 (mg/L)', 500, 100000, 5000, 100,
                                            help="화이트 와인은 레드보다 당도 범위가 넓습니다. 아주 단 디저트 와인은 수치가 매우 높을 수 있습니다.")
            acidity = st.sidebar.number_input('휘발성 산도 (g/L)', 0.1, 1.5, 0.3, 0.01,
                                            help="화이트 와인의 상큼함을 결정하는 산도 패턴입니다.")
            sulphates = st.sidebar.number_input('황산염 (g/L)', 0.1, 1.2, 0.5, 0.01,
                                                help="화이트 와인의 변색을 막고 신선한 과일 향을 유지해주는 성분입니다.")
            
            data = {
                'fixed acidity': 6.8, 'volatile acidity': acidity, 'citric acid': 0.33,
                'residual sugar': sugar_mg / 1000, 'chlorides': 0.05, 'free sulfur dioxide': 35.0,
                'total sulfur dioxide': 138.0, 'density': 0.994, 'pH': 3.2,
                'sulphates': sulphates, 'alcohol': alcohol
            }

        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    st.caption("※ 알코올, 잔당, 산도, 황산염 외 나머지 성분은 기본값으로 계산됩니다.")
    

    if st.button('🔍 품질 예측 시작'):
        with st.spinner("🔍 와인의 성분을 분석하고 있습니다..."):
            prediction = model.predict(input_df)
            user_alc = input_df['alcohol'].iloc[0]
            avg_alc = df['alcohol'].mean()

            user_sugar = input_df['residual sugar'].iloc[0]
            avg_sugar = df['residual sugar'].mean()

            user_sulph = input_df['sulphates'].iloc[0]
            avg_sulph = df['sulphates'].mean()

            user_acidity = input_df['volatile acidity'].iloc[0]
            avg_acidity = df['volatile acidity'].mean()
            prob = model.predict_proba(input_df)

            st.subheader(f"{wine_label} 품질 예측 결과")
            st.caption("※ 본 결과는 머신러닝 기반 참고용 예측이며, 실제 와인 품질 평가와 차이가 있을 수 있습니다.")

            high_quality_index = list(model.classes_).index(1)
            high_quality_prob = float(prob[0][high_quality_index])
            normal_wine_prob = float(prob[0][list(model.classes_).index(0)])

            st.markdown(f"""
            <div class="result-box">
                <h2>{f'고급 {wine_label}' if prediction[0]==1 else f'🙂 일반 {wine_label}'}</h2>
                <h3>{(high_quality_prob if prediction[0]==1 else normal_wine_prob)*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            if prediction[0] == 1:
                st.success(f"고급 와인 패턴 ({high_quality_prob*100:.1f}%)")
                st.caption("※ 본 결과는 머신러닝 기반 참고용 예측이며, 실제 와인 품질 평가와 차이가 있을 수 있습니다.")
                if wine_type == "레드 와인":
                    st.info("📌 알코올이 높고 산도가 낮아 묵직한 고급 레드 와인 패턴입니다.")
                else:
                    st.info("📌 산미와 당도의 균형이 좋아 상큼한 고급 화이트 와인 패턴입니다.")
                
                if user_alc > avg_alc:
                    st.write("✔ 알코올 수치가 평균보다 높습니다")
                if user_acidity < avg_acidity:
                    st.write("✔ 산도가 낮아 부드러운 맛이 예상됩니다")
                if user_sulph > avg_sulph:
                    st.write("✔ 황산염 수치가 높아 풍미가 강화될 수 있습니다")

                st.write("👉 저렴한 와인에서 이 성분이면 가성비 최고!")

            else:
                st.warning(f"일반 와인 패턴 ({normal_wine_prob*100:.1f}%)")
                
                st.info("📌 평균적인 성분으로 구성된 일반 와인입니다.")
                
                if user_alc < avg_alc:
                    st.write("✔ 알코올 수치가 평균보다 낮습니다")
                if user_acidity > avg_acidity:
                    st.write("✔ 산도가 높아 신맛이 강할 수 있습니다")
                if user_sugar > avg_sugar:
                    st.write("✔ 잔당이 높아 단맛이 느껴질 수 있습니다")

                st.write("👉 무난하게 즐기기 좋은 와인")
            st.progress(int(high_quality_prob * 100))
            st.caption(f"📊 고급 와인 확률: {high_quality_prob*100:.1f}%")
            
            if 0.45 < high_quality_prob < 0.55:
                st.info("중간 영역입니다. (고급과 일반의 경계)")

            st.divider()
            st.subheader("🍷 와인 스타일 분석")

            if user_sugar < 2:
                st.write("✔ 드라이 와인")
            else:
                st.write("✔ 스위트 와인")

            if user_alc > avg_alc:
                st.write("✔ 바디감이 강한 와인")
            else:
                st.write("✔ 가벼운 바디감")
            
            st.divider()
            st.subheader("🧪 평균 대비 성분 비교")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("알코올", f"{user_alc:.2f}%", f"{user_alc - avg_alc:.2f}%")

            user_sugar = input_df['residual sugar'].iloc[0]
            avg_sugar = df['residual sugar'].mean()
            col2.metric("잔당", f"{user_sugar:.2f}", f"{user_sugar - avg_sugar:.2f}", delta_color="inverse")

            user_sulph = input_df['sulphates'].iloc[0]
            avg_sulph = df['sulphates'].mean()
            col3.metric("황산염", f"{user_sulph:.2f}", f"{user_sulph - avg_sulph:.2f}")
            
            user_acidity = input_df['volatile acidity'].iloc[0]
            avg_acidity = df['volatile acidity'].mean()
            col4.metric("산도", f"{user_acidity:.2f}", f"{user_acidity - avg_acidity:.2f}")
            st.divider()
            st.subheader("🍽️ 추천 페어링")

            if prediction[0] == 1:
                if user_acidity < avg_acidity and user_alc > avg_alc:
                    st.write("🥩 추천 음식: 스테이크, 바비큐, 양갈비")
                    st.write("🧀 추천 치즈: 체다, 고다, 파르미지아노")
                else:
                    st.write("🍝 추천 음식: 크림 파스타, 리조또, 버섯 요리")
                    st.write("🧀 추천 치즈: 브리, 까망베르")
            else:
                if user_sugar > avg_sugar:
                    st.write("🍰 추천 음식: 과일, 디저트, 케이크")
                    st.write("🧀 추천 치즈: 크림치즈, 마스카포네")
                else:
                    st.write("🍕 추천 음식: 피자, 파스타, 가벼운 육류 요리")
                    st.write("🧀 추천 치즈: 모짜렐라, 몬테레이잭")
            
            st.divider()      
            st.subheader("📈 더 좋은 와인을 위한 팁")

            if prediction[0] == 0:
                if user_alc < avg_alc:
                    st.write("👉 알코올 도수가 조금 더 높아지면 고급 와인 패턴에 가까워질 수 있습니다.")
                if user_acidity > avg_acidity:
                    st.write("👉 산도가 조금 낮아지면 더 부드러운 인상을 줄 수 있습니다.")
                if user_sulph < avg_sulph:
                    st.write("👉 황산염 수치가 약간 높아지면 풍미가 더 살아날 수 있습니다.")
                if user_sugar > avg_sugar:
                    st.write("👉 잔당이 너무 높으면 고급 와인 패턴과 멀어질 수 있습니다.")
            else:
                st.write("👉 현재 입력값은 비교적 고급 와인 패턴에 가깝습니다.")
                st.write("👉 비슷한 성향의 와인을 찾으면 만족도가 높을 가능성이 큽니다.")
                
            st.write("---")
            st.subheader("📝 총평")

            if prediction[0] == 1:
                st.write("👉 고급 와인 스타일로 풍미와 밸런스가 뛰어난 와인입니다.")
            else:
                st.write("👉 부담 없이 즐길 수 있는 일상형 와인입니다.")
                
            st.write("---")
            st.subheader("📈 데이터 인사이트")

            fig, ax = plt.subplots(figsize=(6,4))
            ax.set_title(f"{wine_label} 알코올 도수 대비 품질 분포", fontsize=12)
            ax.set_xlabel("알코올 도수")
            ax.set_ylabel("품질 점수")
            
            sns.scatterplot(
                data=df,
                x='alcohol',
                y='quality',
                hue='quality',
                palette='coolwarm',
                alpha=0.6,
                ax=ax
            )

            estimated_quality = df['quality'].mean() + (high_quality_prob - 0.5) * 2

            ax.scatter(user_alc, estimated_quality,
                    s=250, color='#FFD700',
                    edgecolor='black', linewidth=2, zorder=5)

            ax.text(user_alc, estimated_quality,
                    f"{estimated_quality:.1f}",
                    color='black', fontsize=9,
                    ha='center', va='center', weight='bold', zorder=6)

            from matplotlib.lines import Line2D

            handles, labels = ax.get_legend_handles_labels()

            custom_marker = Line2D([0], [0], marker='o', color='w',
                                markerfacecolor='gold', markeredgecolor='black',
                                markersize=10, label='내 와인')

            handles.append(custom_marker)
            labels.append('내 와인')

            ax.legend(handles=handles, labels=labels, loc='lower right')
            st.caption("※ 내 와인 위치는 입력값과 예측 확률을 기반으로 추정된 참고용 위치입니다.")
            st.pyplot(fig)

# -----------------------------
# ✅ 2. 취향 매치
# -----------------------------
elif page == "취향 매치":

    st.write(f"{wine_label}의 특성을 반영하여 나의 취향에 맞는 와인을 추천합니다.")
    
    # 용어 가이드 박스 추가
    with st.expander("📝 와인 용어 설명 (클릭해서 확인)"):
        st.markdown("""
        * **🍯 당도 (Sweet)**: 혀 끝에서 느껴지는 달콤함의 정도입니다.
        * **🍷 바디감 (Body)**: 입안에서 느껴지는 와인의 '무게감'이나 '질감'입니다. (물 vs 우유 vs 두유의 차이와 비슷합니다.)
        * **🍋 산미 (Acidity)**: 입안에 침이 고이게 하는 상큼하거나 신맛의 정도입니다. 와인의 신선함을 결정합니다.
        """)

    st.subheader("🎯 나의 취향 입력")
    st.caption("버튼으로 빠르게 선택하거나, 슬라이더로 세밀하게 조정하세요.")

    # 세션 상태 초기화
    if 'sweet_val' not in st.session_state:
        st.session_state.sweet_val = 5
    if 'body_val' not in st.session_state:
        st.session_state.body_val = 5
    if 'acidity_val' not in st.session_state:
        st.session_state.acidity_val = 5
    if 'sweet_key' not in st.session_state:
        st.session_state.sweet_key = 0
    if 'body_key' not in st.session_state:
        st.session_state.body_key = 0
    if 'acidity_key' not in st.session_state:
        st.session_state.acidity_key = 0

    # --- 당도 ---
    st.markdown(" 당도 (Sweet)*")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" 드라이", key="dry"):
            st.session_state.sweet_val = 2
            st.session_state.sweet_key += 1
    with col2:
        if st.button(" 미디엄", key="medium_sweet"):
            st.session_state.sweet_val = 5
            st.session_state.sweet_key += 1
    with col3:
        if st.button(" 스위트", key="sweet"):
            st.session_state.sweet_val = 8
            st.session_state.sweet_key += 1

    sweet = st.slider("", 1, 10, st.session_state.sweet_val,
                      key=f"sweet_slider_{st.session_state.sweet_key}",
                      label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 바디감 ---
    st.markdown(" 바디감 (Body)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" 라이트", key="light"):
            st.session_state.body_val = 2
            st.session_state.body_key += 1
    with col2:
        if st.button(" 미디엄", key="medium_body"):
            st.session_state.body_val = 5
            st.session_state.body_key += 1
    with col3:
        if st.button(" 풀바디", key="full"):
            st.session_state.body_val = 8
            st.session_state.body_key += 1

    body = st.slider("", 1, 10, st.session_state.body_val,
                     key=f"body_slider_{st.session_state.body_key}",
                     label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 산미 ---
    st.markdown(" 산미 (Acidity)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😌 낮음", key="low"):
            st.session_state.acidity_val = 2
            st.session_state.acidity_key += 1
    with col2:
        if st.button("😊 보통", key="medium_acid"):
            st.session_state.acidity_val = 5
            st.session_state.acidity_key += 1
    with col3:
        if st.button("😆 높음", key="high"):
            st.session_state.acidity_val = 8
            st.session_state.acidity_key += 1

    acidity = st.slider("", 1, 10, st.session_state.acidity_val,
                        key=f"acidity_slider_{st.session_state.acidity_key}",
                        label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 취향 분석 시작"):
        with st.spinner("🍷 당신의 취향을 분석하고 있습니다..."):
            diff = max(sweet, body, acidity) - min(sweet, body, acidity)

            if wine_type == "레드 와인":
                if diff <= 1:
                    result = " 균형 잡힌 레드 와인"
                    recommend = "메를로 / 하우스 레드"
                else:
                    if body == max(sweet, body, acidity):
                        result = " 묵직한 레드 와인"
                        recommend = "까베르네 소비뇽 / 쉬라"
                    elif acidity == max(sweet, body, acidity):
                        result = " 산미 있는 레드 와인"
                        recommend = "피노 누아"
                    else:
                        result = " 부드러운 과실향 레드 와인"
                        recommend = "메를로"
            else:
                if diff <= 1:
                    result = " 균형 잡힌 화이트 와인"
                    recommend = "피노 그리지오 / 하우스 화이트"
                else:
                    if sweet == max(sweet, body, acidity):
                        result = " 달콤한 화이트 와인"
                        recommend = "모스카토 / 리슬링"
                    elif acidity == max(sweet, body, acidity):
                        result = " 산미 있는 화이트 와인"
                        recommend = "소비뇽 블랑"
                    else:
                        result = " 바디감 있는 화이트 와인"
                        recommend = "샤르도네"

            st.subheader("🎉 분석 결과")
            st.success(f"당신의 취향: **{result}**")
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"💡 추천 와인: **{recommend}**")
            st.markdown("<br><br>", unsafe_allow_html=True)

            st.subheader("🍽️ 함께 즐기면 좋은 음식 & 치즈")

            if wine_type == "레드 와인":
                if "묵직한" in result:
                    st.write("🥩 추천 음식: 스테이크, 바비큐")
                    st.write("🧀 추천 치즈: 체다, 고다")
                elif "산미" in result:
                    st.write("🍖 추천 음식: 토마토 파스타, 피자")
                    st.write("🧀 추천 치즈: 파르미지아노")
                else:
                    st.write("🍝 추천 음식: 파스타, 버섯 요리")
                    st.write("🧀 추천 치즈: 모짜렐라")
            else:
                if "달콤한" in result:
                    st.write("🍰 추천 음식: 디저트, 과일")
                    st.write("🧀 추천 치즈: 브리, 크림치즈")
                elif "산미" in result:
                    st.write("🐟 추천 음식: 해산물, 샐러드")
                    st.write("🧀 추천 치즈: 염소 치즈")
                else:
                    st.write("🥗 추천 음식: 가벼운 음식, 샐러드")
                    st.write("🧀 추천 치즈: 모짜렐라")
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📝 한 줄 총평")

            if "달콤" in result:
                st.write("👉 달콤하고 부드러운 맛을 선호하는 취향입니다.")
            elif "부드러운" in result:
                st.write("👉 과일향이 풍부하고 부드러운 와인을 선호하는 취향입니다.")
            elif "산미" in result:
                st.write("👉 상큼하고 산뜻한 느낌의 와인을 선호하는 취향입니다.")
            elif "묵직한" in result:
                st.write("👉 깊고 진한 풍미의 와인을 선호하는 취향입니다.")
            else:
                st.write("👉 균형 잡힌 와인을 선호하는 안정적인 취향입니다.")