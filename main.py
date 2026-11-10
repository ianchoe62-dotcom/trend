import re
import pandas as pd
from konlpy.tag import Okt

# 1. 데이터 수집 시뮬레이션 및 저장 (실제 구동 시 네이버 API 또는 크롤링 코드로 대체)
def collect_data():
    data = {
        'date': ['2026-03-01', '2026-03-02', '2026-03-03'],
        'source': ['News', 'Twitter', 'Blog'],
        'title': ['제로슈거 음료 인기', '제로 음료 맛 후기', '친환경 트렌드 분석'],
        'content': [
            '제로슈거 음료 맛도 깔끔하고 살 안 찐다니까 매일 마시게 됨. 너무 좋다!',
            '제로 음료에 들어가는 대체당 성분 장기 복용하면 부작용 있다는 연구 결과 나왔는데 무분별하게 마시네...',
            '인스타에서 유행하길래 궁금해서 한 번 사봤음. 인증샷 남기기 딱 좋네.'
        ],
        'label': ['긍정적 수용', '부정적 반발', '단순 소비']  # 모델 학습용 정답 레이블 포함
    }
    df = pd.DataFrame(data)
    df.to_csv('raw_data.csv', index=False, encoding='utf-8-sig')
    print("1단계: raw_data.csv 저장 완료")

# 2. 데이터 전처리 파이프라인
def preprocess_pipeline():
    df = pd.read_csv('raw_data.csv')
    okt = Okt()
    stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '도', '고', '으로', '로']
    
    clean_texts = []
    for text in df['content'].fillna(''):
        # 특수문자, HTML, URL 제거
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
        
        # 형태소 분석 (명사, 형용사 추출) 및 불용어 제거
        morphs = okt.pos(text, stem=True)
        tokens = [word for word, pos in morphs if pos in ['Noun', 'Adjective'] and word not in stopwords]
        clean_texts.append(' '.join(tokens))
        
    df['clean_text'] = clean_texts
    df.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')
    print("2단계: preprocessed_data.csv 전처리 완료")

if __name__ == '__main__':
    collect_data()
    preprocess_pipeline()
    from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

def run_eda():
    df = pd.read_csv('preprocessed_data.csv')
    
    # 상위 키워드 추출
    all_tokens = ' '.join(df['clean_text'].dropna()).split()
    counts = Counter(all_tokens)
    top_20 = counts.most_common(20)
    
    # 윈도우 한글 폰트 설정 (환경에 맞게 수정)
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    
    # 바 차트 시각화
    plt.figure(figsize=(10, 6))
    words, freqs = zip(*top_20)
    sns.barplot(x=list(freqs), y=list(words), palette='viridis')
    plt.title('상위 20개 핵심 키워드 빈도수')
    plt.savefig('top_keywords.png')
    plt.close()
    
    # 워드클라우드 생성
    wc = WordCloud(font_path='malgun.ttf', background_color='white', width=800, height=600)
    cloud = wc.generate_from_frequencies(dict(top_20))
    cloud.to_file('wordcloud.png')
    print("3단계: EDA 시각화 및 워드클라우드 저장 완료")

if __name__ == '__main__':
    run_eda()
    import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv('preprocessed_data.csv')
    
    X = df['clean_text'].fillna('')
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    predictions = model.predict(X_test_vec)
    print("--- 분류 성능 평가 ---")
    print(classification_report(y_test, predictions))
    
    # 모델과 벡터라이저 저장
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("4단계: 모델 학습 및 저장 완료")

def predict_sentiment(text):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    
    return pred, prob

if __name__ == '__main__':
    train_model()
    # 테스트 예측
    print(predict_sentiment("이 제품 정말 편리하고 좋습니다."))
    import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# 모델 및 데이터 로드
@st.cache_resource
def load_resources():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    df = pd.read_csv('preprocessed_data.csv')
    return model, vectorizer, df

model, vectorizer, df = load_resources()

st.title("📊 SNS 및 뉴스 트렌드 대중 반응 분석 대시보드")

# 사이드바: 전체 반응 비율 피자 차트
st.sidebar.header("전체 데이터 반응 비율")
if 'label' in df.columns:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    fig_pie = px.pie(label_counts, names='Label', values='Count', title="반응 유형 분포")
    st.sidebar.plotly_chart(fig_pie)

# 메인 화면: 텍스트 입력 및 예측
st.subheader("새로운 텍스트 반응 분류 예측")
user_input = st.text_area("뉴스 기사 또는 SNS 텍스트를 입력하세요:")

if st.button("분류 예측 실행"):
    if user_input.strip() == "":
        st.warning("텍스트를 입력해주세요.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        
        st.success(f"**예측된 대중 반응 유형:** `{pred}`")
        
        # 확률 분포 바 차트 시각화
        prob_df = pd.DataFrame({'Reaction': classes, 'Probability': probs})
        fig_bar = px.bar(prob_df, x='Reaction', y='Probability', color='Reaction', title="유형별 확률 분포")
        st.plotly_chart(fig_bar)
