# -*- coding: utf-8 -*-
"""
🧠 EfficientNet-B0 표정 감정 인식 실시간 웹 애플리케이션 (app.py)

이 스크립트는 사전 학습 후 최종 저장된 딥러닝 최적 가중치 파일('best_emotion_model.pth' 또는 'model.pth')을 불러와서,
사용자가 브라우저를 통해 직접 이미지를 업로드(혹은 웹캠 사진 구동)하면 실시간으로 얼굴 감정을 초고속 추론하고,
세련된 확률 수평 차트와 감정 매칭 스토리텔링 카드를 시각화하여 띄워주는 고품질 Streamlit 웹 데모 애플리케이션입니다.

[가상환경 구동 방법]:
1. 터미널을 열고 가상환경을 활성화한 후 필수 패키지를 설치합니다.
   pip install streamlit pillow torch torchvision
2. 아래 명령어로 스트림릿 로컬 서버를 작동시킵니다.
   streamlit run app.py
3. 자동으로 열리는 웹 브라우저(http://localhost:8501)에서 아름다운 인공지능 앱을 조작합니다.
"""

import os
import time
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

import streamlit as st

# ------------------------------------------------------------------------------
# 🎨 1. Streamlit 모던 페이지 설정 및 CSS 디자인 스타일링
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="인공지능 표정 감정 인식기 (AI Emotion Recognizer)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 폰트 및 모던 웹 스타일 시트(CSS) 삽입
st.markdown("""
    <style>
        /* 메인 배경 및 텍스트 폰트 설정 */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', 'Noto Sans KR', sans-serif;
        }
        
        /* Glassmorphism 효과가 가미된 헤더 디자인 */
        .main-header {
            background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
            padding: 30px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-weight: 700;
            font-size: 2.6rem;
            margin-bottom: 10px;
        }
        
        .main-header p {
            font-weight: 300;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* 감정 설명 스토리텔링 카드 스타일 */
        .emotion-card {
            background-color: #f8f9fa;
            border-left: 8px solid #2196F3;
            padding: 20px;
            border-radius: 12px;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        
        .emotion-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .emotion-desc {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #495057;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 🧬 2. 딥러닝 EfficientNet-B0 구조 수립 및 가중치 캐싱 엔진 구현
# ------------------------------------------------------------------------------
def get_emotion_efficientnet(num_classes=7):
    """
    사전 학습된 세계 최고의 이미지 분류기 EfficientNet-B0의 맨 하단 리니어 계층을 7개 감정 규격으로 개조합니다.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# 💡 [중요 설계] 캐싱 어노테이션을 지정해 모델 파일 로드가 최초 딱 1번만 동작하게 보장합니다.
@st.cache_resource
def load_emotion_ai_model():
    """
    디렉토리 내의 가중치 파일을 탐색해 읽어온 후 GPU/CPU 디바이스 메모리에 모델을 적재하는 캐시 함수입니다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 가중치 파일 후보 수색
    model_candidates = ['best_emotion_model.pth', 'model.pth']
    selected_model_path = None
    
    for candidate in model_candidates:
        if os.path.exists(candidate):
            selected_model_path = candidate
            break
            
    if selected_model_path is None:
        return None, device
        
    # 빈 모델 껍데기 구축 (원래 학습 데이터 규격인 7로 생성)
    model = get_emotion_efficientnet(num_classes=7)
    
    # 가중치 주입
    model.load_state_dict(torch.load(selected_model_path, map_location=device))
    model.to(device)
    model.eval() # 추론 모드 고정
    
    return model, device


# ------------------------------------------------------------------------------
# 📊 3. 감정 상세 레이블 정보 및 한글 해설 매핑 셋업
# ------------------------------------------------------------------------------
EMOTION_INFO = {
    0: {
        'name': '화남 (Angry)',
        'emoji': '😡',
        'color': '#FF4B4B',
        'desc': '눈썹이 찌푸려지고 얼굴이 붉어지는 강한 분노 상태입니다. 잠시 눈을 감고 깊게 숨을 들이쉬며 따뜻한 온수 한 잔으로 격앙된 마음을 차분히 달래 보시는 건 어떨까요? 평온함이 진정한 내면의 힘입니다.'
    },
    1: {
        'name': '혐오 (Disgust)',
        'emoji': '🤢',
        'color': '#4CAF50',
        'desc': '불쾌하거나 상황에 대한 거부 반응 및 꺼림칙한 마음이 서려 있는 상태입니다. 억지로 마주하지 말고, 환기를 시키거나 기분을 편안하게 만들어줄 수 있는 좋아하는 음악을 재생해 보세요.'
    },
    2: {
        'name': '공포 (Fear)',
        'emoji': '😰',
        'color': '#9C27B0',
        'desc': '마음이 긴장되고 어딘지 불안하며 깜짝 놀란 상태입니다. 심리적 안전지대가 가장 절실한 때이니, 세상의 번잡함을 뒤로하고 조용히 푹 쉴 수 있는 따뜻한 보금자리에서 나만의 힐링 시간을 보내시기 바랍니다.'
    },
    3: {
        'name': '기쁨 (Happy)',
        'emoji': '😊',
        'color': '#FFEB3B',
        'desc': '행복한 미소와 건강하고 긍정적인 활기찬 에너지가 온 얼굴에 가득 채워져 있습니다! 당신의 빛나는 미소는 본인뿐 아니라 마주 보는 타인의 하루까지도 눈부시게 밝혀줍니다. 이 멋진 기분을 만끽하세요! 🎉'
    },
    4: {
        'name': '슬픔 (Sad)',
        'emoji': '😭',
        'color': '#2196F3',
        'desc': '마음의 에너지가 다소 가라앉아 외롭거나 속상한 감정이 조용히 흐르는 상태입니다. 슬플 땐 무리해서 웃으려 하기보다, 가까운 지인과 대화를 나누거나 차분한 산책을 돌며 억눌린 마음을 토닥토닥 돌봐주세요.'
    },
    5: {
        'name': '놀람 (Surprise)',
        'emoji': '😲',
        'color': '#FF9800',
        'desc': '눈과 입이 동그랗게 커지며 뜻밖의 신선한 정보나 충격에 즉각 반응한 흥미로운 표정입니다. 좋은 충격이라면 온몸으로 이 기분 좋은 반전(Surprise)의 설렘을 친구들과 한껏 즐겨보시길 바랍니다!'
    },
    6: {
        'name': '무표정 (Neutral)',
        'emoji': '😐',
        'color': '#9E9E9E',
        'desc': '특별한 감정의 파동 없이 마음이 잔잔하고 고요하며 아주 차분하게 정돈된 중립(Neutral) 상태입니다. 무언가에 온전히 집중하거나 맑은 정신으로 차분히 생각을 정리하고 계획을 세우기에 황금과도 같은 최적의 순간입니다.'
    }
}


# ------------------------------------------------------------------------------
# 🚀 4. 메인 스트림릿 웹 애플리케이션 레이아웃 구동
# ------------------------------------------------------------------------------
def main():
    # 상단 메인 비주얼 배너 마크다운 출력
    st.markdown("""
        <div class="main-header">
            <h1>🧠 AI 실시간 표정 감정 인식 웹 데모</h1>
            <p>사전 학습 완료된 고성능 EfficientNet-B0 딥러닝 신경망을 연동하여 이미지 속 감정 신뢰도를 실시간으로 추론합니다.</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. 사이드바 구성 (모델 상태 및 매핑 안내)
    st.sidebar.markdown("### 🧬 AI 엔진 시스템 구동 현황")
    
    # 모델 로드 개시 (캐싱 덕분에 즉시 불러와집니다)
    model, device = load_emotion_ai_model()
    
    if model is None:
        st.sidebar.error("❌ 가중치 파일('.pth') 미탑재")
        st.sidebar.warning("프로젝트 루트 폴더 내에 'best_emotion_model.pth' 또는 'model.pth' 파일을 업로드해 주십시오.")
        st.error("🚨 가중치 파일 로드 실패! 사이드바의 에러 메커니즘 가이드를 참조해 주세요.")
        return
    else:
        st.sidebar.success("✅ 인공지능 엔진 작동 활성화")
        st.sidebar.info(f"💾 구동 디바이스: [{device.type.upper()}]")
        if device.type == 'cuda':
            st.sidebar.caption(f"🔧 사양: {torch.cuda.get_device_name(0)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ 분류 가능 감정 클래스 (7개)")
    for k, val in EMOTION_INFO.items():
        st.sidebar.markdown(f"{val['emoji']} **{val['name']}**")

    # 2. 이미지 입력 전처리 transforms (평가용과 일치화)
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. 메인 인터랙티브 기능 (파일 업로더)
    st.markdown("### 🖼️ 분석할 인물 이미지 업로드")
    uploaded_file = st.file_uploader(
        "정면 표정이 잘 나온 사진 파일(JPG, JPEG, PNG)을 드래그 앤 드롭으로 업로드해 주세요.",
        type=["jpg", "jpeg", "png"],
        help="얼굴이 크고 밝게 나와야 인공지능이 더 명확히 특징을 잡을 수 있습니다."
    )

    if uploaded_file is not None:
        # 화면을 2개의 Column으로 분할배치 (5:5 비율)
        col1, col2 = st.columns(2, gap="large")
        
        # 1) 왼쪽 컬럼: 업로드된 원본 이미지 미리보기
        with col1:
            st.markdown("#### 👤 원본 이미지")
            try:
                # PIL 이미지 열기
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"이미지 파싱 중 예상치 못한 에러 발생: {e}")
                return

        # 2) 오른쪽 컬럼: 실시간 연산 구동 및 확률 그래프/스토리텔링 카드 출력
        with col2:
            st.markdown("#### 🤖 인공지능 실시간 분석 결과")
            
            with st.spinner("🧠 딥러닝 레이어 순전파(Forward Pass) 고속 연산 가동 중..."):
                # PIL 이미지를 텐서 전처리하여 배치(1) 추가
                input_tensor = val_test_transforms(image).unsqueeze(0).to(device)
                
                # 예측 개시
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # 예측 인덱스 및 신뢰도 확보
                pred_index = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_index])
                
                # 매칭 정보 바인딩
                pred_emotion = EMOTION_INFO[pred_index]
                
            # 최우선 예측 감정 대형 카드 출력
            st.markdown(f"""
                <div class="emotion-card" style="border-left-color: {pred_emotion['color']};">
                    <div class="emotion-title" style="color: {pred_emotion['color']};">
                        {pred_emotion['emoji']} 최종 예측 감정: {pred_emotion['name']}
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 600; margin-bottom: 12px; color: #1e293b;">
                        🎯 인공지능 예측 신뢰도: {confidence * 100:.1f}%
                    </div>
                    <div class="emotion-desc">
                        {pred_emotion['desc']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("##### 📊 7개 전체 표정 감정별 신뢰도 확률 분포")
            
            # 7가지 감정을 확률 높은 순서대로 정렬하여 보기 좋게 출력
            sorted_emotions = sorted(
                [(i, probabilities[i]) for i in range(len(probabilities))],
                key=lambda x: x[1],
                reverse=True
            )
            
            for idx, prob in sorted_emotions:
                em_name = EMOTION_INFO[idx]['name']
                em_emoji = EMOTION_INFO[idx]['emoji']
                em_color = EMOTION_INFO[idx]['color']
                
                # 수평 바 게이지 스타일 스트림릿 기본 progress 기능으로 표현
                col_name, col_gauge = st.columns([4, 6])
                with col_name:
                    st.write(f"{em_emoji} **{em_name}**")
                with col_gauge:
                    # 스트림릿 프로그레스 바는 0.0~1.0을 받습니다.
                    st.progress(float(prob))
                    st.caption(f"{prob * 100:.1f}% 신뢰도")
                    
    else:
        # 파일이 비어있는 디폴트 대기 화면 안내
        st.info("💡 위 슬롯에 이미지 파일을 업로드해 주시면, 인공지능이 즉시 순전파 연산을 돌아 표정을 실시간 분석합니다.")
        
        # 데모용 팁 박스 배치
        st.markdown("""
            > [!TIP]
            > **실제 구동 노하우**: 
            > - 컴퓨터 카메라(웹캠)나 스마트폰으로 얼굴 정면이 뚜렷하게 보이도록 조명 아래에서 찍은 정면 사진을 활용해 보시는 것을 적극 추천드립니다.
            > - 감정을 명확히 연출하여 지어주실수록(예: 크게 활짝 웃기, 인상 팍 찌푸리기 등) 인공지능의 확률 분포 바 차트 요동이 훨씬 역동적으로 나타나 시각적 효과가 배가됩니다!
        """)


if __name__ == "__main__":
    main()
