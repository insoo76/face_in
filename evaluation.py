# -*- coding: utf-8 -*-
"""
🧠 EfficientNet-B0 표정 감정 인식 모델 실전 평가 및 오분류 이미지 시각화 자동화 스크립트 (evaluation.py)

이 스크립트는 주피터 노트북 파일의 웹 브라우저 동기화(캐싱) 문제 등으로 인해 노트북 실행이 원활하지 않을 때,
터미널(Cmd, PowerShell)에서 단 한 줄의 명령어로 인공지능 모델의 성능을 정밀 평가하고
시각화 차트 결과물(혼돈행렬 및 오분류 이미지)을 이미지 파일로 디스크에 즉시 저장해주는 초보자용 고품질 스크립트입니다.

[사용 방법]:
1. 터미널을 열고 가상환경을 활성화한 후, 아래 명령어를 실행합니다.
   python evaluation.py
2. 실행이 완료되면 프로젝트 폴더 내에 'confusion_matrix.png' 및 'incorrect_samples.png'가 자동 생성됩니다.
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 평가 통계 모듈
from sklearn.metrics import confusion_matrix, classification_report

# 💻 Windows OS 기반 맑은 고딕 한글 폰트 적용 및 마이너스 깨짐 방지
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)


# ------------------------------------------------------------------------------
# 🧬 1. EfficientNet-B0 모델 아키텍처 정의 (원래 학습 가중치 크기인 7개 감정으로 고정)
# ------------------------------------------------------------------------------
def get_emotion_efficientnet(num_classes=7):
    """
    사전 학습된 고성능 EfficientNet-B0를 불러와 원래 학습 시 감정 개수(7개)에 맞춘 리니어 계층을 장착합니다.
    """
    # ImageNet 기본 지식이 탑재된 고성능 가중치 로딩
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # 마지막 출력 분류기(Classifier)의 직전 입력 차원(1280차원) 확보
    in_features = model.classifier[1].in_features
    
    # 7개의 원래 감정 클래스를 판별하도록 최종 리니어 계층 스위칭
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ------------------------------------------------------------------------------
# 🔄 2. 메인 평가 및 시각화 파일 생성 실행 파이프라인
# ------------------------------------------------------------------------------
def main():
    # 감정 인덱스 번호를 직관적인 한국어 감정명으로 변환하는 사전(Dictionary) 정의
    EMOTION_MAP = {
        '0': '화남 (Angry)',
        '1': '혐오 (Disgust)',
        '2': '공포 (Fear)',
        '3': '기쁨 (Happy)',
        '4': '슬픔 (Sad)',
        '5': '놀람 (Surprise)',
        '6': '무표정 (Neutral)'
    }

    # 데이터 폴더 경로 정의
    test_dir = 'data/test'
    
    if not os.path.exists(test_dir):
        print("\n❌ [오류] 테스트 세트 폴더 'data/test'를 찾을 수 없습니다!")
        print("  - 가이드: 데이터셋이 정상적으로 분할되었는지 먼저 확인해 주세요.")
        return

    # GPU 연산 가속장치(CUDA) 감지
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ [1단계] 평가 연산 하드웨어 디바이스: [{device.type.upper()}]")
    if device.type == 'cuda':
        print(f"  - 그래픽 카드 기종: {torch.cuda.get_device_name(0)}")

    # 3. 테스트 데이터셋 전처리(Transforms) 수립 (학습 왜곡은 적용 안 함)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. ImageFolder 및 데이터 로더(DataLoader) 로드
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("\n📂 [2단계] 테스트용 데이터셋 연동 완료!")
    print(f"  - 🔬 실전 테스트 이미지 수: {len(test_dataset):,}장")
    print(f"  - 🏷️ 테스트셋 폴더 목록   : {test_dataset.classes}")
    print(f"  - 📊 감정 클래스 색인 정보: {test_dataset.class_to_idx}")

    # 5. 가중치 파일 탐색 및 로딩 (Error 원천 극복형)
    model_candidates = ['best_emotion_model.pth', 'model.pth']
    selected_model_path = None

    for candidate in model_candidates:
        if os.path.exists(candidate):
            selected_model_path = candidate
            break

    if selected_model_path is None:
        print("\n❌ [오류] 저장된 모델 가중치 파일('best_emotion_model.pth' 또는 'model.pth')을 찾을 수 없습니다.")
        return
        
    print(f"\n🎯 [3단계] 최적 가중치 파일 감지: [{selected_model_path}] 로딩 시작...")
    
    # 💡 [핵심 보완] 가중치 데이터 차원(7)과 완벽 호환되도록 num_classes=7로 고정 생성
    evaluation_model = get_emotion_efficientnet(num_classes=7)
    evaluation_model.load_state_dict(torch.load(selected_model_path, map_location=device))
    evaluation_model.to(device)
    evaluation_model.eval() # 평가 모드 활성화
    print("  - ✅ 가중치 로딩 및 GPU/CPU 장치 이식 성공!")

    # 6. 실전 테스트 데이터 평가 시작
    print("\n🔬 [4단계] 테스트 이미지 정밀 추론 및 오차 평가 연산 중...")
    all_preds = []
    all_labels = []
    all_confidences = []
    total_test_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  🔬 정밀 추론 중"):
            inputs = inputs.to(device)
            
            # 💡 [라벨 보정 매핑] ImageFolder의 임의 인덱스(0~3)를 실제 감정 폴더 정수(0, 2, 3, 4)로 동적 변환
            mapped_labels = torch.tensor([int(test_dataset.classes[l]) for l in labels]).to(device)
            
            outputs = evaluation_model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)
            
            total_test_corrects += torch.sum(preds == mapped_labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mapped_labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    final_test_acc = total_test_corrects.double() / len(test_dataset)
    print(f"\n🎉 실전 테스트 완료! 최종 정확도(Accuracy): {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")

    # 7. 정량적 종합 분석 보고서 출력
    print("\n====================== 🏆 정량적 종합 인공지능 평가 리포트 ======================")
    unique_present_labels = sorted(list(set(all_labels)))
    target_report_names = [EMOTION_MAP.get(str(l), f"라벨 {l}") for l in unique_present_labels]

    print(classification_report(
        all_labels, 
        all_preds, 
        labels=unique_present_labels, 
        target_names=target_report_names, 
        digits=4
    ))
    print("=================================================================================")

    # 8. 한글 혼돈행렬(Confusion Matrix) 이미지 파일로 저장
    print("\n📈 [5단계] 한글 혼돈행렬(Confusion Matrix) 이미지 파일 생성 중...")
    korean_labels = [EMOTION_MAP.get(str(l), f"라벨 {l}") for l in unique_present_labels]
    cm = confusion_matrix(all_labels, all_preds, labels=unique_present_labels)

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm, 
        annot=True,              
        fmt='d',                 
        cmap='Blues',            
        xticklabels=korean_labels, 
        yticklabels=korean_labels,
        cbar=True                
    )
    plt.title('🧠 감정 인식 인공지능 실전 혼돈행렬 (Confusion Matrix)', fontsize=14, pad=20, weight='bold')
    plt.xlabel('🤖 인공지능의 예측 감정 (Predicted Label)', fontsize=11, labelpad=12)
    plt.ylabel('👤 실제 정답 감정 (True Label)', fontsize=11, labelpad=12)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 이미지 파일 저장
    cm_save_path = 'confusion_matrix.png'
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"  - 🌟 저장 완료: '{cm_save_path}'")

    # 9. 틀린 건 무엇일까? 오분류 이미지 파일로 시각화 저장
    print("\n❌ [6단계] 예측 오차(오분류) 이미지 분석 및 시각화 저장 중...")
    incorrect_indices = []
    correct_indices = []

    for idx in range(len(all_labels)):
        if all_preds[idx] != all_labels[idx]:
            incorrect_indices.append(idx)
        else:
            correct_indices.append(idx)

    num_samples_to_show = min(5, len(incorrect_indices))

    if num_samples_to_show == 0:
        print("  - 🎉 대단합니다! 오분류된 이미지가 존재하지 않아 이미지 저장을 건너뜁니다.")
    else:
        sampled_incorrect_indices = random.sample(incorrect_indices, num_samples_to_show)
        plt.figure(figsize=(18, 6))
        class_names = test_dataset.classes
        
        for count, test_index in enumerate(sampled_incorrect_indices):
            img_tensor, label = test_dataset[test_index]
            
            # 이미지 역정규화 복원
            img_numpy = img_tensor.numpy().transpose((1, 2, 0))
            mean_values = np.array([0.485, 0.456, 0.406])
            std_values = np.array([0.229, 0.224, 0.225])
            img_numpy = std_values * img_numpy + mean_values
            img_numpy = np.clip(img_numpy, 0, 1)
            
            # 실제 및 예측 클래스 감정명 변환
            true_folder_name = class_names[label]
            true_name = EMOTION_MAP.get(true_folder_name, true_folder_name)
            
            pred_index_str = str(all_preds[test_index])
            pred_name = EMOTION_MAP.get(pred_index_str, pred_index_str)
            
            conf_percent = all_confidences[test_index] * 100
            
            plt.subplot(1, num_samples_to_show, count + 1)
            plt.imshow(img_numpy)
            plt.title(
                f"👤 정답: {true_name}\n🤖 예측: {pred_name}\n🎯 확신도: {conf_percent:.1f}%", 
                color='crimson', 
                fontsize=12, 
                fontweight='bold'
            )
            plt.axis('off')
            
        plt.suptitle(
            f"❌ 인공지능이 실수하여 틀린 대표적인 이미지 시각화 ({num_samples_to_show}장 샘플)", 
            fontsize=16, 
            y=1.03, 
            weight='bold', 
            color='crimson'
        )
        plt.tight_layout()
        
        # 이미지 파일 저장
        incorrect_save_path = 'incorrect_samples.png'
        plt.savefig(incorrect_save_path, dpi=300)
        plt.close()
        print(f"  - 🌟 저장 완료: '{incorrect_save_path}'")

    print("\n👍 모든 객관적 성능 평가 및 이미지 시각화 파일 생성 프로세스가 무사히 종료되었습니다!")


if __name__ == "__main__":
    main()
