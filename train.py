# -*- coding: utf-8 -*-
"""
🧠 EfficientNet-B0 기반 표정 감정 인식 모델 실전 학습 및 평가 통합 스크립트 (train.py)

이 스크립트는 주피터 노트북 환경 외부의 터미널(Command Prompt, PowerShell 등)에서 
한 번의 명령어로 전체 표정 딥러닝 모델의 학습(Training)과 최종 테스트 평가(Test Evaluation)를 
일사천리로 수행할 수 있도록 특별히 설계된 고품질 인공지능 파이프라인 실행 스크립트입니다.

[사용방법]:
1. 터미널을 열고 가상환경을 활성화한 후, 아래 명령어를 실행합니다.
   python train.py
"""

import os
import sys
import time
import subprocess

# ------------------------------------------------------------------------------
# 🛠️ 1. 필수 모듈 유효성 검사 및 자동 설치 로직
# ------------------------------------------------------------------------------
required_packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PIL": "pillow",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm"
}

print("🔍 [1단계] 딥러닝 구동 라이브러리 탑재 여부 체크 중...")
for module_name, pip_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"⏳ 진행을 위해 필수 패키지 '{pip_name}'을(를) 가상환경에 긴급 설치하는 중입니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ '{pip_name}' 설치 완료!")
        except Exception as e:
            print(f"❌ '{pip_name}' 설치에 실패했습니다. 수동으로 'pip install {pip_name}'을 진행해 주세요. 에러: {e}")
            sys.exit(1)

# 실제 패키지 불러오기
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 🧬 2. 딥러닝 네트워크 구조 수립 (EfficientNet-B0 개조)
# ------------------------------------------------------------------------------
def get_emotion_efficientnet(num_classes=7):
    """
    사전 학습된 고성능 EfficientNet-B0 모델을 불러와 감정 클래스 개수(7개)로 출력층을 개조합니다.
    """
    print("\n🚀 [2단계] 사전 학습된 세계 수준의 EfficientNet-B0 가중치를 불러옵니다...")
    # 완성도 높은 ImageNet 기본 지식이 탑재된 가중치 로딩
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # 마지막 출력 분류기(Classifier)의 직전 입력 차원(1280차원) 확보
    in_features = model.classifier[1].in_features
    
    # 우리의 7가지 감정 클래스를 판별하도록 최종 리니어 계층 스위칭
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# ------------------------------------------------------------------------------
# 🔄 3. 메인 학습 및 실전 평가 실행 파이프라인
# ------------------------------------------------------------------------------
def main():
    # 데이터 폴더 정의
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'
    
    # 폴더 존재 체크
    if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
        print("\n❌ [오류] 데이터 세트 폴더를 찾을 수 없습니다!")
        print("  - 가이드: 'dataset_splitter.ipynb' 노트북을 먼저 끝까지 구동하여")
        print(f"    '{train_dir}', '{val_dir}', '{test_dir}' 폴더가 정상적으로 생성되었는지 확인해 주세요.")
        return

    # GPU 연산 가속장치(CUDA) 감지
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ [3단계] AI 학습에 적용할 하드웨어 가속 장치: [{device.type.upper()}]")
    if device.type == 'cuda':
        print(f"  - 그래픽 카드 기종명: {torch.cuda.get_device_name(0)}")
    else:
        print("  - 가이드: NVIDIA 그래픽카드가 없는 경우 CPU로 자동 전환되어 다소 훈련이 더딜 수 있습니다.")

    # 4. 데이터셋 전처리(Transforms) 규격 수립
    print("\n🖼️ [4단계] 이미지 전처리 및 증강(Data Augmentation) 파이프라인 구성 중...")
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 과적합 방지용 랜덤 좌우 대칭
        transforms.RandomRotation(degrees=10),  # 랜덤 10도 왜곡 회전
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 5. ImageFolder 및 데이터 로더(DataLoader) 로드
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=val_test_transforms)

    batch_size = 32
    # 터미널 전용이므로 num_workers를 2~4 정도로 올려 성능을 높일 수 있으나 Windows의 호환성을 위해 0으로 기본 안전 세팅합니다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    print(f"  - 🏋️ 학습용(Train) 데이터 수 : {len(train_dataset):,}장")
    print(f"  - 🧪 검증용(Val) 데이터 수   : {len(val_dataset):,}장")
    print(f"  - 🔬 실전 테스트(Test) 데이터 수: {len(test_dataset):,}장")
    print(f"  - 🏷️ 감정 매핑 정보: {train_dataset.class_to_idx}")

    # 6. 모델 껍데기 선언 및 장치 이식
    model = get_emotion_efficientnet(num_classes=num_classes)
    model.to(device)

    # 7. 오차함수 및 최적화함수 바인딩
    criterion = nn.CrossEntropyLoss()
    # 파인 튜닝 시 사전 지식이 붕괴하지 않도록 세밀한 학습률 lr=1e-4 채택
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 8. 실전 훈련 및 검증 루프 시작 (Epoch 3회)
    num_epochs = 3
    best_val_acc = 0.0
    model_save_path = 'best_emotion_model.pth'

    print(f"\n🏁 [5단계] 실전 딥러닝 훈련 세션을 시작합니다 (총 {num_epochs}회 반복)...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n🔄 [에포크 {epoch+1} / {num_epochs}] 진행 중...")
        
        # --- 🏋️ 훈련 단계 ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        # 터미널 훈련 진행률을 아름답게 표시하기 위해 tqdm 프로그레스 바 작동
        train_bar = tqdm(train_loader, desc=f"  🏋️ 학습 진행", unit="batch")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
            # 진행 표시바 오른쪽에 오차 실시간 출력
            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_corrects.double() / len(train_dataset)

        # --- 🧪 검증 단계 ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        val_bar = tqdm(val_loader, desc=f"  🧪 검증 진행", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                val_bar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_corrects.double() / len(val_dataset)
        
        epoch_time = time.time() - epoch_start
        
        print(f"📊 [결과 요약] 에포크 {epoch+1} 완료 (소요시간: {epoch_time:.1f}초)")
        print(f"  - 🏋️ 학습용 손실(Loss): {epoch_train_loss:.4f} | 학습 정확도(Acc): {epoch_train_acc:.4f}")
        print(f"  - 🧪 검증용 손실(Loss): {epoch_val_loss:.4f} | 검증 정확도(Acc): {epoch_val_acc:.4f}")

        # 최고 검증 정확도 도달 시점에 파일 백업 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  🌟 [성능 돌파] 역대 최고 정확도 경신! 최적 가중치 파일 '{model_save_path}' 저장 완료.")

    print(f"\n🎉 모든 학습 반복 세션이 성황리에 종료되었습니다! 최고 검증 정확도: {best_val_acc:.4f}")

    # --------------------------------------------------------------------------
    # 💾 9. 저장된 최고의 가중치 모델을 로드하여(읽어서) 최종 테스트 세트 객관적 평가
    # --------------------------------------------------------------------------
    print("\n💾 [6단계] 디스크에 안전하게 보관된 베스트 가중치를 다시 읽어옵니다(Load)...")
    if not os.path.exists(model_save_path):
        print(f"❌ [오류] 저장된 모델 파일 '{model_save_path}'을 찾을 수 없어 최종 평가를 건너뜁니다.")
        return
        
    # 평가 전용 빈 껍데기 모델 구축
    evaluation_model = get_emotion_efficientnet(num_classes=num_classes)
    # 디스크로부터 가중치 주입 (읽어들이기)
    evaluation_model.load_state_dict(torch.load(model_save_path, map_location=device))
    evaluation_model.to(device)
    print(f"✅ 최적 가중치 파일 '{model_save_path}'을 온전히 로딩했습니다!")

    # 최종 실전 평가 시작
    print("\n🔬 [7단계] 한 번도 모델이 접해보지 못한 독립된 [Test Set]으로 최종 실력을 검증합니다...")
    evaluation_model.eval()
    total_test_loss = 0.0
    total_test_corrects = 0
    
    test_bar = tqdm(test_loader, desc=f"  🔬 실전 테스트", unit="batch")
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = evaluation_model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            total_test_loss += loss.item() * inputs.size(0)
            total_test_corrects += torch.sum(preds == labels.data)
            
            test_bar.set_postfix(loss=loss.item())

    final_test_loss = total_test_loss / len(test_dataset)
    final_test_acc = total_test_corrects.double() / len(test_dataset)

    print("\n===========================================================")
    print("🏆 [최종 실전 인공지능 검증 보고서] 🏆")
    print(f"- 📉 실전 테스트 평균 손실값(Test Loss)     : {final_test_loss:.4f}")
    print(f"- 📈 실전 테스트 최종 정확도(Test Accuracy) : {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print("===========================================================")
    print("✨ 이제 주피터 노트북의 8단계를 통해 학습된 모델의 감정 추론 결과를 눈으로 확인해 보세요!")

if __name__ == "__main__":
    main()
