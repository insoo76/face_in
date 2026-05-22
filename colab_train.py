# -*- coding: utf-8 -*-
"""
Google Colab(구글 코랩) 전용 GPU 초고속 감정 인식 모델 훈련 통합 코드 (colab_train.py)

[사용법]:
이 스크립트 파일의 전체 코드를 복사하셔서 구글 코랩(Colab)의 새 코드 셀 하나에 
몽땅 붙여넣기 하신 후, 런타임 유형을 T4 GPU로 변경하고 재생(Run) 버튼을 누르시면 됩니다!
"""

import sys
import io

# ==============================================================================
# 🛠️ 윈도우 터미널 한글/이모지 출력 인코딩(CP949) 크래시 완벽 극복 방어 코드
# ==============================================================================
if sys.platform.startswith('win'):
    try:
        # 표준 출력 및 표준 에러의 인코딩을 UTF-8로 강제 보정하여 윈도우 UnicodeEncodeError를 완벽 예방합니다.
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

import os
import time
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ==============================================================================
# tqdm 진행 표시줄 로드 (코랩 노트북용, 로컬 터미널용, 미설치용 자동 예외 우회 지원)
# ==============================================================================
try:
    # 1. 구글 코랩 등 주피터 환경 친화형 notebook tqdm 시도
    from tqdm.notebook import tqdm
    print("✨ 코랩 전용 시각화 tqdm 로딩 성공!")
except ImportError:
    try:
        # 2. 일반 로컬 터미널 환경을 대비한 표준 tqdm 시도
        from tqdm import tqdm
        print("✨ 일반 터미널용 tqdm 로딩 성공!")
    except ImportError:
        # 3. 만약 tqdm 라이브러리가 아예 설치되어 있지 않은 극단적인 환경 대비 수동 Mock tqdm 클래스 우회
        print("ℹ️ tqdm 라이브러리가 미설치되어 수동 텍스트 출력 모드로 자동 전환합니다.")
        class tqdm:
            def __init__(self, iterable, *args, **kwargs):
                self.iterable = iterable
                self.desc = kwargs.get('desc', '')
            def __iter__(self):
                for i, item in enumerate(self.iterable):
                    yield item
            def set_postfix(self, *args, **kwargs):
                pass

print("==============================================================================")
print("🧠 구글 코랩 표정 감정 인식(EfficientNet-B0) 초고속 학습 시스템 가동")
print("==============================================================================")

# ==============================================================================
# 1단계: 코랩에 업로드한 data.zip 압축 풀기 (파이썬 표준 라이브러리 사용으로 SyntaxError 완벽 예방!)
# ==============================================================================
zip_file_path = "data.zip"
train_dir = 'data/train'
val_dir = 'data/val'

if os.path.exists(zip_file_path):
    print("⏳ [1단계] 구글 클라우드 공간에 업로드된 data.zip 압축 해제를 개시합니다...")
    # 파이썬 내장 zipfile 모듈을 사용하므로, 코랩뿐 아니라 로컬에서도 문법 에러 없이 완벽하게 작동합니다!
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ 압축 해제 성공! data 폴더 구조가 완벽하게 복원되었습니다.")
elif not os.path.exists(train_dir):
    print("\n❌ [심각한 오류] 코랩 왼쪽 파일(폴더 모양 아이콘) 영역에 'data.zip'이 업로드되지 않았습니다!")
    print("📢 해결 방법:")
    print("  1. 로컬에서 압축한 'data.zip' 파일을 코랩의 왼쪽 파일 창으로 드래그앤드롭 하세요.")
    print("  2. 업로드 완료 표시(왼쪽 아래 동그라미 그래프 완료)를 확인하고 다시 실행해 주세요.")
    print("⚠️ 훈련을 계속할 수 없으므로 프로그램을 안전하게 종료합니다.")
    sys.exit(1)
else:
    print("ℹ️ 이미 압축 해제된 'data' 폴더가 감지되었습니다. 압축 해제 단계를 건너뜁니다.")

# ==============================================================================
# 2단계: 필수 라이브러리 로드 및 가속 장치(GPU) 세팅
# ==============================================================================
print(f"\n🔥 PyTorch 버전: {torch.__version__}")

# GPU(CUDA) 존재 여부 체크 (코랩 런타임 유형이 T4 GPU로 되어있어야 CUDA가 활성화됩니다)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 현재 코랩 가속에 할당된 딥러닝 연산 디바이스: [{device.type.upper()}]")
if device.type == 'cuda':
    print(f"  - 할당된 고성능 GPU 기종명: {torch.cuda.get_device_name(0)}")
else:
    print("\n⚠️ [경고] 현재 CPU 환경으로 작동하고 있습니다!")
    print("📢 해결 방법:")
    print("  - 코랩 상단 메뉴 [런타임] ➡️ [런타임 유형 변경]에서 하드웨어 가속기를 'T4 GPU'로 꼭 선택해 주세요!")
    print("  - 그래야 10분 이내로 초고속 훈련을 마칠 수 있습니다.")
    print("-" * 75)

# ==============================================================================
# 3단계: 이미지 전처리 및 데이터 로더(DataLoader) 로드
# ==============================================================================
print("\n🖼️ [2단계] 이미지 고품질 전처리 및 증강 파이프라인 수립 중...")
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 좌우반전 데이터 증강
    transforms.RandomRotation(degrees=10),  # 삐딱한 각도 대비 10도 회전
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 데이터셋 물리적 연동 시작
try:
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

    # 코랩의 강력한 하드웨어를 활용해 속도를 배가하기 위해 num_workers를 2로 높입니다.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print("=== 📂 데이터셋 로딩 리포트 ===")
    print(f"- 🏋️ 훈련용 이미지 수 : {len(train_dataset):,}장 (미니배치: {len(train_loader)}개)")
    print(f"- 🧪 검증용 이미지 수 : {len(val_dataset):,}장 (미니배치: {len(val_loader)}개)")
    print(f"- 🏷️ 클래스 맵핑     : {train_dataset.class_to_idx}")
except Exception as e:
    print(f"\n❌ [오류] 데이터 폴더 읽기 도중 문제가 발생했습니다: {e}")
    print("📢 해결 방법: 'data' 폴더 내부에 train 및 val 폴더와 하위 감정 폴더들이 올바르게 위치해 있는지 확인해 주세요.")
    sys.exit(1)

# ==============================================================================
# 4단계: EfficientNet-B0 조립 및 파인 튜닝 설정
# ==============================================================================
def get_emotion_efficientnet(num_classes=7):
    # 최신 PyTorch 규격에 맞추어 Weights 변수 설정
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    # 출력층을 우리 데이터의 클래스 개수(예: 7개 표정)에 맞춰 리모델링
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

num_emotions = len(train_dataset.classes)
emotion_model = get_emotion_efficientnet(num_classes=num_emotions)
emotion_model.to(device)

criterion = nn.CrossEntropyLoss()
# 미세 최적화 가중치 감쇠 포함 lr=1e-4 적용
optimizer = optim.AdamW(emotion_model.parameters(), lr=1e-4, weight_decay=1e-2)
print("\n🧬 [3단계] EfficientNet-B0 설계 완료 및 GPU 탑재 성공!")

# ==============================================================================
# 5단계: GPU 초고속 훈련 시작 (3 Epochs)
# ==============================================================================
num_epochs = 3
best_val_acc = 0.0
model_save_name = 'best_emotion_model.pth'

print(f"\n🏁 [4단계] 딥러닝 고속 GPU 훈련을 개시합니다 (총 {num_epochs}회 반복)...")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # --- 🏋️ 학습 모드 ---
    emotion_model.train()
    train_loss = 0.0
    train_corrects = 0
    
    # 코랩 노트북 전용 친화형 tqdm 프로그레스 바 구동
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [학습]", unit="batch")
    for inputs, labels in train_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 순전파 -> 역전파 -> 가중치 갱신
        outputs = emotion_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)
        
        train_bar.set_postfix(loss=loss.item())
        
    epoch_train_loss = train_loss / len(train_dataset)
    epoch_train_acc = train_corrects.double() / len(train_dataset)
    
    # --- 🧪 검증 모드 ---
    emotion_model.eval()
    val_loss = 0.0
    val_corrects = 0
    
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [검증]", unit="batch")
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = emotion_model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
            val_bar.set_postfix(loss=loss.item())
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)
    
    epoch_time = time.time() - start_time
    
    print(f"📊 [에포크 {epoch+1} 완료] (소요시간: {epoch_time:.1f}초)")
    print(f"  - 🏋️ 학습용 -> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
    print(f"  - 🧪 검증용 -> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")
    
    # 베스트 성능 갱신 가중치 저장
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(emotion_model.state_dict(), model_save_name)
        print(f"  🌟 [경신] 최고 검증 정확도 돌파! 최적 가중치 '{model_save_name}' 저장 완료.")
    print("-" * 75)

print(f"\n🎉 훈련 대성공! 최고 검증 정확도: {best_val_acc:.4f}")

# ==============================================================================
# 6단계: 학습 완료된 가중치 모델 로컬 컴퓨터로 자동 다운로드 개시
# ==============================================================================
print("\n💾 [5단계] 완성된 최고의 뇌 세포 파일(best_emotion_model.pth)을 로컬로 귀환시킵니다...")
if os.path.exists(model_save_name):
    try:
        # 구글 코랩 전용 파일 다운로드 모듈을 긴급 호출합니다.
        from google.colab import files
        print("📥 사용자님의 웹 브라우저 다운로드 팝업이 활성화됩니다. 다운로드를 수락해 주세요!")
        files.download(model_save_name)
    except ImportError:
        print("ℹ️ 구글 코랩 환경이 아닙니다. 파일 자동 다운로드 단계를 건너뜁니다.")
        print(f"최종 저장된 가중치 파일 경로는 '{os.path.abspath(model_save_name)}' 입니다.")
else:
    print("❌ 에러: 저장된 모델 파일이 없습니다.")
