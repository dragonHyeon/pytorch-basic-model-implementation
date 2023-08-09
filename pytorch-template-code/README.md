# PyTorch Code Template
- 파이토치 딥 러닝 템플릿 코드
___
### 프로그램 실행 방법
- 학습
  - python -m visdom.server 실행
  - python Main/run_train.py 실행
- 테스트
  - python Main/run_test.py 실행
---
### 프로그램 기능
- 학습 및 테스트
- 모델 파일 저장 및 불러오기
- 학습 진행 과정 그래프로 시각화
---
### 프로그램 구조
- Main/run_train.py 및 Main/run_test.py 에서 디바이스, 모델, optimizer, dataloader, 손실 함수, metric 등 모두 선언 및 실행
---
### 추후 프로그램 modification 방법
- model:
  - DeepLearning/model.py 수정
- 학습:
  - DeepLearning/train.py 에서 Trainer._train 수정
  - 세부 학습 조건은 DeepLearning/train.py 에서 Trainer.running 수정
- 테스트:
  - DeepLearning/test.py 에서 Tester._test 수정
  - 세부 테스트 조건은 DeepLearning/test.py 에서 Tester.running 수정
- 데이터셋:
  - DeepLearning/dataloader.py 수정
- 손실 함수:
  - DeepLearning/loss.py 수정
- metric:
  - DeepLearning/metric.py 수정
