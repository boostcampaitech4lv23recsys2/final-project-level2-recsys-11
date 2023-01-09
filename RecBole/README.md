## Recbole 사용법
1. data 먼저 다운로드 받는다.
```
python data.py
```
2. 원하는 모델 이름의 yaml 파일을 만든다. (Recbole 명칭과 같아야 한다.)
3. train 실행 시 모델 인자에 넣는다.
```
python train.py --model EASE
```
4. saved 에 저장된 model_~~.pth 파일의 절대경로를 복사한다.
5. predict 실행하여 topk 를 구한다.
```
python inference.py --model_path ~~~.pth
```