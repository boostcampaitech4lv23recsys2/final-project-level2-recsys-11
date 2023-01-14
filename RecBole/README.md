## Recbole 사용법 (2023/01/15)
1. data 먼저 다운로드 받는다.
```
python data.py
```
2. 그냥 원래처럼 각기 모델 하나 돌리는건 원하는 모델 py 를 실행한다.
```
python BPR.py
python EASE.py
```
- 이후 saved 폴더에 pth 파일 저장된다.
3. 하이퍼파라미터 모든 경우의 수 돌리기는 sh 파일을 이용한다.
```
bash BPR.sh
bash EASE.sh
```
- shell 파일을 돌리면 saved 폴더 내 타겟 모델의 모든 pth 를 지우고 시작한다.
- 경우의수 pth 만 깔끔하게 남기기 위해서
- bash BPR.sh 실행하면 saved/ 에 있던 BPR 모델 pth 파일들이 삭제되고 실행된다.

4. collector py 파일을 실행하면 위의 saved 폴더 내에 있는 pth 파일들 중 model 명에 맞는 pth 파일들을 모두 로드한 뒤, Recbole/../[model]_config 폴더를 생성하고 여기에 pickle 로 저장한다.

```
python BPR_collector.py
python EASE_collector.py
```
