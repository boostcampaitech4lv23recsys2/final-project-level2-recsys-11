# final-project-level2-recsys-11
## Web4Rec
Web4rec은 추천 시스템의 모델 평가 및 분석을 돕는 실험 관리 페이지 서비스입니다.

사용자는 자신의 실험 결과를 Web4Rec 라이브러리를 통해 웹페이지에 업로드 할 수 있고, 웹페이지에서 더 좋은 추천 결과를 찾아낼 수 있도록 다각적 분석을 제공 합니다. 

이 프로젝트는 실제 서비스 환경에서 더욱 다양하고 질 좋은 추천 서비스를 제공하기 위해 추천 엔진을 개선하고 싶은 머신러닝 / 딥러닝 엔지니어들 사용할 수 있도록 만들어졌습니다. 

[Web4Rec (Demo Page)](http://101.101.216.84:30002/)


#### 유의할 점

</aside>
Web4Rec 정의하는 정량 / 정성 지표와 차이점

**정량 지표**란 *Recall@K*, *NDCG@K*, *MAP@K* 등 예측한 아이템과 실제 정답으로 계산된 지표로, 쉽게 말해 추천 모델이 얼마나 잘 맞혔는지 정확도를 의미하는 지표입니다.

**정성 지표**란 *Diversity*, *Serendipity*, *Novelty*와 같은 지표로, 예측한 아이템 혹은 추천된 아이템 리스트가 얼마나 참신하고 다양한, 새로운 아이템을 갖고 있는지 의미하는 지표입니다.

</aside>

#### Web4Rec 라이브러리 사용법

```
from web4rec import Web4Rec, Web4RecDataset

Web4Rec.login(token='...') # 회원가입시 제공되는 API 키
```

1. Web4Rec 라이브러리는 파이썬 스크립트 상에서 API 키로 로그인이 가능합니다.
```
w4r_dataset = Web4RecDataset(dataset_name='ml-1m')

# pd.DataFrame 형태의 input
w4r_dataset.add_train_interaction(train_interaction)
w4r_dataset.add_ground_truth(ground_truth)
w4r_dataset.add_user_side(user_side)
w4r_dataset.add_item_side(item_side)

Web4Rec.register_dataset(w4r_dataset)
```


2. Web4RecDataset 은 Web4Rec 자체 데이터셋 관리 클래스 입니다.

- 다음과 같은 형태로 pandas Dataframe 을 준비하여 add 멤버함수를 호출합니다.
- pandas Dataframe 캡처 화면

```
# 학습 진행 후 prediction score matrix 구성
prediction_matrix = pd.DataFrame(
    data = user_item_prediction_score_matrix,
    index = user_ids,
    columns = item_ids
)

Web4Rec.upload_experiment(
    experiment_name='BPR-MF',
    hyper_parameters={
        'negative_sampler' = 'popularity',
        'embedding_size' = 16
    },
    prediction_matrix = prediction_matrix
)
```

#### 웹페이지 기능

**1.실험비교**

- 웹페이지에서는 실험들을 각각의 하이퍼파라미터, 정량 / 정성 지표 값들과 함께 하나의 테이블 (Compare Table)과 다양한 그래프로 비교할 수 있습니다 (Model vs. Model).
- 각 테이블과 그래프는 모두 **interactive**하게 살펴볼 수 있습니다. ([**Plotly](https://plotly.com/python/plotly-fundamentals/), [AgGrid](https://www.ag-grid.com/)**)
    - 사용자가 자신에게 맞게 세부적인 시각화가 가능합니다.



**2.리랭킹**

###### 리랭킹 전략

- 추천 모델의 Top K 후보군을 정성 지표를 활용해 기존 ranking에 변화를 주는 기법입니다.
- 사용자는 리랭킹 기법을 통해 추천 모델의 정확도와 정성 지표의 트레이드 오프를 고려한 추천을 제공할 수 있습니다.
- 이는 추천의 질을 높임으로써 정확도만을 최적화하는 모델의 한계를 보완할 수 있습니다.



**3. 사후 분석**

###### 설명 가능한 추천

- 모델의 추천 결과를 직관적으로 해석할 수 있는 다양한 시각화를 제공합니다.
- 사이드 정보 혹은 2차원으로 축소된 임베딩 그래프를 통해 원하는 유저 / 아이템 군집을 선택하여 분석을 진행할 수 있습니다.
- 선택된 유저 / 아이템 군집에 대한 다양한 시각화를 통해 추천 이유와 관련된 직관적인 분석이 가능합니다.
    - E.g. 한 아이템에 대해 상호작용 이력이 있는 유저와 해당 아이템을 추천받은 유저 간의 사이드 정보 시각화

###### 리랭킹을 통한 심화 분석

- 원하는 유저 군집을 선택한 후 커스터마이징한 리랭킹을 진행할 수 있습니다.
- 기존 추천리스트와 리랭킹을 통해 나온 추천리스트 간 비교를 통해 효과적인 리랭킹 전략을 구상할 수 있습니다.

