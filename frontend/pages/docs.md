## 1. 정량지표와 정성지표란?

- **정량지표**란 _Recall@K_, _NDCG@K_, _MAP@K_ 등 예측한 아이템과 실제 정답으로 계산된 지표로, 쉽게 말해 추천 모델이 얼마나 잘 맞혔는지 의미하는 지표입니다.

- **정성지표**란 _Diversity_, _Serendipity_, _Novelty_와 같은 지표로, 예측한 아이템 혹은 추천된 아이템 리스트가 얼마나 참신하고 다양한, 새로운 아이템을 갖고 있는지 의미하는 지표입니다.

- 정량지표가 조금 낮더라도 정성지표값이 중요하다고 생각될 때는, 이러한 지표들간의 trade-off를 잘 살펴보면서 모델을 선택할 수도 있습니다.

- **정량지표 수식**

    $\\mathrm{Recall@K}=\\frac{1}{|U|}\\sum_{u \\in U} \\frac{|{i \\in I_u | rank_u (i) \\leq K}|}{min(K,|I_u|)}$

    $\\mathrm{MAP@K}=\\frac{1}{|U|}\\sum_{u \\in U} (\\frac{1}{min(|\\hat R(u)|, K)} \\sum_{j=1}^{|\\hat{R}(u)|} I\\left(\\hat{R}_{j}(u) \\in R(u)\\right) \\cdot  Precision@j)$

    $\\mathrm {NDCG@K}=\\frac{1}{|U|}\\sum_{u \\in U} (\\frac{1}{\\sum_{i=1}^{\\min (|R(u)|, K)}\\frac{1}{\\log _{2}(i+1)}} \\sum_{i=1}^{K} \\delta(i \\in R(u)) \\frac{1}{\\log _{2}(i+1)})$

    $\\mathrm {TailPercentage@K}=\\frac{1}{|U|} \\sum_{u \\in U} \\frac{\\sum_{i \\in R_{u}} {\\delta(i \\in T)}}{|R_{u}|}$

    $\\mathrm{AveragePopularity@K}=\\frac{1}{|U|} \\sum_{u \\in U } \\frac{\\sum_{i \\in R_{u}} \\phi(i)}{|R_{u}|}$

    $\\mathrm{Coverage}=\\frac{|\\cup_{u \\in U} R_u|}{|I|}$

- **정성지표 수식**
    - **objective function**

        $\\mathrm{Diversity(R)}=\\frac{\\sum_{i \\in R} \\sum_{j \\in R \\setminus \\{i\\} }  dist(i,j)}{|R|(|R|-1)}$
        
        $\\mathrm{Serendipity(R,u)=\\frac{|R_unexp \\cap R_useful|}{|R|}}$
        
        $\\mathrm{Novelty(R)}=\\frac{\\sum_{i \\in R} - \\log_2 p(i)}{|R|}$ 
    - **dist function**
        - _rating (cosine)_ : interaction matrix를 이용한 함수입니다.
        
        $\\mathrm{dist(i,j)}=\\frac{1}{2} - \\frac{\\sum_{u \\in U}(r_{ui}-\\bar{r_i})}{2\\sqrt{\\sum_u \\in U (r_{ui}-\\bar{r_i}^2)}\\sqrt{\\sum_u \\in U (r_{uj}-\\bar{r_j}^2)}}$

        - _jaccard_ : genre와 같은 item side information 등을 이용한 함수입니다.
        
        $\\mathrm{dist(i,j)}=1-\\frac{|L_i \\cap L_j|}{|L_i \\cup L_j|}$

        - _PMI_ : interaction matrix를 이용한 함수입니다.
        
        $\\mathrm{dist(i,j)}=\\frac{\\log_2\\frac{p(i,j)}{p(i)p(j)}}{-\\log_2p(i,j)}$, $p(i)=\\frac{|\\{u \\in U, r_{ui} \\neq \\emptyset\\}|}{|U|}$, $p(i,j)=\\frac{|\\{u \\in U, r_{ui} \\neq \\emptyset \\wedge r_{uj} \\neq \\emptyset\\}|}{|U|}$

    - reference : https://dl.acm.org/doi/abs/10.1145/2926720

## 2. Reranking이란?

- 위에서 정의한 정성지표들을 고려한 추천 리스트를 반영하기 위한 일종의 재정렬 방법입니다. 과정은 다음과 같습니다.
    
    1. 정성지표중 Reranking에 사용할 목적함수를 고릅니다. e.g., Diversity-jaccard
    
    2. Reranking 정도를 조절할 parameter $\\alpha$를 정합니다.

    3. $f_{obj}(i,R) = \\alpha \\cdot rel(i) + (1-\\alpha) \\cdot obj(i,R)$ 값을 계산한 뒤 Top K개의 아이템을 추천합니다.
        - $rel(i)$는 기존 모델이 추천한 스코어이며, $obj(i,R)$은 Reranking에 사용할 목적함수로써 가중합으로 최종 스코어를 계산해 Top K 추천이 이루어집니다.

        
- 간단한 연산을 하는 일종의 새로운 모델을 이용해 추천 결과물을 만들어 내는 과정이기 때문에 시간이 꽤 소요되는 작업입니다.
    
- Reranking 전 후의 정량, 정성 지표들과 추천 결과물을 비교하여 원하는 정도의 Reranking을 설정하는 alpha를 조절하며 자신만의 모델을 만들어보세요!

- **Hint: Reranking에 쓰이는 목적함수와 정성지표들간의 관계를 살펴보면 아주 재밌답니다!**

## 3. Web4Rec Library란?

- Web4Rec을 사용할 땐 라이브러리를 통해 업로드 합니다.
- 라이브러리 코드를 다운받아 추천시스템 학습 프로젝트에서 시작할 수 있습니다.

```
from web4rec.web4rec import Web4Rec, Web4RecDataset
Web4Rec.login(token='...')
```
1. web4rec 패키지에서 Web4Rec, Web4RecDataset 클래스를 임포트합니다.
2. 웹사이트에서 회원가입 시 발급 받는 API token을 login 함수에 인자로 보내면 스크립트 상에서 로그인 할 수 있습니다.

```
w4r_dataset = Web4RecDataset(dataset_name='ml-1m')

w4r_dataset.add_train_interaction(train_interaction: pd.DataFrame)
w4r_dataset.add_ground_truth(ground_truth: pd.DataFrame)
w4r_dataset.add_user_side(user_side: pd.DataFrame)
w4r_dataset.add_item_side(item_side: pd.DataFrame)

Web4Rec.register_dataset(w4r_dataset)
```
3. Web4RecDataset 클래스로 객체를 생성후 네 가지 멤버함수를 호출해 객체 내 필요 데이터를 삽입합니다.
(다음과 같은 데이터프레임 형태를 인자에 전달해야 합니다.)

    - train_interaction은 상호작용이 있을 때의 'user_id', 'item_id' 쌍이 행으로 주어집니다.
    - timestamp 정보가 있다면 유저별, 시간별로 정렬 후 삽입해 주세요.
    - ground truth 또한 train interaction과 같은 형식입니다.
    - item_side, user_side 는 각각 'item_id', 'user_id' 컬럼을 지녀야 합니다.
    - 모든 side information 컬럼은 범주형 정보를 가정하며, 수치형의 경우 반드시 구간 범주화를 진행해야 합니다.
    - item_side 에서 한 범주에서 여러 특성을 갖는 경우 컬럼 이름에 ':multi' postfix 를 붙여야 하며, 각 셀은 띄어쓰기로 구분되어야 합니다.
    - 위와 같은 전처리를 진행하면 라이브러리 내부에서 이 컬럼을 아이템 멀티 핫 벡터로 계산하고, jaccard 관련 정성 지표들을 구할 수 있습니다.

4. 이후 Web4Rec 전역 함수 register_dataset 을 호출하여 데이터셋 객체를 전달합니다. 

    - 데이터셋 객체가 이미 서버에 존재하면 pass 하고, 없으면 서버에 업로드합니다.

```
# 학습 진행 후 prediction score matrix 구성

prediction_matrix = pd.DataFrame(
    data = user_item_prediction_score_matrix,
    index = user_ids,
    columns = item_ids
)

Web4Rec.upload_experiment(
    experiment_name='...',
    hyper_parameters={
        '...' = '...',
        '...' = 1234
    },
    prediction_matrix = prediction_matrix
)
```
5. Web4Rec upload_experiment 함수를 호출하여 세 가지 인자를 전달하면, 함수가 종료되면서 각종 실험결과를 서버에 업로드 합니다.

    - prediction_matrix 데이터 프레임은 행은 user_id, 컬럼은 item_id 로 각 유저, 아이템 페어별 모델의 prediction score로 구성됩니다.
    - 반드시 유저, 아이템은 위의 데이터셋 내 user_id, item_id 와 같아야 합니다.
    - 여러 유저와 아이템으로 구성할수록 t-sne 정보가 구체화 됩니다.
    - 유저 아이템 각각 최소 100개 이상 최대 20,000개 이하를 추천합니다.  



## 4. Item vector 추출과 t-sne는 어떻게 이루어지나요?

- prediction matrix 만들기
    - 모든 모델에서 Item vector를 추출하기 위해서는 prediction matrix가 필요합니다. 
    - prediction matrix란, 모델이 모든 User별로 예측한 Item의 선호도 혹은 score를 의미합니다. 
    - 또한, 생성된 prediction matrix는 dense한 형태이지만 threshold 값을 지정해 sparse matrix로 변경하여 메모리를 절약할 수도 있습니다.


- embedding vector 생성하기
    - 저희 library에서는 2단계로 나누어 최종적으로 2차원 embedding vector를 생성합니다. 
    - 먼저 (user size, item size) 크기인 prediction matrix를 transpose를 취해 (item size, user size)로 변경 시켜준뒤, scikit-learn library의 pca를 활용해 50차원으로 축소시킵니다.
    - 그 다음으로, 50차원으로 축소된 embedding vector를 sklearn.manifold library를 활용해 2차원으로 축소합니다.
    - 이 때, 1차적으로 pca를 활용해 축소시키고 싶은 차원을 정할 수 있습니다