# CAM2
아래 논문의 구현 코드입니다.

[Sentiment classification with word localization based on weakly supervised learning with a convolutional neural network](https://www.sciencedirect.com/science/article/abs/pii/S0950705118301710)

### 요구사항

- python3
- tensorflow 1.4 이상
- KoNLPy, Mecab : [설치 방법](http://konlpy.org/en/latest/install/)

### 실행방법

- 데이터 준비 : [네이버 영화 리뷰 코퍼스](https://github.com/e9t/nsmc)를 `data` 디렉토리에 다운로드합니다.

```bash
sh run.sh prepare
```

- 데이터 전처리 및 Convolutional Neural Network 모델 학습

```bash
sh run.sh train
```

- 테스트 (python console) : 학습이 끝나면 CAM2 점수를 확인할 수 있습니다. get_scores 함수는 예측된 범주와 단어별 CAM2 스코어를 반환합니다.

```python
from config import Config
from lib.predict_util import CAM2
model = CAM2(Config())
model.get_scores('재미있다')
model.get_scores('재미없다')
```

- 웹 데모 : 다음을 실행하면 단어별 점수를 로컬 웹페이지로 확인할 수 있습니다.

```bash
sh run.sh web-demo
```