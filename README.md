### Deep Learning, 딥러닝

-   인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
-   인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
-   인간은 학습 시 뇌에 있는 뉴런이 자극을 받아들여 일정 자극 이상이 되면 화학물질을 통해 다른 뉴런과 연결되고 해당 부분이 발달한다.
-   자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
-   입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
-   초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥러닝으로 발전한다.
-   딥러닝에는 Input nodes layer, Hidden nodes layer, Output nodes layer의 3가지 층이 존재한다.

<img src="./a_intro/images/deep_learning.png" style="margin-left: 0">

### SLP (Single Layer Perceptron), 단층(단일) 퍼셉트론

-   가장 단순한 형태의 신경망으로서, Hidden Layer가 없고 Single Layer로 구성되어 있다.
-   퍼셉트론의 구조는 입력 feature와 가중치, activation function, 출력 값으로 구성되어 있다.
-   신경 세포에서 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신하고, 입력 값과 가중치 값은 모두 인공 뉴런(활성 함수)으로 도착한다.
-   가중치의 값이 클 수록 해당 입력 값이 중요하다는 뜻이고, 인공 뉴런(활성 함수)에 도착한 각 입력 값가 가중치 값을 곱한 뒤 전체 합한 값을 구한다.
-   인공 뉴런(활성 함수)은 보통 시그모이드 함수와 같은 계단 함수를 사용하여, 합한 값을 확률로 변환하고 이때 임계치를 기준으로 0 또는 1을 출력한다.

<img src="./b_perceptron/images/perceptron01.png" style="margin-left: 0">

-   로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런으로 볼 수 있다.
-   결과적으로 퍼셉트론은 회귀 모델과 마찬가지로 실제 값과 예측 값의 차이가 최소가 되는 가중치 값을 찾는 과정이 퍼셉트론이 학습하는 과정이다.
-   최초 가중치 값을 설정한 뒤 입력 feature 값으로 예측 값을 계산하고, 실제 값과의 차이를 구한 뒤 이를 줄일 수 있도록 가중치 값을 변경한다.
-   퍼셉트론의 활성화 정도를 편향(bias)으로 조절할 수 있으며, 편향을 통해 어느 정도의 자극을 미리 주고 시작할 수 있다.
-   뉴런이 활성화되기 위해 필요한 자극이 1000이라고 가정하면, 입력값을 500만 받아도 편향을 2로 주어 1000을 만들 수 있다.

<div style="display: flex; width: 500px; margin-left: 20px;">
    <img src="./b_perceptron/images/perceptron02.png" width="200px" style="margin-left: 0">
    <img src="./b_perceptron/images/perceptron03.png" width="225px" style="margin-left: 50px; margin-top: -10px;">
</div>

-   퍼셉트론의 출력값과 실제값의 차이를 줄여나가는 방향성으로 계속해서 가중치 값을 변경하며, 이 때 경사하강법을 사용한다.

<img src="./b_perceptron/images/gd01.gif" width="600px">

#### SGD (Stochastic Gradient Descent), 확률적 경사 하강법

-   경사 하강법 방식은 전체 학습 데이터를 기반으로 계산한다. 하지만 입력 데이터가 크고 레이어가 많을 수록 많은 자원이 소모된다.
-   일반적으로 메모리 부족으로 인해 연산이 불가능하기 때문에, 이를 극복하기 위해 SGD 방식이 도입되었다.
-   전체 학습 데이터 중, 단 한 건만 임의로 선택하여 경사 하강법을 실시하는 방식을 의미한다.
-   많은 건 수 중에 한 건만 실시하기 때문에, 빠르게 최적점을 찾을 수 있지만 노이즈가 심하다.
-   무작위로 추출된 샘플 데이터에 대해 경사 하강법을 실시하기 때문에 진폭이 크고 불안정해보일 수 있다.
-   일반적으로 사용되지 않고, SGD를 얘기할 때에는 보통 미니 배치 경사 하강법을 의미한다.

<img src="./b_perceptron/images/gd02.png" width="600px">

#### Mini-Batch Gradient Descent, 미니 배치 경사 하강법

-   전체 학습 데이터 중, 특정 크기(Batch 크기)만큼 임의로 선택해서 경사 하강법을 실시한다. 이 또한 확률적 경사 하강법이다.

<img src="./b_perceptron/images/gd03.png" width="800px">

-   전체 학습 데이터가 1,000건이라고 하고, batch size를 100건이라고 가정하면, 전체 데이터를 batch size만큼 나눠서 가져온 뒤 섞고, 경사하강법을 계산한다. 이 경우, 10번 반복해야 1,000개의 데이터가 모두 학습되고 이를 epoch라고 한다. 즉, 10 epoch \* 100 batch 이다.

<img src="./b_perceptron/images/gd04.png" width="650px">

### Multi Layer Perceptron (다층 퍼셉트론, 다중 퍼셉트론)

-   보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.
-   퍼셉트론을 여러 층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.
-   층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

<div style="display: flex">
    <div>
        <img src="./b_perceptron/images/mlp01.png" width="500" style="margin-left: 20px">
    </div>
    <div>
        <img src="./b_perceptron/images/mlp02.png" width="500" style="margin-top:50px; margin-left: 20px">
    </div>
</div>

### ANN (Artificial Neural Network), 인공 신경망

-   은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

<img src="./b_perceptron/images/ann.png">

### DNN (Deep Neural Network), 심층 신경망

-   은닉층이 2개 이상일 경우 이를 심층 신경망이라고 한다.

<img src="./b_perceptron/images/dnn.png">

### Back-propagation, 역전파

-   심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
-   즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트하며, 손실 함수의 최솟값을 찾아야 하는데, 순방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.

#### 합성 함수의 미분

<img src="./b_perceptron/images/chain_rule01.png" width="150" style="margin-left: 0">

---

<img src="./b_perceptron/images/chain_rule02.png" width="550" style="margin-left: 0">

<img src="./b_perceptron/images/backpropagation01.png" width="800" style="margin-left: 0">  
<img src="./b_perceptron/images/backpropagation02.png" width="800" style="margin-left: 0">  
<img src="./b_perceptron/images/backpropagation03.png" width="500" style="margin-left: 0">

### Activation Function, 활성화 함수

-   인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.

---

1. 시그모이드 함수
    - 은닉층이 아닌 최종 활성화 함수 즉, 출력층에서 사용된다.
    - 은닉층에서 사용 시, 입력 값이 양의 방향으로 큰 값일 경우 출력 값의 변화가 없으며, 음의 방향도 마찬가지이다.
    - 평균이 0이 아니기 때문에 정규 분포 형태가 아니고, 이는 방향에 따라 기울기가 달라져서 탐색 경로가 비효율적(지그재그)이 된다.

<img src="./b_perceptron/images/sigmoid.png" width="500px">
   
2. 소프트맥스 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 시그모이드와 유사하게 0 ~ 1 사이의 값을 출력하지만, 이진 분류가 아닌 다중 분류를 통해 모든 확률값이 1이 되도록 해준다.
    - 여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.
  
<img src="./b_perceptron/images/softmax.png" width="450px">
 
3. 탄젠트 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 은닉층에서 사용 시, 시그모이드와 달리 -1 ~ 1 사이의 값을 출력해서 평균이 0이 될 수 있지만,  
      여전히 입력 값의 양의 방향으로 큰 값일 경우 출력 값의 변화가 미비하고 음의 방향도 마찬가지이다.

<img src="./b_perceptron/images/tanh.png" width="650px">

4. 렐루 함수
    - 대표적인 은닉층의 활성 함수이다.
    - 입력 값이 0보다 작으면 출려은 0, 0보다 크면 입력 값을 출력하게 된다.

<img src="./b_perceptron/images/relu.png" width="480px">

### Cross Entropy

-   실제 데이터의 확률 분포와, 학습된 모델이 계산한 확률 분포의 차이를 구하는 데 사용된다.
-   분류 문제에서 원-핫 인코딩을 통해 사용할 수 있는 오차 계산법이다.

<img src="./b_perceptron/images/cross_entropy01.png" width="350" style="margin:20px; margin-left: 10px">
<img src="./b_perceptron/images/cross_entropy02.png" width="700" style="margin-left:0">
 
### Optimizer, 최적화

-   최적의 경사 하강법을 적용하기 위해 필요하며, 최솟값을 찾아가는 방법들을 의미한다.
-   loss를 줄이는 방향으로 최소 loss에 보다 빠르고 안정적으로 수렴할 수 있어야 한다.

<img src="./b_perceptron/images/optimizer.png" width="650px" style="margin-left: 10px;">

#### Momentum

-   가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.
-   지역 최솟값에서 벗어나지 못하는 문제를 해결할 수 있으며, 진행했던 방향만큼 추가적으로 더해주어, 관성처럼 빠져나올 수 있게 해준다.

<img src="./b_perceptron/images/momentum.png" width="600px">

#### AdaGrad (Adaptive Gradient)

-   가중치 별로 서로 다른 학습률을 동적으로 적용한다.
-   적게 변화된 가중치는 보다 큰 학습률을 적용하고, 많이 변화된 가중치는 보다 작은 학습률을 적용시킨다.
-   처음에는 큰 보폭으로 이동하다가 최솟값에 가까워질 수록 작은 보폭으로 이동하게 된다.
-   과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.

<div style="display: flex">
    <div>
        <img src="./b_perceptron/images/adagrad02.png" width="100" style="margin-top: 20px; margin-left: 20px;">
    </div>
    <div>
        <img src="./b_perceptron/images/adagrad01.png" width="400" style="margin-top: 20px; margin-left: 80px">
    </div>
</div>

#### RMSProp (Root Mean Square Propagation)

-   AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(exponentially weighted average)을 통해 구한다.
-   지수 가중 평균법이란, 데이터의 이동 평균을 구할 때 오래된 데이터가 미치는 영향을 지수적으로 감쇠하도록 하는 방법이다.
-   이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.
-   feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.

#### Adam (Adaptive Moment Estimation)

-   Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.
-   최적화 방법 중에서 가장 많이 사용되는 알고리즘이며, 수식은 아래와 같다.

<div style="display: flex">
    <div>
        <img src="./b_perceptron/images/adam01.png" width="300" style="margin-top: 20px; margin-left: 0">
    </div>
    <div>
        <img src="./b_perceptron/images/adam02.png" width="200" style="margin-top: 20px; margin-left: 80px">
    </div>
</div>

### Tensorflow, 텐서플로우

-   구글이 개발한 오픈소스 소프트웨어 라이브러리이며, 머신러닝과 딥러닝을 쉽게 사용할 수 있도록 다양한 기능을 제공한다.
-   주로 이미지 인식이나 반복 신경망 구성, 기계 번역, 필기 숫자 판별 등을 위한 각종 신경망 학습에 사용된다.
-   딥러닝 모델을 만들 때, 기초부터 세세하게 작업해야 하기 때문에 진입장벽이 높다.

<img src="./c_tensorflow/images/tensorflow.png" width="600" style="margin-left: 20px">

### Keras, 케라스

-   일반 사용 사례에 **최적화되어 있고 간단하며 일관되고 단순화된** 인터페이스를 제공한다.
-   손쉽게 딥러닝 모델을 개발하고 활용할 수 있도록 직관적인 API를 제공한다.
-   텐서플로우 2버전 이상부터 케라스가 포함되었기 때문에 텐서플로우를 통해 케라스를 사용한다.
-   기존 Keras 패키지보다는 이제 Tensorflow에 내장된 Keras 사용이 더 권장된다.

<img src="./c_tensorflow/images/keras.png" width="600" style="margin-left: -20px">

### Grayscale, RGB

-   흑백 이미지와 컬러 이미지는 각 2차원과 3차원으로 표현될 수 있다.
-   흑백 이미지는 0 ~ 255를 갖는 2차원 배열(높이 X 너비)이고,  
    컬러 이미지는 0 ~ 255를 갖는 R, G, B 2차원 배열 3개를 갖는 3차원 배열(높이 X 너비 X 채널)이다.

<div style="display: flex; margin-top:20px;">
    <div>
        <img src="./c_tensorflow/images/grayscale.png" width="300" style="margin-left: -20px">
    </div>
    <div>
        <img src="./c_tensorflow/images/rgb.png" width="280" style="margin-left: 50px">
    </div>
</div>

### Grayscale Image Matrix

-   검은색에 가까운 색은 0에 가깝고 흰색에 가까우면 255에 가깝다.
-   모든 픽셀이 feature이다.

<img src="./c_tensorflow/images/matrix.png" width="500" style="margin-top: 20px; margin-left; 0">

### Sequential API, Functional API

#### Sequential API

-   간단한 모델을 구현하기에 적합하며 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단하다.
-   단일 입력 및 출력만 있으므로 레이어를 공유하거나 여러 입력 또는 출력을 가질 수 있는 모델을 생성할 수 없다.

#### Functional API

-   Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.
-   여러 개의 입력 및 출력을 가진 모델을 구현하거나 층 간의 연결 및 연산을 수행하는 모델 구현 시 사용한다.

#### 성능 평가

<img src="./c_tensorflow/images/validation.png" width="700" style="margin-left: 0">

### Callback API

-   모델이 학습 중에 충돌이 발생하거나 네트워크가 끊기면, 모든 훈련 시간이 낭비될 수 있고,  
    과적합을 방지하기 위해 훈련을 중간에 중지해야 할 수도 있다.
-   모델이 학습을 시작하면 학습이 완료될 때까지 아무런 제어를 하지 못하게 되고,  
    신경망 훈련을 완료하는 데에는 몇 시간 또는 며칠이 걸릴 수 있기 때문에 모델을 모니터링하고 제어할 수 있는 기능이 필요하다.
-   훈련 시(`fit()`) `Callback API`를 등록시키면 반복 내에서 특정 이벤트 발생마다 등록된 `callback`이 호출되어 수행된다.

**ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weight_only=False, mode='auto')**

-   특정 조건에 따라 모델 또는 가중치를 파일로 저장한다.
-   filepath: "weights.{epoch:03d}-{val_loss:.4f}-{acc:.4f}.weights.hdf5"와 같이 모델의 체크포인트를 저장한다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   save_best_only: 가장 좋은 성능을 나타내는 모델을 저장할지에 대한 여부
-   save_weights_only: weights만 저장할지에 대한 여부
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto

**ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',min_lr=0)**

-   특정 반복동안 성능이 개선되지 않을 때, 학습률을 동적으로 감소시킨다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   factor: 학습률을 감소시킬 비유, 새로운 학습률 = 기존 학습률 \* factor
-   patience: 학습률을 줄이기 전에 monitor할 반복 횟수
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto

**EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')**

-   특정 반복동안 성능이 개선되지 않을 때, 학습을 조기에 중단한다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   patience: Early Stopping을 적용하기 전에 monitor할 반복 횟수
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto

### CNN (Convolutional Neural Network), 합성곱 신경망

-   실제 이미지 데이터는 분류 대상이 이미지에서 고정된 위치에 있지 않은 경우가 대부분이다.
-   실제 이미지 데이터를 분류하기 위해서는, 이미지의 각 feature들을 그대로 학습하는 것이 아닌, CNN으로 패턴을 인식한 뒤 학습해야 한다.

<div style="display: flex; width:70%; margin-bottom: 30px;">
    <div>
        <img src="./d_cnn/images/dogs01.png" width="500" style="margin-left: 20px">
    </div>
    <div>
        <img src="./d_cnn/images/dogs02.png" width="720" style="margin-left: 80px">
    </div>
</div>

-   이미지의 크기가 커질 수록 굉장히 많은 weight가 필요하기 때문에 분류기에 바로 넣지 않고, 이를 사전에 추출 및 축소해야 한다.
-   CNN은 인간의 시신경 구조를 모방한 기술로서, 이미지의 패턴을 찾을 때 사용한다.
-   Feature Extraction을 통해 각 단계를 거치면서, 함축된 이미지 조각으로 분리되고 각 이미지 조각을 통해 이미지의 패턴을 인식한다.

<img src="./d_cnn/images/cnn01.png" width="700" style="margin-left: 0; margin-bottom: 20px">

-   CNN은 분류하기에 적합한 최적의 feature를 추출하고, 최적의 feature를 추출하기 위한 최적의 Weight와 Filter를 계산한다.

<img src="./d_cnn/images/cnn02.png" width="500" style="margin-left: 50px">

---

#### Filter

-   보통 정방 행렬로 구성되어 있고, 원본 이미지에 슬라이딩 윈도우 알고리즘을 사용하여 순차적으로 새로운 픽셀값을 만들면서 적용한다.
-   사용자가 목적에 맞는 특정 필터를 만들거나 기존에 설계된 다양한 필터를 선택하여 이미지에 적용한다.  
    하지만, CNN은 최적의 필터값을 학습하여 스스로 최적화한다.

<img src="./d_cnn/images/filter.gif" width="400" style="margin-left: 0px; margin-top: -30px; margin-bottom: -50px">
<img src="./d_cnn/images/filter.png" width="500" style="margin-left: 0;">

-   필터 하나 당, 이미지의 채널 수만큼 Kernel이 존재하고, 각 채널에 할당된 필터의 커널을 적용하여 출력 이미지를 생성한다.
-   출력 feature map의 개수는 필터의 개수와 동일하다.

<img src="./d_cnn/images/filter_channel.gif" width="500">

#### Kernel

-   filter 안에 1 ~ n 개의 커널이 존재한다. 커널의 개수는 반드시 이미지의 채널 수와 동일해야 한다.
-   Kernel Size는 가로 X 세로를 의미하며, 가로와 세로는 서로 다를 수 있지만 보통은 일치시킨다.
-   Kernel Size가 크면 클 수록 입력한 이미지에서 더 많은 feature 정보를 가져올 수 있지만,  
    큰 사이즈의 Kernel로 Convolution Backbone 연산을 할 경우 훨씬 더 많은 연산량과 파라미터가 필요하다.

<img src="./d_cnn/images/kernel.gif" width="500">

#### Stride

-   입력 이미지에 Convolution Filter를 적용할 때 Sliding Window가 이동하는 간격을 의미한다.
-   기본 stride는 1이지만, 2를 적용하면 입력 feature map 대비 출력 feature map의 크기가 절반 정도로 줄어든다.
-   stride를 키우면 feature 정보를 손실할 가능성이 높아지지만, 오히려 불필요한 특성을 제거하는 효과를 가져올 수 있고 Convolution 연산 속도를 향상시킨다.

<div style="display: flex; width:70%; margin-top: 10px;">
    <div>
        <img src="./d_cnn/images/stride01.gif" width="600" style="margin-left: 0; margin-top: 0">
    </div>
    <div>
        <img src="./d_cnn/images/stride02.gif" width="600" style="margin-left: 50px">
    </div>
</div>

#### Padding

-   Filter를 적용하여 Convolution 수행 시 출력 feature map이 입력 feature map 대비 계속해서 작아지는 것을 막기 위해 사용한다.
-   Filter 적용 전, 입력 feature map의 상하좌우 끝에 각각 열과 행을 추가한 뒤, 0으로 채워서 크기를 증가시킨다.
-   출력 이미지의 크기를 입력 이미지의 크기와 동일하게 유지하기 위해서 직접 계산할 필요 없이 "same"이라는 값을 통해 입력 이미지의 크기와 동일하게 맞출 수 있다.

<img src="./d_cnn/images/padding.gif" width="600" style="margin-left: 0">

#### Pooling

-   Convolution이 적용된 feature map의 일정 영역별로 하나의 값을 추출하여 feature map의 사이즈를 줄인다.
-   보통은 Convolution -> Relu activation -> Pooling 순서로 적용한다.
-   비슷한 feature들이 서로 다른 이미지에서 위치가 달라지면서 다르게 해석되는 현상을 중화시킬 수 있고,  
    feature map의 크기가 줄어들기 때문에 연산 성능이 향상된다.
-   Max Pooling과 Average Pooling이 있으며, Max Pooling은 중요도가 가장 높은 feature를 추출하고, Average Pooling은 전체를 버무려서 추출한다.

<img src="./d_cnn/images/pooling.gif" width="450" style="margin-top: 20px; margin-bottom: 30px">

#### 🚩정리

-   Stride를 증가시키는 것과 Pooling을 적용하는 것은 출력 feature map의 크기를 줄이는 데 사용하는 것이다.
-   Convolution 연산을 진행하면서, feature map의 크기를 줄이면 위치 변화에 따른 feature의 영향도도 줄어들기 때문에 과적합을 방지할 수 있는 장점이 있다.
-   Pooling의 경우 특정 위치의 feature 값이 손실되는 이슈 등으로 인하여 최근 Advanced CNN에서는 많이 사용되지 않는다.
-   Classifier에서는 Fully Connected Layer의 지나친 연결로 인해 많은 파라미터가 생성되므로 오히려 과적합이 발생할 수 있다.

<img src="./d_cnn/images/cnn03.png" width="850px">

-   Dropout을 사용해서 Layer간 연결을 줄일 수 있으며 과적합을 방지할 수 있다.

<img src="./d_cnn/images/dropout.png">
