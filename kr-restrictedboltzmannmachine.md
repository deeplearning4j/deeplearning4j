---
title: 제한된 볼츠만 머신(Restricted Boltzmann Machines)을 위한 초보자용 튜토리얼
layout: kr-default
---

# 제한된 볼츠만 머신(Restricted Boltzmann Machines)를 위한 초보자용 튜토리얼

내용

* <a href="#define">정의 & 구조</a>
* <a href="#reconstruct">재구성</a>
* <a href="#probability">확률 분포</a>
* <a href="#code">코드 샘플: DL4J로 RBM 상에서 Iris 초기화 하기</a>
* <a href="#params">파라미터 & k</a>
* <a href="#CRBM">연속 RBMs</a>
* <a href="#next">다음 단계</a>
* <a href="#resource">다른 리소스들</a>

## <a name="define">정의 & 구조</a>

Geoff Hinton에 의해 발명된, 제한된 볼츠만 머신(Restricted Boltzmann machine)은 차원 감소, 분류, [회귀 분석](../linear-regression.html), 협업 필터링, 속성 학습 및 주제 모델링에 유용한 알고리즘 입니다. (RBMs과 같은 [신경망](../neuralnet-overview.html)이 이용될 수 있는 방법의 보다 구체적인 예제들을 위해서는 저희의 페이지 [이용 사례들](../use_cases.html)을 참조하십시오).

그들의 상대적인 단순성으로 인해 제한된 볼츠만 머신이 저희가 시도할 첫번째 신경망 입니다. 아래의 단락에서 저희는 어떻게 그들이 작동하는지를 다이어그램과 쉬운 언어로 설명할 것 입니다. 

RBMs은 *deep-belief networks*의 빌딩 블럭을 구성하는 얕은 두 레이어의 신경망 입니다. RBM의 첫번째 레이어는 보여지는(visible), 또는 입력, 레이어로 불리고, 두번째는 숨겨진(hidden) 레이어 입니다. 

![Alt text](../img/two_layer_RBM.png)

상기 그래프에서 각각의 원은 *노드*로 불리는 뉴런과 같은 객체를 나타내고, 노드들은 단지 계산이 수행되는 장소 입니다. 그 노드들은 레이어를 통해 서로에게 접속되어 있으나 동일한 레이어의 어떠한 두개의 노드도 연결되어 있지 않습니다.

즉, 인트라-레이어 커뮤니케이션은 없습니다 – 이것이 제한된 볼츠만 머신에서의 *제한* 입니다. 각각의 노드는 입력을 수행하는 계산의 현장이며, 입력을 송신할 지의 여부에 대한 [stochastic](../glossary.html#stochasticgradientdescent) 결정을 함으로써 시작합니다. (*Stochastic*는 “임의로 결정된다는” 것을 의미하고, 이 경우, 입력을 수정하는 상대 계수는 임의로 초기화 됩니다.)

각각의 보여지는 노드는 학습될 데이터 세트의 한 항목에서 낮은 수준의 속성을 취합니다. 예를 들어, 회색 스케일 이미지의 한 데이터 세트로부터 각각의 보여지는 노드는 하나의 이미지에서 각각의 픽셀을 위한 하나의 픽셀-가치를 수신할 것 입니다. (MNIST 이미지는 784개의 픽셀을 가지므로, 그들을 처리하는 신경망은 보여지는 레이어 상에서 반드시 784개의 입력 노드를 가져야 합니다.)

이제 두개의 레이어 망을 통해, 단일 픽셀 값, *x*를 수행해보도록 하겠습니다. 숨겨진 레이어의 노드 1 에서, x는 *가중치*에 의해 곱해지고, *바이어스*에 추가됩니다. 그 두 수행의 결과는 주어진 입력 x로 그 노드의 출력, 또는 그것을 통과하는 신호의 강도를 제공하는 *활성화 함수*에 공급됩니다.

		activation f((weight w * input x) + bias b ) = output a

![Alt text](../img/input_path_RBM.png)

다음으로, 몇가지 입력이 하나의 숨겨진 노드에서 결합하는 방법에 대해 살펴보겠습니다. 각각의 x는 별도의 가중치에 의해 곱해지고, 그 곱들은 더해지고, 바이어스에 추가되고, 그 결과는 그 노드의 출력을 생성하기 위해 활성화 함수를 통해 전달됩니다.

![Alt text](../img/weighted_input_RBM.png)

모든 보여지는 노드들로부터의 입력이 모든 숨겨진 노드들로 옮겨지기 때문에, 한 RBM은 *symmetrical bipartite graph*로서 정의될 수 있습니다.

*Symmetrical*은 각각의 보여지는 노드가 각각의 숨겨진 노드와 연결되어 있다는 것을 의미합니다 (아래를 보십시오). *Bipartite*는 그것이 두개의 부분, 혹은 레이어를 가지고 있다는 것을 의미하고, *graph*는 노드의 망을 위한 수학적 용어 입니다.

각각의 숨겨진 노드에서 각 입력 x는 그의 해당하는 가중치 w에 의해 곱해집니다. 즉, 단일 입력 x는 여기에서 모두 합쳐 12개의 가중치를 만드는 세개의 가중치를 가질 것 입니다 (4개의 입력 노드 x 3개의 숨겨진 노드). 두 레이어들 사이의 가중치는 항상 열(row)들이 입력 노드들과 동일하고, 줄(column)들이 출력 노드들과 동일한 곳인 매트릭스를 형성할 것 입니다.

각각의 숨겨진 노드는 그들의 해당하는 가중치에 의해 곱해진 네개의 입력을 수신합니다. 그 곱의 합은 다시 (최소한 일부의 활성화가 발생하도록 강제할) 바이어스로 더해지고, 그 결과는 각각의 숨겨진 노드를 위한 하나의 출력을 제공하기 위해 활성 알고리즘을 통해 전달됩니다.

![Alt text](../img/multiple_inputs_RBM.png)

이 두 레이어들이 deeper neural network의 일부라면, 숨겨진 레이어 1의 출력은 숨겨진 레이어 2로 입력으로서, 그들이 최종의 분류 레이어에 도달할 때까지 거기에서부터 여러분께서 원하시는 만큼의 숨겨진 레이어까지 도달할 것 입니다. (단순 피드-포워드 동작(simple feed-forward movements)을 위해서, RBM 노드들은 단지 *오토인코더(autoencoder)*로서 기능합니다.)

![Alt text](../img/multiple_hidden_layers_RBM.png)

## <a name="reconstructions">재구성</a>

하지만 제한된 볼츠만 머신으로의 이 소개에서는, 저희는 그들이 deeper network의 개입 없이, 보여지는 레이어와 숨겨진 레이어 1 사이에서 몇몇의 포워드 및 백워드 패스들을 생성하는 자율 방식으로 스스로 데이터를 재구성하는 학습 방법에 초점을 맞출 것 입니다 (자율은 테스트 세트에서 ground-truth 레이블이 없다는 것을 의미합니다).

재구성 단계에서, 숨겨진 레이어 1의 활성화는 백워드 패스에서의 입력 입니다. x가 포워드 패스 상에서 가중치-조정된 것과 같이, 그들은 동일한 가중치에 의해, 인터노드 엣지 당 하나씩 곱해집니다. 그 곱들의 합은 각 보여지는 노드에서 보여지는-레이어 바이어스로 추가되고, 그 운영의 출력이 재구성 입니다; 즉, 오리지널 입력의 근사치. 이것은 다음의 다이어그램에 의해 표현될 수 있습니다:

![Alt text](../img/reconstruction_RBM.png)

RBM의 가중치가 임의로 초기화 되기 때문에, 재구성과 오리지널 입력 사이의 차이는 종종 큽니다. 여러분은 `r`의 값과 입력 값 사이의 차이로서 재구성 에러를 생각하실 수 있고, 반복적인 학습 과정에서 에러 최소한에 도달할 때 까지 다시 또 다시 입력 값, 값, 그 에러는 RBM의 가중치에 반해 backpropagated 됩니다. 

Backpropagation의 보다 자세한 설명은 [여기](../neuralnet-overview.html#forward)에 있습니다. 

여러분께서 보시는 바와 같이, RBM은 그의 포워드 패스 상에서 노드 활성화, 또는 [x 가중치가 주어졌을 때 출력의 확률](https://en.wikipedia.org/wiki/Bayes%27_theorem)에 대한 추정을 위해 입력을 사용합니다: `p(a|x; w)`. 

그러나 그것의 백워드 패스 상에서, 활성화가 공급되고, 재구성이, 혹은 오리지널 데이터에 대한 추측들이, 배출되면, RBM은 포워드 패스 상에서 사용된 것들과 같은 상관계수들로 가중화된 입력 `x`와 주어진 활성화 `a`의 확률을 추정하려고 시도합니다. 이 두번째 단계는 `p(x|a; w)`로서 표현될 수 있습니다. 

그 두 추정치는 함께 여러분을 입력 *x*와 활성화 *a*의 결합 확률 분포, 또는 `p(x, a)`로 안내할 것 입니다. 

재구성은 많은 입력에 기초한 연속적인 값을 추정하는 회귀 분석과 다르고,  어떤 개별 레이블을 주어진 입력 예제에 적용할 지에 대한 예상을 하는 분류와 다른 작업을 합니다. 

재구성은 오리지널 입력의 확률 분포에 대한 추정을 합니다; 예를 들어, 동시에 많은 다양한 포인트들의 값들. 이는 입력을 레이블에 매핑하는 분류에 의해 수행되는, 그룹의 데이터 포인트들 사이에서 선들을 효과적으로 긋는 discriminative learning으로부터 반드시 구분되어야 하는 [generative learning](http://cs229.stanford.edu/notes/cs229-notes2.pdf)로 알려져 있습니다.

입력 데이터와 그 재구성 둘 모두가 부분적으로만 중첩되는 다른 형태의 정규 곡선들이라고 상상해보십시오.

입력의 ground-truth 분포와 추정된 확률 분포 사이의 거리를 측정하기 위해서, RBMs은 [Kullback Leibler Divergence](https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-Divergence)을 사용합니다. 그 수학의 자세한 설명은 [Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 상에서 찾으실 수 있습니다. 

KL-Divergence는 두 곡선 아래의 영역을 비중첩, 또는 발산(diverging)하고, RBM의 최적화 알고리즘이 그 영역들을 최소화 하기를 시도해, 숨겨진 레이어 1의 활성화에 의해 곱해졌을 때 공유된 가중치들이 오리지널 입력의 근사치를 제공합니다. 왼쪽에 오리지널 입력 *p*와 재구성된 분포 *q*로 병치된 한 세트의 확률 분포가 있습니다; 오른쪽에 그들의 차이의 통합이 있습니다.

![Alt text](../img/KL_divergence_RBM.png)

그들이 제공하는 에러에 따라 가중치는 반복적으로 조정되기 때문에, RBM는 오리지널 데이트를 근사화하는 것을 학습합니다. 여러분은 가중치가 서서히 첫번째 숨겨진 레이어의 활성화에서 코드화된 입력의 구조를 반영하게 된다고 할 수 있습니다. 그 학습 과정은 단계적으로 두개의 중첩하는 확률 분포인 것 처럼 보입니다.

![Alt text](../img/KLD_update_RBM.png)

### <a name="probability">확률 분포</a> 

잠시 확률 분포에 대해 얘기해보도록 하겠습니다. 만약 여러분께서 두개의 주사위를 굴리신다면, 모든 결과에 대한 확률 분포는 이와 같을 것 입니다:

![Alt text](https://upload.wikimedia.org/wikipedia/commons/1/12/Dice_Distribution_%28bar%29.svg)

즉, 7s가 가장 가능성이 높고, 주사위 던지기의 결과를 예측하는 어떤 수식도 그것을 고려할 필요가 있습니다. 

언어들은 그들의 문자들의 확률 분포에 구체적입니다. 왜냐하면 각각의 언어는 특정 문자들을 다른 이들보다 더 많이 사용하기 때문 입니다. 아이슬랜드어에서 가장 많이 사용되는 문자는 *a*, *r* 및 *n*인 반면, 영어에서는 문자 *e*, *t* 및 *a*가 가장  많이 사용됩니다. 아이슬랜드어를 영어에 기초한 가중치 세트로 재구성하려는 시도는 큰 차이를 나을 것 입니다. 

동일하게, 이미지 데이터 세트들은 그 세트에서 이미지의 종류에 따라 그들의 픽셀 값들에 대한 독특한 확률 분포를 가집니다. 픽셀 값들은 그 데이터 세트가 MNIST의 필기체 숫자들을 포함하는지의 여부에 따라 다르게 분포됩니다:

![Alt text](../img/mnist_render.png)

또는 Labeled Faces in the Wild에서 보이는 사진들:

![Alt text](../img/LFW_reconstruction.jpg)

잠시 단지 두개의 출력 노드, 각 동물 당 하나씩을 가진 코끼리와 개의 이미지를 공급받은 RBM를 상상해보십시오. 그 RBM이 자신에게 포워드 패스에서 하는 질문은 이것 입니다: 이 픽셀로, 나의 가중치는 코끼리 노드 또는 개 노드로 더 강력한 신호를 보내야 하는가? 그리고 이 RBM이 백워드 패스에서 하는 질문은: 코끼리로, 나는 픽셀의 어떤 분포를 기대해야 하는가? 

그것은 결합 분포 입니다: *a*가 주어진 *x*와 *x*가 주어진 *a*의 동시 확률은 RBM의 두 레이어 사이에서 공유 가중치로서 표시됩니다. 

학습 재구성의 과정은, 한편, 어떤 그룹의 픽셀들이 주어진 세트의 이미지들을 위해 공동 발생하는 경향이 있는지를 배우는 것 입니다. 네트워크에서 숨겨진 레이어들의 노드에 의해 생산된 활성화는 공동 발생을 나타냅니다; 즉 "nonlinear gray tube + big, floppy ears + wrinkles”가 하나일 수 있습니다.

위의 두 이미지에서 여러분은 Deeplearning4j의 RBM 구현에 의해 학습된 재구성을 보실 수 있습니다. 이 재구성들은 오리지널 데이터가 어떻게 생겼는지에 대해 RBM이 무엇을 “생각”하는지를 보여줍니다. Geoff Hinton은 일종의 “꿈꾸는” 머신으로서 이를 언급했습니다. 신경망 학습 동안 렌더링 시, 시각화는  RBM이 실제로 학습 중이라고 자신을 재확신 시키는데 매우 유용한 추론 입니다. 만약 그렇지 않다면, 아래에서 언급된 그의 하이퍼파라미터들은 조정되어야 합니다.

마지막 사항: 여러분께서는 RBMs이 두개의 바이어스를 가지고 있음을 알게 되실 것 입니다. 이는 그들을 다른 오토인코더들로부터 구분시키는 한 측면 입니다. 숨겨진 바이어스는 RBM이 포워드 패스에서 활성화를 생산하는 것을 돕습니다 (바이어스들은 하한선을 부과하기 때문에 아무리 데이터를 적게 해도 최소 몇개의 노드들은 사라집니다), 반면*보여지는* 레이어의 바이어스들은 RBM이 백워드 패스에서 재구성을 학습하게 돕습니다. 

### 여러개의 레이어들

이 RBM이 첫번째 숨겨진 레이어의 활성화에 연관되기 때문에 입력 데이터의 구조를 학습하면 그 데이터는 그 망의 한 레이어 낮은 곳으로 보내집니다. 여러분의 첫번째 숨겨진 레이어가 보여지는 레이어의 역할을 가집니다. 이제 그 활성화는 여러분의 입력이 되고, 그들은 또다른 세트의 활성화를 생성하기 위해 두번째 숨겨진 레이어의 노드들에서 가중치에 의해 곱해집니다. 

그룹화된 속성들에 의해 직력화된 세트의 활성화를 생성하고 속성들의 그룹들을 그룹화 하는 이 과정은 신경망이 데이터의 더 복잡하고 추상적인 표현을 배움으로써 *속성 계층(feature hierarchy)*의 기초가 됩니다. 

각각의 새로운 숨겨진 레이어로, 가중치들은 그 레이어가 이전의 레이어로부터 입력을 근사화할 수 있을때 까지 조정됩니다. 이것은 욕심많은 레이어식의 자율적인 선학습 입니다. 네트워크의 가중치를 개선하기 위해 어떤 레이블을 요구하지 않습니다. 이는 여러분께서 레이블되지 않고, 인간의 손에 의해 작업되지 않은, 세상에 존재하는 엄청난 대부분의 데이터 상에서 학습할 수 있다는 것을 의미 합니다. 하나의 법칙으로서 더 많은 데이터에 노출된 알고리즘은 더 정확한 데이터를 생산합니다. 그리고 이것이 딥 러닝 알고리즘이 잘 나가고 있는 이유들 중 하나 입니다.

그 레이어들이 이미 데이터의 속성들을 근사화하기 때문에, 그들은 두번째 단계에서 여러분께서 후속 지도 학습 단계에서 deep-belief network와 함께 이미지들을 분류하고자 할 때 더 잘 학습하기 위해 제대로 위치되었습니다.

RBMs는 많은 용도들을 가지고 있지만, 후기 학습을 촉진할 가중치의 적절한 초기화는 그들의 주된 장점 중 하나 입니다. 어떤 의미에서 그들은 backpropagation와 유사한 것을 수행합니다: 그들은 데이터를 잘 모델화할 가중치들을 제공합니다. 여러분은 선학습과 backprop은 동일한 끝으로의 치환 수단이라고 할 수 있습니다. 

한 다이어그램에서 제한된 볼츠만 머신을 합성하려면, 여기 symmetrical bipartite와 bidirectional graph가 있습니다:

![Alt text](../img/sym_bipartite_graph_RBM.png)

RBMs의 구조를 더 깊이 공부하는데 관심이 있는 경우, 그들은 [directional, acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph)의 한 유형 입니다.

### <a name="code">코드 샘플: DL4J로 Iris 상에서  RBM 초기화 하기</a>

RBM이 어떻게 단순히 더 많은 일반적인 클래스들로 공급된 파라미터인`NeuralNetConfiguration`에서 한 레이어로서 생성되는지 아래를 참고하십시오. 마찬가지로, RBM 객체는 보여지는 및 숨겨진 레이어들에 각각 적용되는 Gaussian과 Rectified Linear transforms와 같은 properties를 저장하는데 사용됩니다. 

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rbm/RBMIrisExample.java?slice=37:87"></script>

이것은 저희가 다른 튜토리얼에서 다룰 [RBM processing the Iris flower dataset](../iris-flower-dataset-tutorial.html)의 한 예제 입니다. 

## <a name="params">파라미터들 & k</a>

변수 k는 여러분께서 [contrastive divergence](../glossary.html#contrastivedivergence)를 실행하시는 횟수의 수 입니다. Contrastive divergence는 기울기 (네트워크의 가중치와 그의 에러 사이의 관계를 나타내는 경사)를 계산하기 위해 사용된 방식으로, 이것이 없이는 학습이 발생하지 않습니다.

매번 contrastive divergence가 실행할 때 마다, 그것은 제한된 볼츠만 머신을 구성하는 Markov Chain의 샘플이 됩니다. 일반적인 값은 1 입니다. 

위의 예제에서, 여러분께서는 RBMs가 더 일반적인 `MultiLayerConfiguration`로 어떻게 레이어들로 생성될 수 있는지를 볼 수 있습니다. 각각의 점(dot) 뒤에서 여러분은 deep neural net의 구조 및 성능에 영향을 미치는 추가적인 파라미터를 찾으실 것 입니다. 대부분의 그 파라미터들은 이 사이트에 정의되어 있습니다. 

**weightInit**, 또는 `weightInitialization`는 represents the starting value of the coefficients that 각 노드로 들어오는 입력 신호를 증폭하거나 소거하는 상관계수의 시작값을 표현합니다. 적절한 가중치 초기화는 여러분께서 많은 학습 시간을 절약할 수 있게 합니다. 왜냐하면 망을 학습하는 것은 망이 정확하게 분류하도록 하는 최상의 신호들을 전송하는 상관계수들을 조정하는 것에 지나지 않기 때문입니다.

**activationFunction**은 한 신호가 그 노드를 통과하는 각 노드의 위와, 그것이 차단된 아래에서 임계를 결정하는 함수의 세트 중 하나를 의미 합니다. 한 노드가 신호를 통과하는 경우, 그것은 “활성화 되었습니다.”

**optimizationAlgo**는 한 신경망이 그의 상관계수들을 단계별로 조정할 때, 에러를 최소화 하거나 최소의 에러 로커스를 찾는 방식을 의미합니다. 각 문자들이 각각 그 발명자들의 성을 의미하는 약어 LBFGS는 상관계수들이 조정됨에 따라 기울기의 각도를 계산하는 2차 유도체를 사용하는 최적화 알고리즘 입니다.

**regularization** 방식들은, **l2**와 같은, 신경망에서 과한 적용을 막는 것을 돕습니다. Regularization은 정의에 의한 큰 상관계수들이 그 망이 그의 결과들을 몇몇 강하게 가중치가 적용된 입력들에 고정시키는 것을 학습해왔다는 것을 의미하기 때문에 본질적으로 큰 상관계수들을 징계합니다. 지나치게 강한 가중치는 새로운 데이터에 노출될 때 망의 모델을 일반화하는 것을 어렵게 할 수 있습니다.

**VisibleUnit/HiddenUnit**은 한 신경망의 레이어들을 의미합니다. `VisibleUnit`는, 또는 layer, 입력이 들어가는 노드들의 레이어고, `HiddenUnit`는 그 입력들이 더 복잡합 속성들과 재결합하는 레이어 입니다. 둘 모두의 단위는 그들 자신의 말하자면 transforms를 가집니다. 이 경우, visible을 위해서는 Gaussian, hidden을 위해서는 Rectified Linear 입니다. 이는 그들 각각의 레이어들로부터 나온 신호는 새로운 공간으로 매핑합니다. 

**lossFunction**는 여러분께서 에러를 측정하시는 방법, 또는 여러분의 망의 예상들과 그 테스트 세트에 포함된 올바른 레이블들 간의 차이 입니다. 여기에서 저희는 모든 에러를 positive로 만들어 그들이 더해지고 backpropagated 될 수 있도록 하는`SQUARED_ERROR`를 사용합니다.

**learningRate**는, **momentum**과 같은, 신경망이 에러를 위해 수정 시, 각각의 반복 상에서 얼마나 상관계수들을 조정하는지에 영향을 줍니다. 이 두개의 파라미터들은 망이 로컬 최적화를 향해 기울기를 줄이는 단계들의 크기를 결정하는 것을 돕습니다. 큰 learning rate은 망이 빠르게 학습하게 하고, 아마 최적화를 지나치게 겨냥할 것 입니다. 작은 learning rate은 학습을 느리게 해 비효율적 일 수 있습니다. 

### <a name="CRBM">연속 RBMs</a>

연속적인 제한된 볼츠만 머신은 contrastive divergence sampling의 한 다른 유형을 통해 연속적인 입력 (즉, integers보다 더 정밀하게 잘려진 숫자들)을 수용하는 RBM의 한 형태 입니다. 이는 CRBM이 이미지 픽셀 또는 0과 1 사이의 소수로 정규화하는 단어-수 벡터와 같은 것들을 처리할 수 있게 합니다.

deep-learning net의 각 레이어는 네가지 요소들: 입력, 상관계수, 바이어스 및 transform (활성화 알고리즘)을 요구한다는 것을 기억하시기 바랍니다. 

입력은 그 이전의 레이어로부터 (또는 오리지널 데이터로서) 그것에 공급된 숫자적인 데이터, 벡터, 입니다. 상관계수들은 각 노드 레이어를 통과하는 다양한 속성들에 주어진 가중치들 입니다. 바이어스는 한 레이어에서 무슨 일이 있든 일부 노드들이 활성화될 것을 보장합니다. Transformation은 기울기들을 더 쉽게 계산할 수 있게 만드는 한 방법으로 (그리고 기울기들은 망이 학습하는데 필요합니다) 각각의 레이어를 통화한 후 데이터를 부서뜨리는 추가적인 알고리즘 입니다. 

그 추가적인 알고리즘들과 그의 조합들은 레이어마다 다양할 수 있습니다. 

효과적인 제한된 볼츠만 머신은 보여지는 (또는 입력) 레이어 상의 Gaussian transformation과 숨겨진 레이어 상의 rectified-linear-unit transformation을 구축합니다. 그것은 특히 [얼굴 재구성](../facial-reconstruction-tutorial.html)에 유용합니다. 이진법 데이터를 다루는 RBMs은 단순히 두 개의 transformations을 이진법의 것들로 만듭니다. 

Gaussian transformations은 RBMs의 숨겨진 레이어에서 잘 작동하지 않습니다. 대신 사용되는 rectified-linear-unit transformations은 이진법의 transformations보다 더 많은 속성들을 표현할 수 있고, 저희는 이를 [deep-belief nets](../deepbeliefnetwork.html) 상에서 구축합니다.

### <a name="next">결론 & 다음 단계</a>

여러분은 RBMs의 출력 숫자들을 백불율로 해석하실 수 있습니다. 재구성에서 숫자가 *영(0)이 아니라고* 할 때마다, 그것은 RBM이 입력을 학습하는 좋은 지표 입니다. 제한된 볼츠만 머신을  tick하게 만드는 매커니즘에 대한 또다른 관점을 얻으시려면, [여기](../understandingRBMs.html)를 눌러주십시오. 

다음으로, 저희는 여러분께 이는 단순히 서로의 위에 쌓은 많은 제한된 볼츠만 머신들인 [deep-belief network](../deepbeliefnetwork.html)를 구축하는 방법을 보여드리겠습니다.

### <a name="resources">다른 리소스들</a>

* [Geoff Hinton on Boltzmann Machines](http://www.scholarpedia.org/article/Boltzmann_machine)
* [Deeplearning.net의 Restricted Boltzmann Machine 튜토리얼](http://deeplearning.net/tutorial/rbm.html)
* [Restricted Boltzmann Machines 학습으로의 실용적인 가이드](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf); Geoff Hinton

### 다른 초보자 가이드

* [Recurrent Networks/LSTMs](../lstm.html)
* [Neural Networks](../neuralnet-overview.html)
* [Eigenvectors, PCA and Entropy](../eigenvector.html)
* [Neural Networks & Regression](../linear-regression.html)
* [Convolutional Networks](../convolutionalnets.html)
