---
title: "스파크와 분산 GPU 시스템에서 심층 신경망 학습하기"
layout: kr-default
redirect_from: /kr-spark-gpus
---

# 스파크와 분산 GPU 시스템에서 심층 신경망 학습하기

Deeplearning4j는 스파크(Spark)와 분산 GPU 시스템을 이용해 심층 신경망을 학습할 수 있습니다. 여기에서는 스파크를 이용해 데이터를 불러오고 여러 개의 GPU와 cuDNN 라이브러리를 이용해 이미지를 처리하는 과정을 소개합니다.

Deeplearning4j는 인공 신경망을 자동으로 튜닝, 배포, 시각화 및 다른 데이터와 통합하는 라이브러리를 제공하며 이를 통해 분산 구조 서비스에 인공 신경망을 쉽게 적용할 수 있습니다.

이 페이지에서는 우선 우리가 사용하는 다양한 기술을 간략히 소개합니다. 그 뒤에 나오는 이미지 분류 예제를 보시면 먼저 소개한 기술을 구체적으로 어떻게 활용하는지 알 수 있습니다.

우선 아래의 네 가지 기술을 소개하겠습니다.

1. 아파치 스파크
2. CUDA
3. cuDNN
4. DL4J 생태계 (Deeplearning4j, ND4J, DataVec, JavaCPP)

![Alt text](./img/dl4j-diagram.png)

## 아파치 스파크

아파치 스파크는 분산 구조의 데이터 처리 프레임워크입니다. Deeplearning4j의 실제 연산 과정은 스파크보다 빠른 속도와 용량을 필요로 하기 때문에 우리는 스파크를 데이터에 접근하기 위한 인터페이스로만 사용합니다. 간단히 이야기하면 스파크는 고속 ETL (Extract, transform, load/추출, 변환, 적재) 작업을 수행하며 이를 통해 하둡 생태계(HDFS)를 이용할 수 있게 해줍니다. 결과적으로 이를 이용하면 하둡의 장점인 데이터 분산 처리와 일반적인 컴퓨팅의 빠른 연산 속도를 활용할 수 있습니다.

이 과정에서 스파크는 스파크의 자료구조인 RDD(Resilient Distributed Dataset)을 이용합니다. RDD는 여러 클러스터에 저장된 데이터에 접근하는 인터페이스를 제공해줍니다. 아래에서 RDD를 이용해 데이터를 불러오는 과정을 보여드리겠습니다. 이 데이터는 특징값 행렬과 이에 해당하는 라벨을 포함합니다.

## CUDA

CUDA는 NVIDIA에서 제공하는 연산 병렬화 플랫폼입니다. CUDA를 통해 GPU의 C, C++, 포트란으로 된 API를 이용할 수 있습니다. Deeplearning4j는 CUDA 커널과 자바 인터페이스를 이용해 GPU 연산을 수행합니다.

## cuDNN

cuDNN은 CUDA Deep Neural Network Library의 약자로 NVIDIA에서 제공하는 심층신경망 라이브러리입니다. cuDNN은 CUDA보다 상위 수준의 라이브러리로 다양한 심층신경망 연산(컨볼루션, 풀링, 정규화, 활성함수)을 제공합니다.

cuDNN은 컨볼루션 신경망과 리커런트 신경망 등 다양한 인공 신경망 구조에 최적화되어 있습니다. [이미지 프로세싱 벤치마크](https://github.com/soumith/convnet-benchmarks)를 보면 cuDNN의 연산 속도가 상위권에 있는 것을 확인할 수 있습니다. Deeplearning4j는 자바 이용자가 쉽게 cuDNN을 이용할 수 있도록 cuDNN의 자바 인터페이스를 구현했습니다.

## Deeplearning4j, ND4J, DataVec, JavaCPP

[Deeplearning4j](http://deeplearning4j.org/)는 자바, 스칼라, 클로저를 포함한 JVM 기반 심층 신경망 프레임워크입니다. 우리의 목표는 심층 학습을 쉽게 상용화할 수 있도록 하둡이나 스파크같은 빅데이터 프레임워크와 통합하는 것입니다. DL4J는 이미지, 텍스트, 시계열 데이터 등 다양한 형식의 데이터를 처리할 수 있습니다. 데이터 처리는 DL4J에서 제공하는 컨볼루션 신경망, 리커런트 신경망, 자연어 처리 도구 (Word2Vec, Doc2Vec), 오코인코더 등 다양한 인공 신경망을 활용합니다.

아래의 라이브러리는 Deeplearning4j가 인공 신경망 알고리듬을 구현하는데 이용하는 도구로 모두 스카이마인드의 엔지니어들이 관리하는 라이브러리입니다.

* [ND4J](http://nd4j.org/): 선형 대수 및 미적분 라이브러리로 신경망 학습에 필수적인 연산을 제공합니다.
* [libnd4j](https://github.com/deeplearning4j/libnd4j): C++로 구현한 ND4J 가속화 라이브러리입니다.
* [DataVec](https://github.com/deeplearning4j/DataVec): 데이터를 벡터화하는데 사용하는 라이브러리입니다.
* [JavaCPP](https://github.com/bytedeco/javacpp): 자바와 C++을 이어주는 라이브러리입니다. DL4J는 이 JavaCPP를 이용해 cuDNN을 사용합니다.



## 스파크와 DL4J

Deeplearning4j는 스파크와 통합되어있어 쉽게 인공 신경망을 분산 학습할 수 있습니다. 여기에서는 데이터를 병렬화하여 여러 대의 컴퓨터에 있는 GPU에서 각각 학습을 진행합니다. 그리고 데이터 접근은 스파크를 이용합니다. 즉, 스파크의 RDD 파티션에서 각각 학습을 진행합니다. 

이렇게 분산 파일 시스템을 이용해 각 데이터가 있는 노드에서 연산을 하는 것이 분산 학습 방법중에서 가장 쉽습니다.



# 자바와 C++: 자바를 쓰면 CUDA 연산 속도가 느려지지 않나요?

일반적으로는 그 말이 맞습니다. 그래서 우리는 off-heap 메모리를 사용하고 있습니다. JavaCPP에는 `Pointer` 클래스가 구현되어 있는데, 이를 이용하면 가비지 컬렉터와 빠른 속도의 장점을 유연하게 사용할 수 있습니다. 아파치 Flink, 스파크, Hbase 등의 분산 시스템은 모두 이런 방법을 이용합니다.

자바의 선형 대수/행렬 연산 속도는 그렇게 빠르지 않습니다. 이를 해결하기 위해 libnd4j를 사용합니다. 결과적으로 C++을 사용해 부동소수점 연산을 가속화하고, 이 과정에서 libnd4j를 이용한다고 보면 됩니다.

## 분산 심층 학습과 매개변수

신경망 분산 학습에는 두 가지 종류가 있습니다. 데이터 병렬화와 모델 병렬화입니다.

데이터 병렬화는 거대한 데이터 셋을 작은 셋으로 나눈 뒤 각각을 다른 하드웨어에서 동시에 학습하는 것 입니다. 

예를 들면 백과사전을 10페이지씩 나누고 각각을 다른 모델로 보내 학습하는 셈 입니다. 그리고 각 모델에서 학습한 매개변수의 평균 값을 이용해 통합 모델의 매개변수를 결정합니다. Deeplearning4j에서도 이 데이터 병렬화를 이용합니다. 

데이터 병렬화를 시각화하면 아래와 같습니다.

![Alt text](./img/mapreduce_v_iterative.png)



## 매개 변수의 평균값을 활용
데이터 병렬화의 결과로 매개 변수의 평균값을 사용하는 방법과 관련해 페이지 하단에 참고 문헌을 정리해 놓았으니 참고하시기 바랍니다.

## 예제 코드

아래 코드는 스파크와 여러 대의 GPU로 LeNet을 학습하는 코드입니다.

우선 스파크를 준비하고 데이터를 불러옵니다.

    public static void main(String[] args) throws Exception {

        //Create spark context, and load data into memory
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("MNIST");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        int examplesPerDataSetObject = 32;
        DataSetIterator mnistTrain = new MnistDataSetIterator(32, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(32, false, 12345);
        List<DataSet> trainData = new ArrayList<>();
        List<DataSet> testData = new ArrayList<>();
        while(mnistTrain.hasNext()) trainData.add(mnistTrain.next());
        Collections.shuffle(trainData,new Random(12345));
        while(mnistTest.hasNext()) testData.add(mnistTest.next());

        //Get training data. Note that using parallelize isn't recommended for real problems
        JavaRDD<DataSet> train = sc.parallelize(trainData);
        JavaRDD<DataSet> test = sc.parallelize(testData);

이제 신경망을 구성합니다.

        //Set up network configuration (as per standard DL4J networks)
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 1;
        int seed = 123;

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .nOut(50)
                        .stride(2,2)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .nOut(200).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

아래의 블록도를 참고하십시오.
![Alt text](./img/flow2.png)


참고로, [Computation Graph API](http://deeplearning4j.org/compgraph)를 이용하면 더욱 복잡한 구조의 신경망을 사용할 할 수 있습니다.

위의 코드에서는 빌더 패턴을 이용하고 있습니다. 키워드 매개 변수가 없는 자바의 특성상 빌더 패턴을 이용하는 것이 클로저/스칼라와 통합된 API를 설계하기에 편리합니다.

위의 구성은 YAML이나 JSON을 이용해 설정 가능합니다.

마지막으로, DL4J는 곧 케라스(Keras) API와 유사한 스칼라 API를 공개할 예정입니다.




## 스파크를 이용한 분산 학습

이제 스파크를 설정해주면 됩니다. 얼마나 자주 매개변수를 갱신할지, 각 머신에서 데이터를 어떻게 사용할지를 정해줍니다.

        //Create Spark multi layer network from configuration
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true) //save things like adagrad squared gradient histories
                .averagingFrequency(5) //Do 5 minibatch fit operations per worker, then average and redistribute parameters
                .batchSizePerWorker(examplesPerDataSetObject) //Number of examples that each worker uses per fit operation
                .build();

        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm);

이제 `.fit()`으로 신경망을 학습합니다.




        //Train network
        log.info("--- Starting network training ---");
        int nEpochs = 5;
        for( int i = 0; i < nEpochs; i++ ){
            sparkNetwork.fit(train);
            System.out.println("----- Epoch " + i + " complete -----");

            //Evaluate using Spark:
            Evaluation evaluation = sparkNetwork.evaluate(test);
            System.out.println(evaluation.stats());
        }

## 스파크를 이용한 데이터 병렬화

지금까지 가장 기본적인 내용을 알아봤습니다.

실제로 심층 신경망을 분산GPU와 스파크로 학습하려면 다음의 두 가지 설정을 해줘야합니다.

1. [퀵 스타트 가이드](http://deeplearning4j.org/kr-quickstart)에 나온 DL4J 기본 설정
2. [스파크 안내](http://deeplearning4j.org/spark)의 스파크 설정 및 코드 예제 참고

더 자세한 질문이 있다면 [DL4J gitter 대화방](https://gitter.im/deeplearning4j/deeplearning4j)을 이용하시길 바랍니다.



## 참고 자료

[1] Training with intra-block parallel optimization and blockwise model-update filtering. In 2016

IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages

5880–5884. IEEE, 2016.

[2] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior,

Paul Tucker, Ke Yang, Quoc V Le, et al. Large scale distributed deep networks. In Advances in

Neural Information Processing Systems, pages 1223–1231, 2012.

[3] Augustus Odena. Faster asynchronous sgd. arXiv preprint arXiv:1601.04033, 2016.

[4] Nikko Strom. Scalable distributed dnn training using commodity gpu cloud computing. In Six-
teenth Annual Conference of the International Speech Communication Association, 2015. http:

//nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf.

[5] Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-sgd for distributed

deep learning. CoRR, abs/1511.05950, 2015. http://arxiv.org/abs/1511.05950.

[6]: http://arxiv.org/abs/1404.5997
