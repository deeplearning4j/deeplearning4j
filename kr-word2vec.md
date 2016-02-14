---
title: "Word2vec: Java에서 Neural Word Embeddings"
layout: kr-default
---

# Word2Vec

내용

* <a href="#intro">소개</a>
* <a href="#embed">Neural Word Embeddings</a>
* <a href="#crazy">재미있는 Word2vec 결과</a>
* <a href="#just">**코드를 주십시오**</a>
* <a href="#anatomy">Word2Vec 해부학</a>
* <a href="#setup">설정, 로드 및 학습</a>
* <a href="#code">코드 예제</a>
* <a href="#trouble">문제 해결 및 Word2Vec 튜닝하기</a>
* <a href="#use">Word2vec 이용 사례</a>
* <a href="#foreign">외국어</a>
* <a href="#glove">GloVe (Global Vectors) & Doc2Vec</a>

##<a name="intro">Word2Vec 소개</a>

Word2vec는 텍스트를 처리하는 두개의 레이어 입니다. 그것의 입력은 텍스트 코퍼스 (corpus)이고, 그 출력은 벡터들의 집합 입니다: 벡터는 그 코퍼스에서 단어들에 대한 속성 벡터 입니다. Word2vec는 [deep neural network](../neuralnet-overview.html)가 아닌 반면, 텍스트를 딥 망들이 이해할 수 있는 숫자의 형태로 전환 합니다. 

Word2vec의 응용 프로그램들은 wild에서 구문 분석 이상으로 확장 합니다. 그것은 또한 패턴들이 식별될 수 있는 <a href="#sequence">유전자, 코드, 재생 목록, 소셜 미디어 그래프 및 다른 언어적 혹은 상징적인 시리즈</a>에 적용될 수 있습니다. [Deeplearning4j](http://deeplearning4j.org/kr-quickstart.html)는 GPUs와 함께 Spark와 작동하는 Java 및 [Scala](http://deeplearning4j.org/scala.html)를 위한 Word2vec의 배포된 형태로 구현 합니다.

Word2vec의 목적과 유용성은 벡터 공간에서 유사한 단어들을 함께 벡터들로 그룹화 한다는 데에 있습니다. 즉, 그것은 수학적으로 유사성을 검출 합니다. Word2vec은 단어 속성들의 숫자 표현을 배분하는 벡터를 생성 합니다. 속성들은 개별 단어들의 문맥과 같은 것들 입니다. 인간의 개입없이 그렇게 합니다. 

충분한 데이터, 사용 및 문맥을 감안할 때, Word2vec는 과거 모습에 기반하여 한 단어의 의미에 대해 매우 충분한 추측을 할 수 있습니다. 그 추측은 다른 단어들과의 연결을 설정하기 위해 사용될 수 있습니다. (말하자면, "남자"는 "소년"으로, "여자"는 "소녀"로), 또는 문서들을 모으고 그들을 주제별로 분류 합니다. 그 클러스터들은 검색의 과학 연구, 법률 검색, 전자 상거래 및 고객 관계 관리와 같은 다양한 분야에서의 검색, [sentiment analysis](../sentiment_analysis_word2vec.html) 및 추천의 기초를 형성할 수 있습니다. 

Word2vec 신경망의 출력은 각각의 항목이 그것에 부착된 벡터를 가지고 있는 어휘로서, 이는 딥 러닝 망으로 공급되거나 단순히 단어들 사이의 관계를 검출하기 위해 조회될 수 있습니다.

[코사인 유사성](../glossary.html#cosine) 측정 시, 1의 전체 유사성이 0도 각으로 완전한 중복인 반면, 어떤 유사성도 90도 각으로서 표현되지 않습니다; 말하자면, 스웨덴은 스웨덴과 동일한 반면, 노르웨이는 스웨덴으로부터 어떤 다른 나라의 가장 높은 0.760124의 코사인 거리를 가집니다. 

여기에 Word2vec를 사용한 "스웨덴"과 관련된 단어의 목록이 근접성 순서로 있습니다:

![Alt text](../img/sweden_cosine_distance.png) 

스칸디나비아의 국가들과 여러 부유한 북유럽, 독일계 나라들이 상위 9 사이에 있습니다.

##<a name="embed">Neural Word Embeddings</a>

단어들을 표현하기 위해 저희가 사용하는 벡터들은 *neural word embeddings*이라고 불리고, 표현들은 이상합니다. 두가지가 근본적으로 다름에도 불구하고 한가지가 다른 하나를 설명합니다. Elvis Costello가 말한 것과 같이: "음악에 대해 작성하는 것은 건축에 대한 무용과 같다." Word2vec는 단어들에 대해 "벡터화"하고, 그렇게 함으로써 그것이 자연 언어를 컴퓨터-판독 가능한 것으로 만듭니다 -- 저희는 그들의 유사성을 검출하기 위해 단어들에 강력한 수학적인 연산 수행을 시작할 수 있습니다. 

그래서 neural word embeddings은 숫자들과 함께 단어를 표현합니다. 그것은 단순합니다, 그러나 가능성은 낮으나, 번역 입니다. 

Word2vec는 벡터에서 각각의 단어를 코딩하는 오토인코더와 비슷하지만, [제한 볼츠만 머신(restricted Boltzmann machine)](../kr-restrictedboltzmannmachine.html)이 하듯이 [재건축(reconstruction)](../kr-restrictedboltzmannmachine.html#reconstruct)을 통해 입력 단어들에 반해 학습하기 보다는, 입력 코퍼스에서 그것들을 인접하는 다른 단어들에 반해 단어들을 학습합니다. 

타겟 단어를 예측하기 위해서 문맥을 사용하거나 (continous bag of words, 또는 CBOW으로 알려진 방식) 혹은 타겟 문맥을 예측하기 위해서 단어를 사용하는 skip-gram 방식, 두가지 중 하나에서 그렇게 합니다. 저희는 대규모 데이터 세트에서 더 정확한 결과를 생산하는 후자의 방식을 사용합니다.

![Alt text](../img/word2vec_diagrams.png) 

한 단어에 할당된 속성 벡터가 그 단어의 문맥을 정확하게 예측하는데 사용될 수 없을 때, 그 벡터의 구성 요소들은 조정됩니다. 그 코퍼스에서 각 단어의 문맥은 속성 벡터를 조정하기 위해서 에러 신호를 돌려보내는 *선생님* 입니다. 그 문맥에 맞게 유사하게 판단된 단어의 벡터는 그 벡터에서 숫자들을 조정함으로써 서로 더 가깝게 접근 되었습니다.

반 고흐의 해바라기 그림이 1880년대 후반의 파리에서 3차원의 공간에 있는 식물성 물질을 *표현하는* 캔버스 상의 2차원 기름 혼합체인 것 처럼, 한 벡터에 배열된 500 숫자들은 한 단어 혹은 그룹의 단어들을 표현할 수 있습니다.

그 숫자들은 500차원의 벡터 공간에서 하나의 점으로서 각 단어를 찾습니다. 3차원 이상의 공간들은 시각화 하기가 어렵습니다. (사람들에게 13차원의 공간을 상상하도록 가르친 Geoff Hinton은 학생들에게 우선 3차원 공간을 생각한 다음 스스로에게 "13, 13, 13."을 말하라고 제안했습니다 :) 

잘 학습된 단어 벡터 세트는 그 공간에서 서로에게 가까이 유사한 단어들을 배치합니다. *oak*, *elm* 및 *birch* 단어들은 한 코너에서 모일 것 입니다. 반면, *war*, *conflict* 및 *strife*는 다른 곳에서 서로 군집할 것 입니다. 

비슷한 것들과 아이디어들은 "가까이" 보여집니다. 그들의 상대적인 의미는 측정 가능한 거리로 번역되어 왔습니다. 질은 양이 되고, 알고리즘은 그들의 작업을 할 수 있습니다. 그러나 유사성은 단지 Word2vec가 배울수 있는 많은 조합들의 기초 입니다. 예를 들어, 이는 한가지 언어의 단어들 사이에서 관계들을 측정할 수 있고 그들을 서로에게 매핑할 수 있습니다.

![Alt text](../img/word2vec_translation.png) 

이 벡터들은 단어들의 포괄적 기하학의 기초 입니다. 로마, 파리, 베를린 및 베이징이 서로 근처에서 모이는 것 뿐만 아니라, 그들은 각각 그 나라의 수도와 벡터 공간에서 비슷한 거리를 가집니다; 즉, 로마 - 이탈리아 = 베이징 - 중국. 만약 여러분께서 로마가 이탈리아의 수도라는 것 만을 알고, 중국의 수도에 대해 모르고 계셨다면, 이 방정식 로마 - 이탈리아 + 중국은 베이징을 답으로 제공할 것 입니다. 농담 아닙니다.

![Alt text](../img/countries_capitals.png) 

##<a name="crazy">재미있는 Word2Vec 결과</a>

이제 Word2vec이 생산할 수 있는 다른 조합들을 살펴보도록 하겠습니다. 

더하기, 빼기, 등호 대신 저희는 여러분께 논리적 유추의 표기법에서 결과들을 제공할 것 입니다.`:`는 "is to"를 의미하고 `::`는 "as"를 의미합니다; 즉, "Rome is to Italy as China is to Beijing" =  `Rome:Italy::Beijing:China`. 마지막 자리에서 "답"을 제공하기 보다는, 첫 세가지 요소들이 주어지면 저희는 여러분께 Word2vec 모델이 제안하는 단어들의 리스트를 제공할 것 입니다:

    king:queen::man:[woman, Attempted abduction, teenager, girl] 
    //Weird, but you can kind of see it
    
    China:Taiwan::Russia:[Ukraine, Moscow, Moldova, Armenia]
    //Two large countries and their small, estranged neighbors
    
    house:roof::castle:[dome, bell_tower, spire, crenellations, turrets]
    
    knee:leg::elbow:[forearm, arm, ulna_bone]
    
    New York Times:Sulzberger::Fox:[Murdoch, Chernin, Bancroft, Ailes]
    //The Sulzberger-Ochs family owns and runs the NYT.
    //The Murdoch family owns News Corp., which owns Fox News. 
    //Peter Chernin was News Corp.'s COO for 13 yrs.
    //Roger Ailes is president of Fox News. 
    //The Bancroft family sold the Wall St. Journal to News Corp.
    
    love:indifference::fear:[apathy, callousness, timidity, helplessness, inaction]
    //the poetry of this single array is simply amazing...
    
    Donald Trump:Republican::Barack Obama:[Democratic, GOP, Democrats, McCain]
    //It's interesting to note that, just as Obama and McCain were rivals,
    //so too, Word2vec thinks Trump has a rivalry with the idea Republican.
    
    monkey:human::dinosaur:[fossil, fossilized, Ice_Age_mammals, fossilization]
    //Humans are fossilized monkeys? Humans are what's left 
    //over from monkeys? Humans are the species that beat monkeys
    //just as Ice Age mammals beat dinosaurs? Plausible.
    
    building:architect::software:[programmer, SecurityCenter, WinPcap]

이 모델은 여러분께서 [import](#import)하고 플레이 할 수 있는 구글 뉴스 어휘에서 학습되었습니다. 잠시 Word2vec 알고리즘이 영어 구문의 단 하나의 규칙도 배운 적이 없다고 생각해보십시오. 그것은 세상에 대해 아무것도 모르며, 어떤 규칙 기반의 상징적인 논리 또는 지식 그래프와 무관합니다. 그리고 그것은 여전히 대부분의 지식 그래프들이 수년 간의 인간의 노동 이후 배우는 것 보다 유연하고 자동화된 방식에서 더 많이 배울 것 입니다. 그것은 빈 석판으로서 구글 뉴스 문서에게로 오고, 학습의 마지막에는 인간에게 무언가를 의미하는 복잡한 비유를 계산할 수 있습니다.

여러분은 또한 다른 조합을 위한 Word2vec 모델을 조회하실 수 있습니다. 모든 것이 서로를 대칭하는 두개의 아날로그일 필요는 없습니다. ([저희는 아래에 방법을 설명합니다....](#eval))

* Geopolitics: *Iraq - Violence = Jordan*
* Distinction: *Human - Animal = Ethics*
* *President - Power = Prime Minister*
* *Library - Books = Hall*
* Analogy: *Stock Market ≈ Thermometer*

반드시 동일한 문자들을 포함하지는 않는, 다른 비슷한 단어들로의 한 단어의 근접성의 감각을 구축함으로써, 저희는 딱딱한 토큰들을 넘어서 더 부드럽고 더 일반적인 의미의 감각으로 이동했습니다. 

# <a name="just">코드를 주십시오</a>

##<a name="anatomy">DL4J에서 Word2vec의 해부학</a>

여기 Deeplearning4j의 자연-언어 처리 구성 요소들이 있습니다:

* **SentenceIterator/DocumentIterator**: 데이터 세트를 반복하는데 사용됩니다. SentenceIterator는 문자열을 반환하고 DocumentIterator는 inputstream들과 작동합니다. 가능한 어떤 곳에서든 SentenceIterator를 사용하십시오.
* **Tokenizer/TokenizerFactory**: 텍스트를 토근화 하는데 사용됩니다. NLP 조건에서, 한 문장은 토근의 한 시리즈로서 표현 됩니다. TokenizerFactory는 한 "문장"을 위한 tokenizer의 한 순간을 생성합니다. 
* **VocabCache**: 단어 세기, 문서 발생, 토근의 세트 (이 경우 vocab이 아니라 발생되어 온 토큰), vocab (단어 벡터 검색 테이블 뿐만 아니라 [단어들의 모음](../bagofwords-tf-idf.html) 둘 모두에 포함된 속성들)을 포함한 메타 데이터를 추적하기 위해 사용됩니다.
* **Inverted Index**: 어디에서 단어들이 발생했는지에 대한 메타 데이터를 저장합니다. 그 데이터 세트를 이해하는데 사용될 수 있습니다. Lucene implementation[1]으로 Lucene index는 자동으로 생성됩니다.

Word2vec이 관련 알고리즘의 종족을 참조하는 반면, 이 구현은 <a href="../glossary.html#skipgram">Skip-Gram</a> Negative Sampling을 사용합니다.

## <a name="setup">Word2Vec 설정</a> 

Maven을 사용하여 IntelliJ에 새로운 프로젝트를 생성하십시오. 만약 여러분께서 그 방법을 모르신다면, 저희의 [퀵스타트 페이지](../kr-quickstart.html)를 보시기 바랍니다. 그리고 나서 이 속성들과 종속성들을 여러분의 프로젝트의 루트 디렉터리에 있는 POM.xml 파일에 지정하십시오 (여러분은 최신의 버전들을 위해 [Maven을 확인](https://search.maven.org/#search%7Cga%7C1%7Cnd4j)하실 수 있습니다 -- 그것들을 사용하시기 바랍니다...).

                <properties>
                  <nd4j.version>0.4-rc3.8</nd4j.version> // check Maven Central for latest versions!
                  <dl4j.version>0.4-rc3.8</dl4j.version>
                </properties>
                
                <dependencies>
                  <dependency>
                     <groupId>org.deeplearning4j</groupId>
                     <artifactId>deeplearning4j-ui</artifactId>
                     <version>${dl4j.version}</version>
                   </dependency>
                   <dependency>
                     <groupId>org.deeplearning4j</groupId>
                     <artifactId>deeplearning4j-nlp</artifactId>
                     <version>${dl4j.version}</version>
                   </dependency>
                   <dependency>
                     <groupId>org.nd4j</groupId>
                     <artifactId>nd4j-x86</artifactId> 
                     <version>${nd4j.version}</version>
                   </dependency>
                </dependencies>

### 데이터 로딩하기

이제 Java에서 새로운 클래스를 생성하고 이름을 지정하시기 바랍니다. 그 후, 여러분은 여러분의 .txt 파일에서 가공되지 않은 문장들을 가져와 그들을 여러분의 iterator로 통과하고, 그것들을 모든 단어를 소문자로 변환하는 것과 같은 일종의 전처리 과정으로 종속시킵니다. 

        log.info("Load data....");
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

만약 여러분께서 저희의 예제에서 제공된 문장들 외에 텍스트 파일을 로드하기를 원하신다면, 이를 실행하시기 바랍니다:

        log.info("Load data....");
        SentenceIterator iter = new LineSentenceIterator(new File("/Users/cvn/Desktop/file.txt"));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

말하자면, `ClassPathResource`를 삭제하고 여러분의 `.txt` 파일의 절대 경로를 `LineSentenceIterator`로 공급하는 것 입니다. 

        SentenceIterator iter = new LineSentenceIterator(new File("/your/absolute/file/path/here.txt"));

첫 시도에 여러분께서는 여러분의 커맨드 라인에 `pwd`를 입력하여 어떠한 디렉터리의 절대 파일 경로를 그 동일한 디렉터리 내에서 찾으실 수 있습니다. 그 경로로, 여러분은 그 파일 이름을 추가하고 *voila*를 하실 수 있습니다. 

### 데이터 토근화 하기

Word2vec는 전체 문장들 보다는 단어들로 공급될 필요가 있으므로, 다음 단계는 데이터를 토근화 하는 것 입니다. 한 텍스트를 토근화 하는 것은 그 텍스트를, 예를 들어 공백을 칠 때마다 새로운 토큰을 생성하는, 원자 단위로 부서뜨리는 것 입니다.

        log.info("Tokenize data....");
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

이는 한 줄 당 한 단어를 여러분께 제공할 것 입니다.

### 모델 학습하기

이제 데이터가 준비되었으므로 여러분께서는 Word2vec 신경망을 구성하고 토큰에서 공급하실 수 있습니다.

        int batchSize = 1000;
        int iterations = 3;
        int layerSize = 150;
        
        log.info("Build model....");
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(batchSize) //# words per minibatch.
                .minWordFrequency(5) // 
                .useAdaGrad(false) //
                .layerSize(layerSize) // word feature vector size
                .iterations(iterations) // # iterations to train
                .learningRate(0.025) // 
                .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
                .negativeSample(10) // sample size 10 words
                .iterate(iter) //
                .tokenizerFactory(tokenizer)
                .build();
        vec.fit();

이 구성은 상당수의 하이퍼파라미터들을 받아들입니다. 몇몇은 약간의 설명이 필요 합니다:

* *batchSize*는 한번에 여러분께서 처리하실 수 있는 단어의 양 입니다.
* *minWordFrequency*는 코퍼스에서 한 단어가 반드시 나타나야 하는 최소한의 숫자 입니다. 여기에서 만약 그것이 5번 미만으로 나타나면 그것은 학습되지 않은 것 입니다. 단어들은 그들에 대한 유용한 속성을 배우기 위해서 반드시 여러 문맥들에서 나타나야 합니다. 매우 큰 코퍼스에서, 최소한을 발생시키는 것은 합리적 입니다.
* *useAdaGrad* - Adagrad는 각각의 속성을 위한 다양한 기울기를 생성합니다. 저희는 여기에서 그것와 관련되지 않습니다.
* *layerSize*는 단어 벡터에서 속성의 수를 지정합니다. 이는 속성 공간에서 차원의 수와 동일 합니다. 500 속성에 의해 표현된 단어들은 500 차원 공간에서 점수가 됩니다.
* *iterations* 이는 여러분께서 망이 그 데이터의 배치 작업마다 그 계수를 업데이트 하게 하는 숫자의 수 입니다. 너무 적은 iterations은 가능한 모두를 배울 시간이 없을 수 있다는 것을 의미합니다; 너무 많을 경우 그 망의 학습을 더 길어지게 할 수 있습니다.
* *learningRate*는 단어들이 속성 공간에 재배치 되는 것과 같이, 계수들의 각 업데이트 마다의 단계 크기 입니다.
* *minLearningRate*는 학습 비율 상의 최저 한도 입니다. 학습 비율은 여러분께서 학습하고 있는 단어들의 수가 감소하는 대로 감소합니다. 만약 학습 비율이 너무 많이 줄어들면, 망의 학습은 더 이상 효율적이지 않습니다. 이는 계수들이 이동하도록 유지 합니다.
* *iterate*은 망에게 어떤 데이터 세트의 배치 작업에서 그것이 학습하는지를 알려줍니다.
* *tokenizer*는 현재의 배치 작업 단어들로 그것을 공급합니다. 
* *vec.fit()*은 구성된 망에게 학습을 시작하도록 알려줍니다.

### <a name="eval">Word2vec를 사용하여 모델 평가하기</a> 

다음 단계는 여러분의 속성 벡터들의 질을 평가하는 것 입니다.

        log.info("Evaluate model....");
        double sim = vec.similarity("people", "money");
        log.info("Similarity between people and money: " + sim);
        Collection<String> similar = vec.wordsNearest("day", 10);
        log.info("Similar words to 'day' : " + similar);
        
        //output: [night, week, year, game, season, during, office, until, -]

`vec.similarity("word1","word2")`는 여러분께서 입력한 두 단어의 코사인 유사성을 반환할 것 입니다. 그것이 1에 가까워질수록 망은 단어들을 더 가깝게 인식합니다 (위 스웨덴-노르웨이 예제를 보십시오). 예를 들면:

        double cosSim = vec.similarity("day", "night");
        System.out.println(cosSim);
        //output: 0.7704452276229858

`vec.wordsNearest("word1", numWordsNearest)`로 스크린에 프린트 된 단어들은 여러분께서 망이 의미상 비슷한 단어들을 모이게 했는지를 눈으로 확인하게 합니다. wordsNearest의 두번째 파라미터로 여러분께서 원하시는 가장 가까운 단어들의 수를 설정할 수 있습니다. 예를 들어:

        Collection<String> lst3 = vec.wordsNearest("man", 10);
        System.out.println(lst3);
        //output: [director, company, program, former, university, family, group, such, general]

### 모델 시각화 하기

저희는 단어 속성 벡터의 차원성을 감소하고, 단어들을 2 또는 3차원 공간으로 나타나게 하기 위해서 [TSNE](https://lvdmaaten.github.io/tsne/)에 의존합니다. 

        log.info("Plot TSNE....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);

### 저장하기, 재로드하기 & 모델 사용하기

여러분께서는 모델을 저장하기를 원하실 것 입니다. Deeplearning4j에서 모델들을 저장하는 일반적인 방법은 직렬화(serialization) utils을 통해서 입니다 (Java 직렬화는 한 객체를 *일련의* bytes로 전환하는 Python pickling에 가깝습니다).

        log.info("Save vectors....");
        WordVectorSerializer.writeWordVectors(vec, "words.txt");

이는 Word2vec가 학습되는 디렉터리의 루트에 나타날 `words.txt`라고 불리는 파일에 벡터들을 저장할 것 입니다. 그 파일에 있는 출력은 그의 벡터 표현과 함께 있는 일련의 숫자 다음의 줄 당 한 단어여야 합니다.

벡터들과 함께 작업을 계속하려면, 다음과 같이 단순히 `vec` 상에서 방식들을 불러오십시오:

        Collection<String> kingList = vec.wordsNearest(Arrays.asList("king", "woman"), Arrays.asList("queen"), 10);

Word2vec의 단어 산술의 고전적인 예제는 "king - queen = man - woman"과 그의 논리 확장인 its logical extension "king - queen + woman = man" 입니다. 

위의 예제는 벡터 `king - queen + woman`에 10개의 가장 가까운 단어들을 산출할 것 입니다. 이는 `man`을 포함해야 합니다. wordsNearest를 위한 첫번째 파라미터는 + 표시를 가지고 단어들과 연결된 "positive" 단어들 `king` 과 `woman`을 포함해야 합니다; 두번째 파라미터는 - 표시로 연결된 "negative" 단어 `queen` (positive 및 negative는 여기에 어떤 감정적인 의미를 가지지 않습니다)을 포함 합니다; 세번째는 여러분께서 보시고자 하는 가장 가까운 단어들의 목록의 길이 입니다. 이것을 파일: `import java.util.Arrays;`의 상단에 추가하는 것을 기억하십시오.

어떤 수의 조합도 가능하지만, 여러분께서 요청하는 단어들이 코퍼스에서 충분한 빈도를 가지고 발생한다면 그들은 합리적인 결과만을 반환할 것 입니다. 당연히 비슷한 단어 (또는 문서)를 반환하는 기능은 검색과 추천 엔진 모두의 기반 입니다. 

여러분께서는 벡터들을 이와 같은 메모리로 재로드 하실 수 있습니다:

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("words.txt"));

여러분은 검색 테이블로서 Word2vec을 사용하실 수 있습니다:

        WeightLookupTable weightLookupTable = wordVectors.lookupTable();
        Iterator<INDArray> vectors = weightLookupTable.vectors();
        INDArray wordVector = wordVectors.getWordVectorMatrix("myword");
        double[] wordVector = wordVectors.getWordVector("myword");

그 단어가 그 어휘 안에 있지 않으면 Word2vec은 영(0)을 반환합니다.

### <a name="import">Word2vec 모델 Importing</a>

저희가 저희의 학습된 망의 정확성을 테스트 하기 위해 사용하는 [구글 뉴스 코퍼스 모델](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)은 S3에 호스팅하고 있습니다. 현재 가지고 계신 하드웨어가 대규모 코퍼스에서 학습하는데 오랜 시간이 걸리는 사용자들께서는 머릿말 없이 Word2vec 모델을 탐구하기 위해서 간단히 그것을 다운로드 하실 수 있습니다.

만약 여러분께서 [C vectors](https://docs.google.com/file/d/0B7XkCwpI5KDYaDBDQm1tZGNDRHc/edit) 또는 Gensimm와 함께 학습하신다면, 이 문장이 모델을 import할 것 입니다.

    File gModel = new File("/Developer/Vector Models/GoogleNews-vectors-negative300.bin.gz");
    Word2Vec vec = WordVectorSerializer.loadGoogleModel(gModel, true);

`import java.io.File;`을 여러분의 가져온 패키지에 추가하는 것을 기억하십시오.

대형 모델들과 작업 시 힙 스페이스에 문제가 있을 수 있습니다. 구글 모델은 10G의 RAM 만큼 사용할 것이고, JVM은 단지  256 MB의 RAM으로 실행하므로, 여러분은 여러분의 힙 스페이스를 조정하셔야 합니다. `bash_profile` 파일로 (저희의 [Troubleshooting 섹션](../kr-gettingstarted.html#trouble)을 보십시오) 또는 IntelliJ 자체를 통해 그렇게 하실 수 있습니다: 

    //Click:
    IntelliJ Preferences > Compiler > Command Line Options 
    //Then paste:
    -Xms1024m
    -Xmx10g
    -XX:MaxPermSize=2g

### <a name="grams">N-grams & Skip-grams</a>

단어들은 한번에 벡터 하나에 판독되고, *일정한 범위 내에서 전후로 스캔 됩니다*. 그 범위들은 n-grams, 그리고 한 n-gram은 주어진 언어적인 시퀀스로부터의 *n* 항목들의 연속적인 시퀀스 입니다; 그것은 unigram, bigram, trigram, four-gram 또는 five-gram의 n번째 버전 입니다. skip-gram은 단순히 n-gram으로부터 항목들을 삭제합니다. 

Mikolov에 의해 대중화되고 DL4J 구현에서 사용된 skip-gram 표현은 더 일반화 가능한 문맥들의 생성으로 인해 continuous bag of words와 같은 다른 모델들보다 더 정확한 것으로 입증되었습니다. 

이 n-gram은 이제 주어진 단어 벡터의 의미를 배우기 위해 신경망으로 공급됩니다; 예를 들어 의미는 어떤 더 큰 의미, 혹은 레이블의 지표로서 그것의 유용성으로 정의됩니다.

### <a name="code">작업 예제</a>

이제 여러분께서는 Word2Vec를 설정하는 방법에 대한 기본적인 개념을 가지고 있고, 여기에 어떻게 DL4J의 API와 함께 사용될 수 있는지의 [한 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/Word2VecRawTextExample.java)가 있습니다:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/Word2VecRawTextExample.java?slice=22:64"></script>

[퀵스타트](../kr-quickstart.html)의 설명을 따른 후, 여러분은 IntelliJ에서 이 예제를 여시고 그것이 작동하는지 보기 위해 실행을 누르실 수 있습니다. 만약 학습 코퍼스에 포함되지 않는 단어로 Word2vec 모델을 조회한다면 그것은 null 반환할 것 입니다. 

### <a name="trouble">문제 해결 및 Word2Vec 튜닝하기</a>

*질문: 이와 같은 많은 stack trace들을 얻었습니다*

       java.lang.StackOverflowError: null
       at java.lang.ref.Reference.<init>(Reference.java:254) ~[na:1.8.0_11]
       at java.lang.ref.WeakReference.<init>(WeakReference.java:69) ~[na:1.8.0_11]
       at java.io.ObjectStreamClass$WeakClassKey.<init>(ObjectStreamClass.java:2306) [na:1.8.0_11]
       at java.io.ObjectStreamClass.lookup(ObjectStreamClass.java:322) ~[na:1.8.0_11]
       at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1134) ~[na:1.8.0_11]
       at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548) ~[na:1.8.0_11]

*답:* 여러분의 Word2vec 응용 프로그램을 시작하셨던 디렉터리를 살펴보십시오. 이는, 예를 들면, IntelliJ 프로젝트 홈 디렉터리 또는 여러분께서 커맨드 라인에 Java를 입력하셨던 디렉터리일 수 있습니다. 아래와 같이 보이는 일부 디렉터리를 가지고 있어야 합니다:

       ehcache_auto_created2810726831714447871diskstore  
       ehcache_auto_created4727787669919058795diskstore
       ehcache_auto_created3883187579728988119diskstore  
       ehcache_auto_created9101229611634051478diskstore

여러분의 Word2vec 응용 프로그램을 닫으시고 삭제를 시도하실 수 있습니다.

*질문: 저의 가공 전 텍스트 데이터로부터의 단어들이 저의 Word2vec 개체에 모두  나타나지 않습니다…*

*답:* 여러분의 Word2Vec 개체 상에서 **.layerSize()**를 통해 레이어 사이즈를 이처럼 증가시켜 보십시오.

        Word2Vec vec = new Word2Vec.Builder().layerSize(300).windowSize(5)
                .layerSize(300).iterate(iter).tokenizerFactory(t).build();

*질문: 어떻게 제 데이터를 로드하나요? 왜 학습은 오래 걸리나요?*

*답:* 만약 여러분의 모든 문장들이 *한* 문장으로 로드된 경우, Word2vec 학습은 매우 오랜 시간이 걸릴 수 있습니다. 그것은 Word2vec이 문장-수준의 알고리즘이기 때문입니다. 따라서 문장 범주들은 아주 중요합니다. 왜냐하면 동시 발생 통계는 문장 하나, 하나로 수집되기 때문 입니다. (GloVe의 경우, 문장 범주들은 문제가 되지 않습니다. 왜냐하면 그것은 코퍼스-전반으로 동시 발생을 보기 때문입니다. 많은 코퍼스의 경우, 평균 문장 길이는 6개의 단어 입니다. 이는 5의 윈도우 사이즈로 여러분께서 30번의 skip-gram 계산을 가지실 거라는 것을 의미합니다 (대략의 숫자로). 만약 여러분의 문장 범주를 지정하는 것을 잊었다면, 여러분은 10,000 단어 길이의 한 "문장"을 로그하실 것 입니다. 그 경우, Word2vec는 전체 10,000-단어 "문장"을 위해 전체 skip-gram 주기를 시도할 것 입니다. DL4J의 구현에서 한 줄은 한 문장을 가정합니다. 여러분 자신의 SentenceIterator와 Tokenizer에 연결하셔야 합니다. 여러분께 여러분의 문장들이 끝나도록 지정하는 것을 요청함으로써, DL4J는 언어-독립적으로 남습니다. UimaSentenceIterator는 그렇게 하는 한 방법 입니다. 그것은 문장 범주 검출을 위해 OpenNLP를 사용합니다.

*질문: 알려주신 모든 것을 했는데 그 결과는 여전히 제대로 된 것으로 보이지 않습니다.*

*답:* 만약 Ubuntu를 사용하신다면 직렬화된 데이터가 제대로 로드되지 않았을 수 있습니다. 이는 Ubuntu의 문제 입니다. 저희는 Linux의 다른 버전에서 이 버전의 Wordvec를 테스트 하시기를 추천드립니다.

###<a name="use">이용 사례</a>

구글 학술 검색은 [Deeplearning4j의 Word2vec 구현을 여기에](https://scholar.google.com/scholar?hl=en&q=deeplearning4j+word2vec&btnG=&as_sdt=1%2C5&as_sdtp=) 인용하여 논문의 실행 집계를 유지하고 있습니다.

벨기에에 기반을 둔 데이터 과학자, Kenny Helsens는 [Deeplearning4j의 Word2vec 구현](thinkdata.be/2015/06/10/word2vec-on-raw-omim-database/)을 NCBI'의 Online Mendelian Inheritance In Man (OMIM) 데이터베이스에 적용했습니다. 그리고 나서 그는 non-small cell lung carcinoma의 알려진 종양 유전자인 alk와 가장 유사한 단어들을 검색했고, Word2vec는 다음을 반환했습니다: "nonsmall, carcinomas, carcinoma, mapdkd." 거기에서 그는 다른 암 표현형들과 그들의 유전자형들 간의 유사성을 설립했습니다. 이는 큰 코퍼스 상에서 Word2vec이 학습할 수 있는 조합의 단지 한 예제일 뿐 입니다. 중요한 질병의 새로운 측면들을 발견하기 위한 가능성은 이제 막 시작했고 의학의 외부에도 그 기회는 동일하게 다양합니다.

스웨덴에서 Deeplearning4j의 Word2vec 구현을 학습한 Andreas Klintberg는 [Medium에 자세한 안내](https://medium.com/@klintcho/training-a-word2vec-model-for-swedish-e14b15be6cb)를 작성했습니다. 

Word2Vec는 DL4J가 [딥 오토인코더들](../deepautoencoder.html)로 구현하는 정보 검색 및 QA 시스템을 위한 텍스트 기반의 데이터를 준비하는데 특히 유용합니다. 

마케터들은 추천 엔진을 구축할 제품들 간의 관계를 설정하고자 할 수 있습니다. 투자자들은 단일 그룹의 피상적인 멤버들에 대한 소셜 그래프, 혹은 그들이 가지고 있을 위치나 재정적인 후원에 대한 다른 관계들을 분석할 수 있습니다.

###<a name="patent">구글의 Word2vec 특허</a>

Word2vec는 Tomas Mikolov가 이끄는 구글의 연구자들로 구성된 팀에 의해 소개된 [단어들의 벡터 표현들을 계산하는 방법](http://arxiv.org/pdf/1301.3781.pdf) 입니다. 구글은 Apache 2.0 라이센스에 따라 출시된 [오픈 소스 버전의 Word2vec를 호스트 합니다](https://code.google.com/p/word2vec/). 2014년, Mikolov는 페이스북으로 가기 위해 구글을 떠났고, 2015년 5월, [구글은 출시되어 온 Apache 라이센스를 폐지하지 않는 방식에 대한 특허](http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=1&f=G&l=50&co1=AND&d=PTXT&s1=9037464&OS=9037464&RS=9037464)를 획득했습니다.

###<a name="foreign">외국어</a>

모든 언어들로 단어들은 Word2vec를 통해 벡터로 전환될 수 있고 그 벡터들은 Deeplearning4j로 학습될 수 있는 반면, NLP 전처리는 매우 언어 구체적이며 저희의 라이브러리 이상의 도구들을 요구할 수 있습니다. [Stanford Natural Language Processing Group](http://nlp.stanford.edu/software/)은 [만다린 중국어](http://nlp.stanford.edu/projects/chinese-nlp.shtml), 아랍어, 프랑스어, 독일어 및 스페인어와 같은 언어들을 위한 토큰화, 품사 태깅 및 명명된 개체 인식을 위한 많은 Java 기반의 도구들을 가집니다. 일본어를 위해서는, [Kuromoji](http://www.atilika.org/)와 같은 NLP 도구들이 유용합니다. [텍스트 코퍼스를 포함한 다른 외국어 리소스는 여기에서 보실 수 있습니다](http://www-nlp.stanford.edu/links/statnlp.html).

### <a name="glove">GloVe: Global Veoctors</a>

GloVe 모델들을 word2vec에 로딩하고 저장하는 것은 이렇게 수행될 수 있습니다:

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("glove.6B.50d.txt"));

### <a name="sequence"SequenceVectors</a>

Deeplearning4j는 단어 벡터 위의 추상화의 한 수준이며, 여러분께서 소셜 미디어 프로필, 거래들, 단백질, 등을 포함한 어떤 시퀀스로부터 속성들을 추출하도록 하는  [SequenceVectors](https://github.com/deeplearning4j/deeplearning4j/blob/b6d1cdd2445b9aa36a7e8230c51cea14d00b37b3/deeplearning4j-scaleout/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/sequencevectors/SequenceVectors.java)라고 불리는 클래스를 가지고 있습니다. 만약 데이터가 시퀀스로 설명될 수 있다면 그것은 AbstractVectors 클래스와 함께 skip-gram과 계층적 softmax를 통해 학습될 수 있습니다. 이것은 [DeepWalk 알고리즘](https://github.com/deeplearning4j/deeplearning4j/blob/1ee1666d3a02953fc41ef41542668dd14e020396/deeplearning4j-scaleout/deeplearning4j-graph/src/main/java/org/deeplearning4j/graph/models/DeepWalk/DeepWalk.java)과 호환되며, 또한 Deeplearning4j에서 구현됩니다. 

### <a name="features">Deeplearning4j 상의 Word2Vec 속성</a>

* 모델 직렬화/역 직렬화 후 가산 업데이트가 추가되었습니다. 즉, 여러분은 `loadFullModel`을 요청하고, `TokenizerFactory` 및 `SentenceIterator`를 그것에 추가하고, 복원된 모델 상에서 `fit()`을 요청함으로써, 200GB의 새로운 텍스트로 모델 상태를 업데이트 할 수 있습니다.
* vocab 건설을 위한 여러 데이터 소스를 위한 옵션이 추가되었습니다.
* Epochs와 Iterations은 둘 모두 일반적으로 "1" 임에도 불구하고, 개별적으로 지정될 수 있습니다.
* Word2Vec.Builder는 이 옵션을 가지고 있습니다: `hugeModelExpected`. `true`로 설정하면, 그 vocab는 주기적으로 그 build 동안 주기적으로 절단될 것 입니다.
* `minWordFrequency`가 코퍼스에서 드문 단어들을 무시하는 데 유용한 반면, 어떤 숫자의 단어들은 사용자 지정에서 제외될 수 있습니다.
* 두개의 새로운 WordVectorsSerialiaztion 방식들이 소개되었습니다: `writeFullModel` 및 `loadFullModel`. 이들은 전체 모델 상태를 저장하고 로드합니다.
* 좋은 워크 스테이션은 몇 백만 단어들을 처리할 수 있어야 합니다. Deeplearning4j의 Word2vec 구현은 단일 머신에서 몇 terabytes의 데이터를 모델링 할 수 있습니다. 대략, 그 수학은: `vectorSize * 4 * 3 * vocab.size()`.

### Doc2vec과 다른 리소스들

* [DL4J 단락 벡터들로 텍스트 분류의 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/ParagraphVectorsClassifierExample.java)
* [Deeplearning4j와 함께 Doc2vec, 또는 단락 벡터](../doc2vec.html)
* [사고 벡터, 자연 언어 처리하기 & AI의 미래](../thoughtvectors.html)
* [Quora: Word2vec는 어떻게 작동하는가?](http://www.quora.com/How-does-word2vec-work)
* [Quora: 흥미로운 Word2Vec 결과들은 무엇인가?](http://www.quora.com/Word2vec/What-are-some-interesting-Word2Vec-results/answer/Omer-Levy)
* [Word2Vec: 소개](http://www.folgertkarsdorp.nl/word2vec-an-introduction/); Folgert Karsdorp
* [Mikolov'의 Word2vec 코드 원문 @구글](https://code.google.com/p/word2vec/)
* [word2vec 설명: Mikolov et al.’의 Negative-Sampling Word-Embedding 방식 도출하기](http://arxiv.org/pdf/1402.3722v1.pdf); Yoav Goldberg와 Omer Levy
* [Bag of Words & 용어 빈도-역 문서 빈도 (TF-IDF)](../bagofwords-tf-idf.html)

### <a name="doctorow">문학 속의 Word2Vec</a>

    언어의 모든 글자들이 숫자로 변환되는 것처럼, 숫자들은 마치 언어와 같다, 따라서 그것은 모두들 같은 방식을 이해하는 어떤 것이다. 당신은 문자들의 소리를 잃어버리고, 그들이 클릭하거나 튀어오르거나 미각을 건드리거나, 혹은 우 또는 아 하는지를, 그리고 잘못 읽혀지거나 음악 혹은 사진들로 당신을 속이든, 그것은 당신의 마음에 놓여 있다, 그 모든 것은 액센트와 함께 사라지고, 당신은 새로운 완전한 이해를, 숫자의 언어를 가지고, 모든 것은 모든 이에게 벽 위의 글 처럼 명백해진다. 그래서 내가 말할 때 숫자를 읽기 위한 어떤 시간이 다가온다.
        -- E.L. Doctorow, Billy Bathgate
