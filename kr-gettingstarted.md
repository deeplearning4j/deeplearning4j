---
layout: kr-default
---

# 시작 하기

이 웹 페이지는 공사 중 입니다. 보다 최신의 자료를 원하시면, 영문 페이지를 [방문](/gettingstarted.html)해주세요.

이것은 다단계 설치 입니다. 질문이나 피드백이 있으시다면 저희가 상세한 설명을 드릴 수 있드록 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 강력 추천드립니다. 비사교적 또는 완전 독립적인 성격이시라면 자율 학습 하실 수 있습니다.

Deeplearning4j를 위한 prerequisites 설치는 DL4J의 신경망에 동력을 지원하는 선형 대수학 엔진, ND4J의 [“Getting Started” 페이지](http://nd4j.org/kr-getstarted.html)에 문서화 되어 있습니다:

1. [Java 7](http://nd4j.org/kr-getstarted.html#java)
2. [통합 개발 환경(Integrated Development Environment): IntelliJ](http://nd4j.org/kr-getstarted.html#ide)
3. [Maven](http://nd4j.org/kr-getstarted.html#maven)
4. [Canova: An ML Vectorization Lib](http://nd4j.org/kr-getstarted.html#canova)
5. [GitHub](http://nd4j.org/kr-getstarted.html#github)

위의 설치를 마치신 후, 다음을 읽어주시기 바랍니다.

1. OS-specific instructions:
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
2. <a href="#source">소스를 사용한 작업(Working with Source)</a>
3. <a href="#eclipse">Eclipse</a>
4. <a href="#trouble">문제 해결</a>
5. <a href="#results">재생 가능한 결과(Reproducible Results)</a>
6. <a href="#next">다음 단계</a>

### <a name="linux">Linux</a>

CPU를 위한 Jblas에 대한 저희의 의존도로 인해, Blas를 위한 기본 바인딩(native bindings)이 필요합니다.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

만약 GPU가 파손된 경우, 별도의 커맨드를 입력하셔야 합니다. 우선, Cuda 자체 설치 위치를 확인하십시오. 이는 다음과 같을 것 입니다.

		/usr/local/cuda/lib64

그 다음 Cuda를 연결하는 파일 경로 다음의 터미널에 Idconfig를 입력하십시오. 여러분의 커맨드는 다음과 같을 것 입니다.

		ldconfig /usr/local/cuda/lib64

만약 여전히 Jcublas를 로드할 수 없다면, 여러분의 코드에 변수 -D를 추가하셔야 합니다. (이는 JVM 인수 입니다.):

     	java.library.path (settable via -Djava.librarypath=...) 
     	// ^ for a writable directory, then 
     	-D appended directly to "<OTHER ARGS>" 

여러분의 IDE로서 IntelliJ를 사용하고 계시다면, 이미 작동되고 있을 것 입니다.

### <a name="osx">OSX</a>

Jblas가 이미 OSX에 설치되어 있습니다.

### <a name="windows">Windows</a>

64-bit 컴퓨터를 가지고 계시더라도 [MinGW 32 bits](http://www.mingw.org) 설치하십시오. 그 다음 [Mingw를 사용한 Prebuilt dynamic 라이브러리](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)를 다운로드 하십시오.

[Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)을 설치하십시오. (Lapack이 여러분이 Intel compiler를 가지고 계신지 질문할 것 입니다.)

Lapack은 [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)의 대안을 제공합니다. [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)를 위한 문서들도 확인하십시오.

다른 방법으로, MinGW를 무시하고 여러분의 PATH의 한 폴더에 그 Blas dll 파일을 복사하실 수 있습니다. 예를 들어 MinGW bin 폴더로의 경로는 이와 같습니다: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. Windows에서 PATH 변수에 대한 보다 많은 설명을 원하시면 [top answer on this StackOverflow 페이지](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)를 읽어보십시오.

Cygwin은 지원되지 않습니다. DOS Windows에서 DL4J를 설치하셔야 합니다.

### <a name="source">소스를 사용한 작업</a>

여러분께서 프로젝트에 엄청난 투자를 계획하시고 있지 않는 한, 저희는 여러분이 소스를 사용해 작업하시기 보다는 [Maven Central에서 Deeplearning4j JAR 파일들](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)을 다운로드 하시기를 강력 추천드립니다 (물론, 언제나 환영 입니다만). Maven에서 다운로드 하시려면, [instructions on the ND4J site](http://nd4j.org/getstarted.html#maven)를 확인하십시오.

만약 소스를 사용해 작업하신다면, intelliJ 또는 Eclipse를 위해 [project Lombok plugin](https://projectlombok.org/download.html)의 설치가 필요할 것 입니다.

더 많이 알고 싶으시다면, 저희의 [Github repo](https://github.com/deeplearning4j/deeplearning4j)를 확인하십시오. Deeplearning4j를 개발하시길 원하시면 [Mac](https://mac.github.com) 또는 [Windows](https://windows.github.com)를 위한 Github을 설치하십시오. 이후 그 저장소를 git clone 하시고 Maven을 위한 다음의 커맨드를 실행하십시오.

		mvn clean install -DskipTests -Dmaven.javadoc.skip=true

다음의 단계들을 따르시면, 여러분은 0.0.3.3 예제들을 실행하실 수 있습니다.

### <a name="eclipse">Eclipse</a>

*git clone*을 실행하신 후, 다음의 커맨드를 입력하십시오. 이는 그 소스를 import 해 모든 설정을 완료 할 것 입니다.

		mvn eclipse:eclipse 

### <a name="trouble">문제 해결</a>

저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)을 통해 오류 메시지에 대해 문의해주십시오. 질문을 게시하실 때에는 다음의 정보를 준비해주시기 바랍니다 (처리가 엄청 빨라집니다!):

* Operating System (Windows, OSX, Linux) and version 
* Java version (7, 8) : type java -version in your terminal/CMD
* Maven version : type *mvn --version* in your terminal/CMD
* Stacktrace: Please past the error code on Gist -- [https://gist.github.com/](https://gist.github.com/) -- and share the link with us

이미 DL4J를 설치하셨고 이제 오류를 일으키는 예제들을 보고 계신다면, DL4J와 동일한 루트 디렉터리에 있는 [ND4J](http://nd4j.org/getstarted.html) 상의 git clone을 실행하십시오; ND4J 내에서 새로운 Maven 설치를 실행하십시오; DL4J를 재설치 하십시오; DL4J 내에서 새로운 Maven 설치를 실행하시고, 오류들이 해결되었는지 확인하십시오.

어떤 예제를 실행할 때, 망(net)의 분류(classification)가 정확한지의 확률인 [f1 점수](http://deeplearning4j.org/glossary.html#f1)를 낮게 받으실 수 있습니다. 이 경우, 낮은 f1 점수가 성능 저하를 의미하지 않습니다. 왜냐하면 예제들은 작은 데이터 세트에서 훈련되었기 때문입니다. 저희는 빠른 실행을 위해 예제들에 작은 데이터 세트를 주었습니다. 작은 데이터 세트들은 큰 데이터 세트들보다 덜 대표적이기 때문에, 그 보여지는 결과들은 크게 다를 수 있습니다. 예를 들어, 소문자 예제 데이터에서 저희의 심층 신뢰 망(deep-belief net)의 f1 점수는 현재 0.32 에서 1.0 사이에서 차이를 보입니다.

Deeplearning4J는 autocomplete function을 포함합니다. 어떤 커맨드들이 사용 가능한지 모르는 경우, 어떤 문자든 누르면 다음과 같이 드롭다운 리스트가 보여질 것 입니다.

![Alt text](../img/dl4j_autocomplete.png)

여기에 모든 [Deeplearning4j’s classes and methods](http://deeplearning4j.org/doc/)를 위한 Javadoc이 있습니다.

코드 베이스가 증가함에 따라 소스에서 설치하면 더 많은 메모리가 필요합니다. DL4J 구축 시 Permgen 오류를 경험하신다면, 더 많은 heap space를 추가해야 할 수 있습니다. 이를 위해 여러분의 숨겨진 .bash_profile file을 찾아 변경 하십시오. 이는 bash에 환경 변수들을 추가해 줄 것 입니다. 이 변수들을 보시려면, 커맨드 라인에 env를 입력하십시오. 더 많은 heap space를 추가하시려면 여러분의 콘솔(console)에 다음의 커맨드를 입력하시기 바랍니다: 

		echo “export MAVEN_OPTS=”-Xmx512m -XX:MaxPermSize=512m”” > ~/.bash_profile

3.0.4와 같은 Maven의 이전 버전들은 NoSuchMethodError와 같은 예외 사항들을 줄 가능성이 있습니다. 이는 Maven의 최신 버전으로 업그레이드 함으로써 해결될 수 있습니다.

일부 ND4J 디펜던시들을 컴파일 하시려면, C 또는 C++를 위한 몇 가지 개발 도구들을 설치하셔야 합니다. [저희의 ND4J guide를 보십시오](http://nd4j.org/getstarted.html#devtools).

DL4J를 사용하여 발생하는 일부 문제들은 기계 학습의 아이디어와 기술에 익숙하지 않아서 일 수 있습니다. 저희는 모든 Deeplearning4j 이용자들이 기본을 이해하기 위해 이 웹사이트를 넘어선 리소스들을 이용하기를 강력히 권장합니다. Andrew Ng의 훌륭한 강의인 [machine-learning lectures on Coursera](https://www.coursera.org/learn/machine-learning/home/info)가 좋은 시작이 될 수 있습니다. [Youtube에 있는 Geoff Hinton의 neural nets course](https://www.youtube.com/watch?v=S3bx8xKpdQE) 역시 매우 교육적입니다. 저희가 부분적으로 DL4J를 문서화 해 왔지만 deep learning을 위해서는 여전히 코드의 많은 부분들이 본질적으로 완성되지 않은 도메인 특정 언어 입니다.

[Java CPP](https://github.com/bytedeco/javacpp)를 위한 include path가 항상 windows에서 작동하지만 않습니다. 한 가지 해결 방법은 Visual Studio의 include directory로부터 header 파일들을 가져와 Java가 설치되어 있는 Java Run-Time Environment (JRE)의 include directory에 붙여 넣는 것 입니다. (이는 standardio.h와 같은 파일들에 영향을 미칠 것 입니다.)

### <a name="results">재생 가능한 결과</a>

신경망 가중치는 임의로 초기화 됩니다. 이는 모델이 매 번 중량 공간의 다른 위치에서 학습을 시작해 다른 로컬 최적 조건을 이끌어낼 수 있슴을 의미합니다. 재생 가능한 결과를 원하시는 이용자는 동일한 임의의 가중치를 사용하셔야 하며 이 가중치는 모델이 생성되기 이전에 초기화 되어야 합니다. 동일한 임의 가중치는 다음의 라인으로 재초기화 될 수 있습니다:

		Nd4j.getRandom().setSeed(123);

### <a name="next">다음 단계: IRIS 예제와 NNs 구축하기</a>

신경망 구축 시작을 위해서는 [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html)에 더 많은 정보가 있습니다.

빨리 배우시려면 [IRIS tutorial](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)을, 다른 데이터 세트를 탐구하시려면 [custom datasets](http://deeplearning4j.org/customdatasets.html)을 이용하시기 바랍니다.

새로운 프로젝트를 시작하고, 필요한 [POM 디펜던시들](http://nd4j.org/dependencies.html)을 포함하시려면 [ND4J Getting Started](http://nd4j.org/getstarted.html) 설명을 따르십시오.
