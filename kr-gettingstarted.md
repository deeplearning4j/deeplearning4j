---
layout: default
---

시작 하기

이것은 다단계 설치 입니다. 질문이나 피드백이 있으시다면 저희가 상세한 설명을 드릴 수 있드록 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 강력 추천드립니다. 비사교적 또는 완전 독립적인 성격이시라면 자율 학습 하실 수 있습니다.


Deeplearning4j를 위한 prerequisites 설치는 DL4J의 신경망에 동력을 지원하는 선형 대수학 엔진, ND4J의 [“Getting Started” 페이지](http://nd4j.org/getstarted.html)에 문서화 되어 있습니다:

1. [Java 7](http://nd4j.org/getstarted.html#java)
2. [통합 개발 환경(Integrated Development Environment): IntelliJ](http://nd4j.org/getstarted.html#ide)
3. [Maven](http://nd4j.org/getstarted.html#maven)
4. [Canova: An ML Vectorization Lib](http://nd4j.org/getstarted.html#canova)
5. [GitHub](http://nd4j.org/getstarted.html#github)

위의 설치를 마치신 후, 다음을 읽어주시기 바랍니다.

OS-specific instructions
[Linux](http://deeplearning4j.org/gettingstarted.html#linux)
[OSX](http://deeplearning4j.org/gettingstarted.html#osx)
[Windows](http://deeplearning4j.org/gettingstarted.html#windows)

2. [소스를 사용한 작업(Working with Source)](http://deeplearning4j.org/gettingstarted.html#source)

3. [Eclipse](http://deeplearning4j.org/gettingstarted.html#eclipse)

4. [문제 해결](http://deeplearning4j.org/gettingstarted.html#trouble)

5. [재생 가능한 결과(Reproducible Results)](http://deeplearning4j.org/gettingstarted.html#results)

6. [다음 단계](http://deeplearning4j.org/gettingstarted.html#next)


Linux

CPU를 위한 Jblas에 대한 저희의 의존도로 인해, Blas를 위한 기본 바인딩(native bindings)이 필요합니다.

————
	•	Fedora/RHEL
	•	  yum -y install blas
	•	
	•	  Ubuntu
	•	  apt-get install libblas* (credit to @sujitpal)
	•	
———-

만약 GPU가 파손된 경우, 별도의 커맨드를 입력하셔야 합니다. 우선, Cuda 자체 설치 위치를 확인하십시오. 이는 다음과 같을 것 입니다.

————
	•	/usr/local/cuda/lib64
————

그 다음 Cuda를 연결하는 파일 경로 다음의 터미널에 Idconfig를 입력하십시오. 여러분의 커맨드는 다음과 같을 것 입니다.

—————
ldconfig /usr/local/cuda/lib64
————

만약 여전히 Jcublas를 로드할 수 없다면, 여러분의 코드에 변수 -D를 추가하셔야 합니다. (이는 JVM 인수 입니다.):

—————
     java.library.path (settable via -Djava.librarypath=...) 
     // ^ for a writable directory, then 
     -D appended directly to "<OTHER ARGS>" 
————

여러분의 IDE로서 IntelliJ를 사용하고 계시다면, 이미 작동되고 있을 것 입니다.


OSX

Jblas가 이미 OSX에 설치되어 있습니다.


Windows

64-bit 컴퓨터를 가지고 계시더라도 [MinGW 32 bits](http://www.mingw.org) 설치하십시오. 그 다음 [Mingw를 사용한 Prebuilt dynamic 라이브러리](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)를 다운로드 하십시오.

[Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)을 설치하십시오. (Lapack이 여러분이 Intel compiler를 가지고 계신지 질문할 것 입니다.)

Lapack은 [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)의 대안을 제공합니다. [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)를 위한 문서들도 확인하십시오.

다른 방법으로 MinGW를 무시하고 여러분의 PATH의 한 폴더에 그 Blas dll 파일을 복사하실 수 있습니다. 예를 들어 MinGW bin 폴라로이드 경로는 다음과 같습니다: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. Windows에서 PATH 변수에 대한 보다 많은 설명을 원하시면 [top answer on this StackOverflow 페이지](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)를 읽어보십시오.

Cygwin은 지원되지 않습니다. DOS Windows에서 DL4J를 설치하셔야 합니다.


소스를 사용한 작업

여러분께서 프로젝트에 엄청난 투자를 계획하시고 있지 않는 한, 저희는 여러분이 소스를 사용해 작업하시기 보다는 [Maven Central에서 Deeplearning4j JAR 파일들](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)을 다운로드 하시기를 강력 추천드립니다 (물론, 언제나 환영 입니다만). Maven에서 다운로드 하시려면, [instructions on the ND4J site](http://nd4j.org/getstarted.html#maven)를 확인하십시오.

만약 소스를 사용해 작업하신다면, intelliJ 또는 Eclipse를 위해 [project Lombok plugin](https://projectlombok.org/download.html)의 설치가 필요할 것 입니다.

더 많이 알고 싶으시다면, 저희의 [Github repo](https://github.com/deeplearning4j/deeplearning4j)를 확인하십시오. Deeplearning4j를 개발하시길 원하시면 [Mac](https://mac.github.com) 또는 [Windows](https://windows.github.com)를 위한 Github을 설치하십시오. 이후 그 저장소를 git clone 하시고 Maven을 위한 다음의 커맨드를 실행하십시오.

————
  mvn clean install -DskipTests -Dmaven.javadoc.skip=true
————-

다음의 단계들을 따르시면, 여러분은 0.0.3.3 예제들을 실행하실 수 있습니다.


Eclipse

git clone을 실행하신 후, 다음의 커맨드를 입력하십시오. 이는 그 소스를 import 해 모든 설정을 완료 할 것 입니다.

————
  mvn eclipse:eclipse 
————


문제 해결

저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)을 통해 오류 메시지에 대해 문의해주십시오. 질문을 게시하실 때에는 다음의 정보를 준비해주시기 바랍니다 (처리가 엄청 빨라집니다!):

————
	•	* Operating System (Windows, OSX, Linux) and version 
	•	* Java version (7, 8) : type java -version in your terminal/CMD
	•	* Maven version : type mvn --version in your terminal/CMD
	•	* Stacktrace: Please past the error code on Gist and share the link with us: https://gist.github.com/
	•	
————

DL4J를 이미 설치하셨고 이제 오류를 일으키는 예제들이을 보고 계신다면, DL4J와 동일한 루트 디렉터리에 있는 [ND4J](http://nd4j.org/getstarted.html) 상의 git clone을 실행하십시오; ND4J 내에서 새로운 Maven 설치를 실행하십시오; DL4J를 재설치 하십시오; DL4J 내에서 새로운 Maven 설치를 실행하시고, 오류들이 해결되었는지 확인하십시오.

어떤 예제를 실행할 때, 망(net)의 분류(classification)가 정확성 확률인 [f1 점수](http://deeplearning4j.org/glossary.html#f1)를 낮게 받으실 수 있습니다. 이 경우, 낮은 f1 점수가 성능 저하를 의미하지 않습니다. 왜냐하면 예제들은 작은 데이터 세트에서 훈련되었기 때문입니다. 저희는 빠른 실행을 위해 예제들에 작은 데이터 세트를 주었습니다. 작은 데이터 세트들은 큰 데이터 세트들보다 덜 대표적이기 때문에, 그 보여지는 결과들은 크게 다를 수 있습니다. 예를 들어, 소문자 예제 데이터에서 저희의 심층 신뢰 망(deep-belief net)의 f1 점수는 현재 0.32 에서 1.0 사이에서 차이를 보입니다.

Deeplearning4J는 autocomplete function을 포함합니다. 어떤 커맨드들이 사용 가능한지 모르는 경우, 어떤 문자든 누르면 다음과 같이 드롭다운 리스트가 보여질 것 입니다.

————
