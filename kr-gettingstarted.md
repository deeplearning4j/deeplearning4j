---
title: 전체 설치
layout: kr-default
---

# 전체 설치

전체 설치는 여러 단계의 소프트웨어를 설치해야 합니다. 질문이나 피드백이 있으시다면 저희가 상세한 설명을 드릴 수 있드록 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 참여하시길 바랍니다. 대화에 꼭 참여하지 않으셔도 사람들의 질문과 대화를 보며 여러 가지를 배울 수 있씁니다. 만일 딥 러닝에 대해 전혀 아는 내용이 없으시면, [시작하실때 무엇을 배워야 할지를 보여주는 로드맵](http://deeplearning4j.org/deeplearningforbeginners.html) 페이지를 참고하시기 바랍니다.

좀 더 빠르게 저희 예제를 실행하시려면 [퀵 스타트 페이지](http://nd4j.org/kr-index.html)를 참고하시기 바랍니다. 사실 전체 설치 전에 예제들을 먼저 실행해보시기를 권하는 편입니다. 그렇게 하시는 편이 좀 더 DL4J를 쉽게 배울 수 있습니다.

저희가 사용하는 자바 기반 연산 엔진입니다

Deeplearning4j를 위한 필수 사항 설치는 DL4J가 사용하는 자바 기반 연산 엔진 ND4J의 [“Getting Started” 페이지](http://nd4j.org/kr-getstarted.html)에 문서화 되어 있습니다:

1. [Java 7 혹은 상위 버전](http://nd4j.org/kr-getstarted.html#java)
2. [통합 개발 환경(Integrated Development Environment): IntelliJ](http://nd4j.org/kr-getstarted.html#ide)
3. [Maven](http://nd4j.org/kr-getstarted.html#maven)

위의 설치를 마치신 후, 다음을 읽어주시기 바랍니다.

1. 각 OS별 세부사항 안내:
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
2. [GitHub](http://nd4j.org/kr-getstarted.html#github)
3. <a href="#eclipse">Eclipse</a>
4. <a href="#trouble">문제 해결</a>
5. <a href="#results">비교용 벤치마크 결과 (Reproducible Results)</a>
6. <a href="#next">다음 단계</a>

### <a name="linux">Linux</a>

* Deeplearning4j는 효율적인 CPU 연산을 위해 다양한 종류의 Blas를 지원합니다. 따라서 각 Blas와 CPU의 네이티브 바인딩(native bindings)이 필요합니다.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

OpenBlas에 대한 자세한 정보는 [이 섹션](http://nd4j.org/kr-getstarted.html#open)을 참고해 주시기 바랍니다.

* GPU연결에 문제가 있는 경우 아래의 커맨드를 입력하셔야 합니다. 우선, 컴퓨터의 Cuda 설치 위치를 확인하십시오. 디폴트 설치 위치는 다음과 같습니다.

		/usr/local/cuda/lib64

그리고 Cuda 링크를 위해 리눅스 파일 경로에 *Idconfig*를 입력해야 합니다. 이는 아래와 같은 커맨드로 입력 가능합니다.

		ldconfig /usr/local/cuda/lib64

위의 단계 이후에도 여전히 Jcublas가 로딩이 되지 않는다면, 여러분의 코드에 옵션 -D를 추가하셔야 합니다(JVM 입력변수입니다).:

     	java.library.path (settable via -Djava.librarypath=...) 
     	// ^ for a writable directory, then 
     	-D appended directly to "<OTHER ARGS>" 

만일 IDE로 IntelliJ를 사용하신다면, 이 과정이 특별히 필요 없을 것 입니다.

### <a name="osx">OSX</a>

* Blas는 OSX에 기본적으로 설치되어 있습니다.

### <a name="windows">Windows</a>

* DL4J를 Windows 설치에 설치하는 과정은 조금 복잡합니다. 하지만 Deeplearning4j는 윈도우 사용자를 지원하는 몇 안되는 오픈 소스 딥 러닝 프로젝트 입니다. 자세한 안내는 저희의 [ND4J 페이지의 윈도우 섹션](http://nd4j.org/kr-getstarted.html#windows)을 참조하십시오.

* 64-bit OS가 설치되어 있더라도 [MinGW 32 bits](http://www.mingw.org)를 설치하시고 나서 (다운로드 버튼은 우측 상단에 있습니다), [Mingw를 사용한 Prebuilt dynamic 라이브러리](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)를 다운로드 하십시오.

* [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)을 설치하십시오. (아마도 설치 과정에서 Lapack은 Intel compiler가 설치되어 있는지를 물어볼 것 입니다. 보통은 설치되어 있지 않습니다.)

* Lapack의 [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)의 대안입니다. 더 자세한 내용은 [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)의 문서를 확인해보십시오.

* 혹은, MinGW를 설치하는 대신에 PATH의 폴더에 Blas dll 파일을 복사하는 방법도 있습니다. 예를 들어, MinGW bin 폴더로의 경로 /usr/x86_64-w64-mingw32/sys-root/mingw/bin 입니다. Windows에서 PATH 설정에 대한 보다 자세한 설명을 원하시면 [StackOverflow의 주요 질문 답변 모음 페이지](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)를 참고하십시오.

* Cygwin은 지원되지 않습니다. **DOS Windows**에서 DL4J를 설치하셔야 합니다.

* [WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100)를 실행하면 여러분의 Window 설치 과정의 문제를 해결할 수 있습니다. 예를 들면 [이런 식으로 문제를 알려줍니다](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f). 우선 WindowsInfo.bat을 다운로드 하시고, 커맨드 창 / 터미널을 엽니다. 그리고 다운로드 폴더로 가서 `WindowsInfo`를 입력하십시오. 실행 결과로 나온 출력 메시지를 복사하려면 커맨드 창에서 오른쪽 마우스를 클릭 -> 전체 선택 -> enter를 누르면 출력이 클립보드에 복사됩니다.

**Windows**에서 OpenBlas (아래 참조)를 사용하려면, 이 [파일](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1)을 다운로드 하십시오. 그리고 임의의 폴더 (예:`C:/BLAS`)에 압축을 푸시고, 여러분의 시스템의 `PATH` 환경 변수에 그 디렉토리를 추가하십시오.

### <a name="openblas">OpenBlas</a>

x86 백엔드 위에서 네이티브 라이브러리를 사용하려면 시스템 경로에 `/opt/OpenBLAS/lib`을 추가해야 합니다. 그리고 프롬프트에서 다음의 커맨드를 입력하십시오.

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3
			
이렇게 하면 [Spark](http://deeplearning4j.org/spark)가 OpenBlas를 사용할 수 있게 됩니다.

만약 OpenBlas가 잘 작동하지 않으면, 아래의 단계들을 따르십시오.

* Openblas가 이미 설치되어 있다면 먼저 삭제하십시오.
* `sudo apt-get remove libopenblas-base`를 실행하십시오.
* OpenBLAS 개발 버전을 다운로드 하십시오.
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* **Linux**에서는, 만약 `libblas.so.3`와 `liblapack.so.3`의 심볼릭 링크가 `LD_LIBRARY_PATH`에 설정되어 있다면 그 링크를 더블 클릭하십시오. 설정되어 있지 않다면 `/usr/lib`에 링크를 추가하십시오. 심볼릭 링크는 다음과 같이 설정하실 수 있습니다 (-s가 심볼릭 링크를 위한 옵션입니다.):

		ln -s TARGET LINK_NAME
		// 해석: ln -s "여기로" <- "여기에서" 연결이 됨
* 위의 "LINK_NAME" 여러분이 만드는 새로운 심볼릭 링크 입니다. 여기 [심볼릭 링크를 만드는 방법 (영문)](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux)의 StackOverflow링크도 참고하시기 바랍니다. 또한 [리눅스 매뉴얼 페이지](http://linux.die.net/man/1/ln)도 참고하시기 바랍니다.
* 위의 단계가 마무리되면 IDE를 재시작 하십시오. 
* Native Blas를 **Centos 6**에서 사용하는 하는 방법은 [이 페이지](https://gist.github.com/jarutis/912e2a4693accee42a94) 또는 [여기](https://gist.github.com/sato-cloudian/a42892c4235e82c27d0d)에 자세히 나와있습니다.

**Ubuntu** (15.10)에서 OpenBlas를 사용하는 방법은 [이 안내](http://pastebin.com/F0Rv2uEk)를 참고하시기 바랍니다.				
### <a name="eclipse">Eclipse</a>

*git clone*을 실행하신 후, 다음 커맨드를 입력하십시오.

		mvn eclipse:eclipse 

이 커맨드는 이클립스에 관련된 설정에 필요한 모든 소스를 import합니다.

Eclipse가 익숙하시다면 유사한 인터페이스를 가지고 있는 IntelliJ를 권해드립니다.
여러 해 동안 Eclipse를 사용한 후, 저희는 비슷한 인터페이스를 가지고 IntelliJ를 권장합니다. Eclipse의 구조적 특성상 Deeplearning4j 및 다른 라이브러리 코드에서 종종 오류가 나기도 합니다.

Eclipse를 사용한다면 우선 [Lombok plugin](https://projectlombok.org/)을 설치해야 합니다. 또 Eclipse용 Maven plugin [eclipse.org/m2e/](https://eclipse.org/m2e/)도 설치해야 합니다.

Michael Depies가 작성한 [Eclipse에서 Deeplearning4j 설치하기](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/) 가이드도 참고하시길 바랍니다.

### <a name="trouble">문제 해결</a>

* 다른 오류가 발생할 경우 저희 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)을 통해 문의주시기 바랍니다. 질문을 올리실 때는 빠른 처리를 위해 다음의 정보를 준비해주시기기 바랍니다.

      * 운영 체제 (윈도우, 맥, 리눅스) 및 버전 
      * 자바 버전 : 커맨드 라인에서 java -version 을 입력하여 확인
      * Maven 버전 : type mvn --version in your terminal/CMD
      * Stacktrace 오류: [Gist (https://gist.github.com/)](https://gist.github.com/)에 에러 코드를 올린 뒤 링크를 공유: 
* 기존에 설치한 DL4J로 예제를 실행했을 때 오류가 발생한다면 우선 라이브러리를 최신 버전으로 업데이트하십시오. Maven을 사용하실 경우 설치된 POM.xml 파일의 내용만 업데이트 하시면 [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)에 나온 최신 버전으로 업데이트가 됩니다. 소스를 받아서 직접 설치하시는 경우엔 [ND4J](http://nd4j.org/kr-getstarted.html), Canova 및 DL4J 상의 `git clone`을 하신 뒤, 순서대로 각 디렉터리에서 `mvn clean install -Dskiptests=true -Dmaven.javadoc.skip=true`을 실행하시면 됩니다.
* 예제를 실행하다 보면 인공 신경망의 분류가 정확하게 이루어지고 있는지를 측정하는 [F1 점수](http://deeplearning4j.org/glossary.html#f1)가 생각보다 낮을수도 있습니다. 우선 예제는 빠른 실행과 검토를 위해 작은 데이터 셋을 사용하여 학습이 되기 때문에 실제 상황보다 낮은 점수가 나올 수 있습니다. 작은 데이터 셋을 사용할 경우 학습 데이터의 분포가 실제 데이터의 분포와 달라 실제 상황을 잘 반영하지 못할 수도 있고 인공 신경망의 학습엔 데이터의 양이 부족할 수도 있습니다. 예를 들어, 소문자 예제 데이터에서 저희의 DBN(Deep-Belief Net)의 F1 점수는 대체로 0.32 에서 1.0 사이의 값이 나옵니다.
* Deeplearning4J는 **자도완성 기능**을 갖고 있습니다. 어떤 커맨드를 사용해야 할지 애매한 경우 아무 문자나 누르면 아래 그림처럼 드롭다운 목록이 나옵니다.
![Alt text](../img/dl4j_autocomplete.png)
* [Deeplearning4j’의 클래스와 메소드](http://deeplearning4j.org/doc/) 페이지를 참고하시기 바랍니다. **Javadoc**으로 쓰여져 있어 편리하게 사용할 수 있습니다.
* 코드 양이 점점 많아지고 있기 때문에 소스에서 직접 설치할 경우 상당히 많은 메모리가 필요합니다. 이와 관련해서 DL4J 빌드 중에 `Permgen 오류`가 나타나는 경우 **힙 스페이스(heap space)**를 늘려야합니다. 이는 히든 파일인 `.bash_profile` 에 환경 변수(Environment variable)을 해서 해결할 수 있습니다. 우선 현재 설정된 환경 변수 목록을 보려면 커맨드 라인에서 `env`를 입력하십시오. 그리고 힙 스페이스를 늘리려면 터미널에서 다음의 커맨드를 입력하시기 바랍니다.
		echo “export MAVEN_OPTS=”-Xmx512m -XX:MaxPermSize=512m”” > ~/.bash_profile
* 구 버전의 Maven(예:3.0.4)은 NoSuchMethodError와 같은 exception 에러가 있을 수 있습니다. 이를 해결하려면 Maven을 최신 버전으로 업데이트 하십시오. Maven의 버전은 커맨드 라인에서 `mvn -v`를 입력하면 확인할 수 있습니다.
* Maven 설치 후 다음과 같은 메시지가 출력될 수 있습니다: `mvn is not recognised as an internal or external command, operable program or batch file.` 이를 해결하려면 환경 변수 [PATH variable](https://www.java.com/en/download/help/path.xml)에 Maven의 경로를 추가해야 합니다.
* 만일 `Invalid JDK version in profile 'java8-and-higher': Unbounded range: [1.8, for project com.github.jai-imageio:jai-imageio-core com.github.jai-imageio:jai-imageio-core:jar:1.3.0`라는 오류가 나온다면, Maven에 문제가 있는 것입니다. Maven을 버전 3.3.x으로 업데이트 하십시오.
* 일부 ND4J 디펜던시는 C 또는 C++를 위한 몇 가지 **개발 도구**들을 설치해야 컴파일을 할 수 있습니다. [저희의 ND4J guide를 보십시오](http://nd4j.org/kr-getstarted.html#devtools).
* [Java CPP](https://github.com/bytedeco/javacpp)의 include path는 **Windows**에서 작동하지는 않을 수도 있습니다. 한 가지 해결 방법은 Visual Studio의 include directory에서 header 파일을 Java가 설치되어 있는 Java Run-Time Environment (JRE)의 include directory에 복사하는 것 입니다. (이는 standardio.h와 같은 파일에 영향을 미칠 것 입니다.) 더 많은 정보는 [여기](http://nd4j.org/kr-getstarted.html#windows)를 참고하시기 바랍니다.
* GPU 모니터링은 [여기](http://nd4j.org/kr-getstarted.html#gpu)를 참고하시기 있습니다.
* 자바의 강점 중 하나는 **[JVisualVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jvisualvm.html)**에서 제공하는 진단 도구(diagnostics) 입니다. 자바를 설치하신 뒤 커맨드 라인에 `jvisualvm`를 입력하면 여러분의 CPU, Heap, PermGen, Classes 및 Threads 의 실시간 정보를 시각화 해서 보여줍니다. 팁: 우측 상단의 `Sampler` 탭을 클릭하시고, 시각화 할 디바이스(CPU 또는 Memory)에 해당하는 버튼을 선택하십시오. 
![Alt text](../img/jvisualvm.png)
* 기계 학습의 아이디어 및 원리에 익숙하지 않은 경우 사용중에 몇 가지 문제가 있을 수 있습니다. 저희는 Deeplearning4j 사용자들이 기계 학습의 기초를 이해할 수 있도록 저희의 튜토리알 뿐만 아니라 더 깊은 내용을 이해하시길 강력하게 추천합니다. 우선 [이 페이지](../deeplearningpapers.html)에 저희가 준비한 기계 학습 및 딥 러닝을 공부 목록을 포함시켰습니다. DL4J의 일부는 잘 문서화가 되어있지만, 전부 문서화가 되어있지는 않습니다. 특히 코드의 일부 핵심적인 부분은 문서화 되어있지 않고 코드로만 존재합니다.
* **Clojure** 에서 `deeplearning4j-nlp`을 사용할 때, 그리고 Leiningen에서 uberjar를 빌드 할 때, `project.clj`에서 akka `reference.conf` 리소스 파일들을 제대로 설정하려면 다음 내용을 따라야 합니다. `:uberjar-merge-with {#"\.properties$" [slurp str spit] "reference.conf" [slurp str spit]}`. (대부분의 경우 .properties 맵의 첫 번째 항목이 디폴트로 설정되어 있습니다.) 이 설정이 되어있지 않으면 uberjar에서 실행하려고 할 때 다음의 exeption 오류가 날 수 있습니다. `Exception in thread "main" com.typesafe.config.ConfigException$Missing: No configuration setting found for key 'akka.version'`.
* Float 데이터 형식은 OSX에서 종종 문제가 생깁니다. 만일 예제 실행 도중에 NAN 값이 나타난다면 변수의 데이터 형식을 `double`로 바꾸고 다시 시도해보십시오.
* 자바 7에서 fork-join에 버그가 있습니다. 이 버그는 Java 8로 업데이트를 해야 해결할 수 있습니다. 만일 아래와 같은 OutofMemory 오류가 나온다면 fork join때문일 수 있으니 참고하십시오. `java.util.concurrent.ExecutionException: java.lang.OutOfMemoryError`
.... `java.util.concurrent.ForkJoinTask.getThrowableException(ForkJoinTask.java:536)`

### <a name="results">재현 가능한 결과</a>

일반적으로 학습 과정에서 신경망의 값은 임의의 값으로 초기화 됩니다. 즉, 시작 조건이 다르므로 학습할 때 마다 다른 최종 값으로 수렴할 수 있습니다. 따라서 알려진 학습 결과를 직접 재현하려면 알려진 것과 동일한 임의의 값으로 초기화해야 하며 이 과정은 모델이 생성되기 전에 완료되어야 합니다. 다음의 코드를 이용하면 초기 값을 설정할 수 있습니다.

		Nd4j.getRandom().setSeed(123);

### <a name="next">다음 단계: IRIS 예제와 인공 신경망 설계하기</a>

신경망을 설계하기 전에 [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html)에서 신경망에 대한 더 자세한 정보를 확인하시기 바랍니다.

속성 안내를 찾으신다면 [IRIS tutorial](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)를, *Deep-Belief Networks (DBNs)*의 기본 원리를 이해하시려면 [Restricted Boltzmann Machines](../restrictedboltzmannmachine.html) 가이드를 확인하시기 바랍니다.

새로운 프로젝트를 만드는 경우에 설계에 필요한 [POM 디펜던시](http://nd4j.org/kr-dependencies.html)를 include하시려면 [ND4J Getting Started](http://nd4j.org/kr-getstarted.html)의 설명을 참고하십시오.
