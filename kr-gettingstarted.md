---
layout: kr-default
---

# 전체 설치

이것은 다단계 설치 입니다. 질문이나 피드백이 있으시다면 저희가 상세한 설명을 드릴 수 있드록 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 강력 추천드립니다. 만약 여러분께서 비사교적 또는 완전 독립적인 성격이시라면 여러분은 자율 학습 하실 수 있도록 초청 되셨습니다. 추가적으로 여러분께서 딥 러닝에 완전히 새로우시다면, 저희는 [여러분께서 시작하실 때 배워야 할 것들에 대한 로드맵](http://deeplearning4j.org/deeplearningforbeginners.html)을 가지고 있습니다.

몇 단계를 통해 저희의 예제들을 실행하시려면 지금 [퀵 스타트 페이지](http://nd4j.org/kr-index.html)로 이동하시기 바랍니다. 예제들을 수행 하시기 전에 여러분께서 이곳을 방문하시기를 진심으로 바랍니다. 이는 DL4J를 시작하는 쉬운 방법 입니다.

Deeplearning4j를 위한 prerequisites 설치는 DL4J의 신경망에 동력을 지원하는 선형 대수학 엔진, ND4J의 [“Getting Started” 페이지](http://nd4j.org/kr-getstarted.html)에 문서화 되어 있습니다:

1. [Java 7 혹은 최신](http://nd4j.org/kr-getstarted.html#java)
2. [통합 개발 환경(Integrated Development Environment): IntelliJ](http://nd4j.org/kr-getstarted.html#ide)
3. [Maven](http://nd4j.org/kr-getstarted.html#maven)

위의 설치를 마치신 후, 다음을 읽어주시기 바랍니다.

1. OS-OS-구체적인 안내:
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
2. [GitHub](http://nd4j.org/kr-getstarted.html#github)
3. <a href="#eclipse">Eclipse</a>
4. <a href="#trouble">문제 해결</a>
5. <a href="#results">재생 가능한 결과(Reproducible Results)</a>
6. <a href="#next">다음 단계</a>

### <a name="linux">Linux</a>

* CPUs를 위한 Blas의 다양한 양식에 대한 저희의 확신으로 인해, Blas를 위한 기본 바인딩(native bindings)이 필요합니다.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

OpenBlas에 대한 더 많은 정보를 위해서는 [이 섹션](http://nd4j.org/kr-getstarted.html#open)을 봐 주시기 바랍니다.

* 만약 GPUs가 파손된 경우, 추가의 커맨드를 입력하셔야 합니다. 우선, Cuda 자체 설치 위치를 확인하십시오. 이는 다음과 같을 것 입니다.

		/usr/local/cuda/lib64

그 다음 Cuda를 연결하는 파일 경로 다음의 터미널에 *Idconfig*를 입력하십시오. 여러분의 커맨드는 다음과 같을 것 입니다.

		ldconfig /usr/local/cuda/lib64

만약 여전히 Jcublas를 로드할 수 없다면, 여러분의 코드에 파라미터 -D를 추가하셔야 합니다. (이는 JVM 인수 입니다):

     	java.library.path (settable via -Djava.librarypath=...) 
     	// ^ for a writable directory, then 
     	-D appended directly to "<OTHER ARGS>" 

여러분의 IDE로서 IntelliJ를 사용하고 계시다면, 이는 이미 작동되고 있어야 합니다.

### <a name="osx">OSX</a>

* Blas가 이미 OSX에 설치되어 있습니다.

### <a name="windows">Windows</a>

* 저희의 Windows 설치는 항상 쉬운 것은 아닌 반면, Deeplearning4j 실제로 윈도우 커뮤니티를 지원하기 위해 노력하는 몇 안되는 오픈 소스 딥 러닝 프로젝트 중 하나 입니다. 자세한 안내를 위해서는 저희의 [ND4J 페이지의 윈도우 섹션](http://nd4j.org/kr-getstarted.html#windows)을 참조하십시오.

* 64-bit 컴퓨터를 가지고 계시더라도 [MinGW 32 bits](http://www.mingw.org)를 설치하시고 나서 (다운로드 버튼은 우측 상단에 있습니다), [Mingw를 사용한 Prebuilt dynamic 라이브러리](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)를 다운로드 하십시오.

* [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)을 설치하십시오. (Lapack이 여러분께서 Intel compiler를 가지고 계신지 질문할 것 입니다. 여러분은 가지고 계시지 않습니다.)

* Lapack은 [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)의 대안을 제공합니다. [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)를 위한 문서들도 확인하십시오.

* 대안으로서, MinGW를 무시하고 여러분의 PATH의 한 폴더에 그 Blas dll 파일을 복사하실 수 있습니다. 예를 들어, MinGW bin 폴더로의 경로는 이와 같습니다: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. Windows에서 PATH 변수에 대한 보다 많은 설명을 원하시면 [top answer on this StackOverflow 페이지](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)를 읽어보십시오.

* Cygwin은 지원되지 않습니다. **DOS Windows**에서 DL4J를 설치하셔야 합니다.

* [WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100)을 실행하는 것이 여러분의 Window 설치 디버그를 도울 수 있습니다. 여기에 무엇을 기대해야 하는지 보여주는 그것의 [출력의 한 예](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f)가 있습니다. 우선 그것을 다운로드 하신 후, 커맨드 창 / 터미널을 엽니다. 그것이 다운로드된 디렉토리에 `cd` 하십시오. `WindowsInfo`를 입력하고 enter를 누르십시오. 그것의 출력을 복사하려면 커맨드 창에서 오른쪽 마우스를 클릭 -> 전체 선택 -> enter를 누르십시오. 그러면 출력은 클립 보드에 있습니다.

**Windows**에서 OpenBlas (아래 참조)를 위해서는, 이 [파일](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1)을 다운로드 하십시오. `C:/BLAS`와 같은 어딘가에 압축을 푸십시오. 여러분의 시스템의 `PATH` 환경 변수에 그 디렉토리를 추가하십시오.

### <a name="openblas">OpenBlas</a>

x86 백엔드 작업 위에 native libs가 있는지 확인하려면, 시스템 경로에 `/opt/OpenBLAS/lib`를 필요로 합니다. 그 후, 프롬프트에서 다음의 커맨드를 입력하십시오.

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3
			
저희는 [Spark](http://deeplearning4j.org/spark)가 OpenBlas와 작동하도록 이를 추가했습니다.

만약 OpenBlas가 제대로 작동하지 않으면, 아래의 단계들을 따르십시오.

* Openblas가 이미 설치되어 있다면 그것을 삭제하십시오.
* `sudo apt-get remove libopenblas-base`를 실행하십시오.
* OpenBLAS 개발 버전을 다운로드 하십시오.
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* **Linux**와는, 만약 `libblas.so.3`와 `liblapack.so.3`를 위한 symlinks가 여러분의 `LD_LIBRARY_PATH`의 어딘가에 존재한다면 더블 클릭하십시오. 존재하지 않는다면, `/usr/lib`에 링크들을 추가하십시오. Symlink는 "symbolic link" 입니다. 여러분께서는 그것을 다음과 같이 설정하실 수 있습니다 (-s가 링크를 symbolic하게 만듭니다):

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"
* 위의 "from-here"이 아직 존재하지 않는, 여러분께서 생성하시는 symbolic 링크 입니다. 여기 [how to create a symlink](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux)에 StackOverflow가 있습니다. 또한 여기 [Linux man page](http://linux.die.net/man/1/ln)가 있습니다.
* 마지막 단계로서 여러분의 IDE를 재시작 하십시오. 
* Native Blas를 **Centos 6**와 작동하게 하는 방법에 대한 전체 설명을 위해서는 [이 페이지](https://gist.github.com/jarutis/912e2a4693accee42a94) 또는 [여기](https://gist.github.com/sato-cloudian/a42892c4235e82c27d0d)를 보십시오.

**Ubuntu** (15.10) 상의 OpenBlas를 위해서는, [여기 안내](http://pastebin.com/F0Rv2uEk)를 보시기 바랍니다.				
### <a name="eclipse">Eclipse</a>

*git clone*을 실행하신 후, 다음의 커맨드를 입력하십시오.

		mvn eclipse:eclipse 
		
이는 그 소스를 import 해 모든 설정을 완료할 것 입니다.

여러 해 동안 Eclipse를 사용한 후, 저희는 비슷한 인터페이스를 가지고 IntelliJ를 권장합니다. Eclipse의 단일식 아키텍처는 저희의 코드 및 다른 분들의 코드에서 이상한 오류를 발생시키는 경향이 있습니다.

Eclipse를 사용하는 경우, 여러분은 [Lombok plugin](https://projectlombok.org/)을 설치하실 필요가 있습니다. 또한 Eclipse를 위한 Maven plugin: [eclipse.org/m2e/](https://eclipse.org/m2e/)이 필요하실 것 입니다.

Michael Depies은 [Eclipse 상에서 Deeplearning4j 설치하기](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/)로의 가이드를 작성했습니다.

### <a name="trouble">문제 해결</a>

* 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)을 통해 오류 메시지에 대해 문의해주십시오. 질문을 게시하실 때에는 다음의 정보를 준비해주시기 바랍니다 (처리가 엄청 빨라집니다!):

      * Operating System (Windows, OSX, Linux) and version 
      * Java version (7, 8) : type java -version in your terminal/CMD
      * Maven version : type mvn --version in your terminal/CMD
      * Stacktrace: Please past the error code on Gist and share the link with us: [https://gist.github.com/](https://gist.github.com/)
* 이전에 DL4J를 설치하셨고 이제 오류를 일으키는 예제들을 보고 계신다면, 여러분의 라이브러리를 업데이트 하십시오. Maven과는 [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j) 상의 최신의 버전들과 상응하게 하기 위해 여러분의 POM.xml 파일에 있는 그 버전들만을 업데이트 하십시오. 소스와는 [ND4J](http://nd4j.org/kr-getstarted.html), Canova 및 DL4J 상의 `git clone`을, 3개 모두의 디렉터리 내에서는 `mvn clean install -Dskiptests=true -Dmaven.javadoc.skip=true`를 순서대로 실행하십시오.
* 어떤 예제를 실행할 때 망(net)의 분류가 정확한지의 확률인 [f1 점수](http://deeplearning4j.org/glossary.html#f1)를 낮게 받으실 수 있습니다. 이 경우, 낮은 f1 점수가 성능 저하를 의미하지 않습니다. 왜냐하면 예제들은 작은 데이터 세트에서 훈련되었기 때문입니다. 저희는 빠른 실행을 위해 예제들에 작은 데이터 세트를 주었습니다. 작은 데이터 세트들은 큰 데이터 세트들보다 덜 대표적이기 때문에, 그 보여지는 결과들은 크게 다를 수 있습니다. 예를 들어, 소문자 예제 데이터에서 저희의 심층 신뢰 망(deep-belief net)의 f1 점수는 현재 0.32 에서 1.0 사이에서 차이를 보입니다.
* Deeplearning4J는 **autocomplete function**을 포함합니다. 어떤 커맨드들이 사용 가능한지 확신이 없는 경우 어떤 문자든 누르면 다음과 같이 드롭다운 리스트가 보여질 것 입니다:
![Alt text](../img/dl4j_autocomplete.png)
* 여기에 모든 [Deeplearning4j’s classes and methods](http://deeplearning4j.org/doc/)를 위한 **Javadoc**이 있습니다.
* 코드 베이스가 증가함에 따라 소스로부터의 설치는 더 많은 메모리를 필요로 합니다. DL4J 구축 시 `Permgen 오류`를 경험하신다면 더 많은 **heap space**를 추가하셔야 할 수 있습니다. 이를 위해 여러분의 숨겨진 `.bash_profile` file을 찾아 변경 하십시오. 이는 bash에 환경 변수들을 추가해 줄 것 입니다. 이 변수들을 보시려면, 커맨드 라인에 `env`를 입력하십시오. 더 많은 heap space를 추가하시려면 여러분의 콘솔에 다음의 커맨드를 입력하시기 바랍니다: 
		echo “export MAVEN_OPTS=”-Xmx512m -XX:MaxPermSize=512m”” > ~/.bash_profile
* 3.0.4와 같은 Maven의 이전 버전들은 NoSuchMethodError와 같은 예외 사항들을 제공할 가능성이 있습니다. 이는 Maven의 최신 버전으로 업그레이드 함으로써 해결될 수 있습니다. Maven 버전을 확인하시려면 여러분의 커맨드 라인에 `mvn -v`를 입력하십시오.
* Maven을 설치하신 후 여러분께서는 다음과 같은 메시지를 받으실 수 있습니다: `mvn is not recognised as an internal or external command, operable program or batch file.` 이는 여러분께서 어떤 다른 환경 변수처럼 변경하실 수 있는 [PATH variable](https://www.java.com/en/download/help/path.xml)에 Maven이 필요하다는 것을 의미합니다.
* 만약 `Invalid JDK version in profile 'java8-and-higher': Unbounded range: [1.8, for project com.github.jai-imageio:jai-imageio-core com.github.jai-imageio:jai-imageio-core:jar:1.3.0`와 같은 오류를 보신다면, 여러분은 Maven 문제가 있을 것 입니다. 버전 3.3.x으로 업데이트 하십시오.
* 일부 ND4J 디펜던시를 컴파일 하시려면, C 또는 C++를 위한 몇 가지 **개발 도구**들을 설치하셔야 합니다. [저희의 ND4J guide를 보십시오](http://nd4j.org/kr-getstarted.html#devtools).
* [Java CPP](https://github.com/bytedeco/javacpp)를 위한 include path가 항상 **Windows**에서 작동하지는 않습니다. 한가지 해결 방법은 Visual Studio의 include directory로부터 header 파일들을 가져와 Java가 설치되어 있는 Java Run-Time Environment (JRE)의 include directory에 붙여 넣는 것 입니다. (이는 standardio.h와 같은 파일들에 영향을 미칠 것 입니다.) 더 많은 정보는 [여기](http://nd4j.org/kr-getstarted.html#windows)에 있습니다.
* 여러분의 GPUs를 모니터링 하기를 위한 정보는 [여기](http://nd4j.org/kr-getstarted.html#gpu)에 있습니다.
* Java를 사용하는 한가지 주요 이유는 **[JVisualVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jvisualvm.html)**내에 이의 사전 만들어진(pre-baked) diagnostics 입니다. 여러분께서 Java를 설치하셨다면 여러분의 커맨드 라인에 단지 `jvisualvm`를 입력하면 여러분의 CPU, Heap, PermGen, Classes 및 Threads 상에 비주얼을 얻으실 수 있습니다. 한가지 유용한 보기: 우측 상단에 있는 `Sampler` 탭을 클릭하시고, 비주얼을 위해 CPU 또는 Memory 버튼을 선택하십시오. 
![Alt text](../img/jvisualvm.png)
* DL4J를 사용하면서 직면하는 몇가지 문제들은 기계 학습의 아이디어 및 테크닉에 익숙하지 않아서 일 수 있습니다. 저희는 모든 Deeplearning4j 사용자들께서 기초를 이해하기 위해 이 웹사이트를 넘어선 리소스에 의존하시기를 강력히 추천합니다. 저희는 [이 페이지](../deeplearningpapers.html)에 기계 및 딥 러닝을 위한 교육적인 리소스의 리스트를 포함시켰습니다. 저희가 부분적으로 DL4J를 문서화해 온 반면, 코드의 일부는 본질적으로 딥 러닝을 위한 완성되지 않은, 도메인 특정의 언어로 남겨져 있습니다.
* **Clojure** application으로부터 `deeplearning4j-nlp`을 사용하고 Leiningen으로 uberjar를 작성할 때, `project.clj`에서 akka `reference.conf` 리소스 파일들이 제대로 되도록 다음을 특정화 하는 것이 필요합니다. `:uberjar-merge-with {#"\.properties$" [slurp str spit] "reference.conf" [slurp str spit]}`. .properties 파일들을 위한 맵에서 첫번째 입력이 보통 기본 설정 임을 기억하십시오. 만약 이것이 되지 않으면, 결과인 uberjar로부터 실행하려고 할 때 다음의 예외 사항이 나타날 수 있습니다: `Exception in thread "main" com.typesafe.config.ConfigException$Missing: No configuration setting found for key 'akka.version'`.
* Float 지원은 OSX에서 버그가 있습니다. 만야 여러분께서 저희의 예제를 실행하면서 숫자를 예상하고 계시는 곳에 NANs을 보신다면, 그 데이터 형식을 `double`로 전환하십시오.
* Java 7에서 fork-join에 버그가 있습니다. Java 8로 업데이트 하시면 그것을 해결하실 수 있습니다. 만약 이와 같은 OutofMemory 오류를 보신다면 fork join이 그 문제 입니다: `java.util.concurrent.ExecutionException: java.lang.OutOfMemoryError`
.... `java.util.concurrent.ForkJoinTask.getThrowableException(ForkJoinTask.java:536)`

### <a name="results">재생 가능한 결과</a>

신경망 가중치는 임의로 초기화 됩니다. 이는 모델이 매번 중량 공간에서 다른 위치로부터 학습을 시작해 다른 로컬 최적 조건으로 이끌어질 수 있슴을 의미합니다. 재생 가능한 결과를 원하시는 이용자께서는 동일한 임의의 가중치를 사용할 필요가 있으며 이 가중치는 모델이 생성되기 이전에 초기화 되어야 합니다. 동일한 임의 가중치는 다음의 라인으로 재초기화 될 수 있습니다:

		Nd4j.getRandom().setSeed(123);

### <a name="next">다음 단계: IRIS 예제와 NNs 구축하기</a>

신경망 구축을 시작하기 위해서는 더 많은 정보를 위해 [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html)를 확인하시기 바랍니다.

빨리 배우시려면 [IRIS tutorial](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)을, *deep-belief networks*의 기본 매커니즘을 이해하시려면 [restricted Boltzmann machines](../restrictedboltzmannmachine.html)을 위한 저희의 가이드를 확인하시기 바랍니다.

새로운 프로젝트를 시작하고, 필요한 [POM 디펜던시](http://nd4j.org/kr-dependencies.html)를 포함하시려면 [ND4J Getting Started](http://nd4j.org/kr-getstarted.html)의 설명을 따르십시오.
