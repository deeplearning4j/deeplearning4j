---
title:
layout: kr-default
---

# 퀵 스타트 가이드 (Quick Start Guide)

## 설치를 위한 필요사항 (Prerequisites)

이 퀵 스타트 가이드를 따라하시려면 먼저 아래의 네 가지 소프트웨어를 설치해야 합니다.

1. Java 7 혹은 상위 버전
2. IntelliJ (또는 다른 Java 통합 개발 환경)
3. Maven (빌드 자동화 도구)
4. Github

자세한 설치 안내는 [ND4J 시작하기](http://nd4j.org/kr-getstarted.html)를 참조하십시오 (ND4J는 DL4J의 딥러닝 작업에 사용되는 연산 엔진이며, ND4J 시작하기에 있는 안내는 ND4J와 DL4J에 모두 적용됩니다). 여기에서 소개된 예제를 실행하려면 ND4J 시작하기 페이지에 열거된 소프트웨어 중 위의 네 가지만 골라서 설치하면 됩니다.

저희가 운영하는 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) 페이지에서는 쉽고 빠르게 저희에게 질문을 하거나 피드백을 줄 수 있습니다. 채팅 페이지에서 다른 사람들의 질문/대화를 읽어보는 것 만으로도 DL4J에 대한 여러 가지를 배울 수 있을 겁니다. 만일 딥 러닝에 대해 전혀 아는 내용이 없으시면, [시작하실때 무엇을 배워야 할지를 보여주는 로드맵](http://deeplearning4j.org/deeplearningforbeginners.html) 페이지를 참고하시기 바랍니다.

Deeplearning4j는 IntelliJ나 Maven과 같은 IDE와 빌드 자동화 도구 사용에 익숙한 고급 자바 개발자를 대상으로 합니다. 만약 여러분이 이미 이런 소프트웨어의 사용에 익숙하시다면 DL4J를 사용하실 준비를 완벽하게 갖춘 셈 입니다.

## DL4J 빠르게 둘러보기

위 소프트웨어를 설치한 뒤엔 아래의 단계를 따라 하시면 바로 딥러닝 코드를 실행하실 수 있습니다. 아래의 내용은 맥 사용을 가정하고 쓰여졌습니다. 윈도우 사용자들은 아래 [Walkthrough](http://deeplearning4j.org/quickstart.html#walk) 섹션을 참고하시기 바랍니다

* 터미널을 열고 `git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git`을 입력 합니다. (예제의 현재 버전은 0.0.4.x 입니다.)
* IntelliJ에서 File/New/Project from Existing Sources로 가서 위에서 클론한 폴더의 최상위 폴더로 가서 프로젝트를 엽니다.
* 이제부터 안내해드리는 코드를 복사/붙여넣기 하시면 여러분의 `POM.xml`이 [이 xml 문서](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)와 같은 지 확인하게 됩니다.
* [윈도우 사용자를 위한 추가 안내는 여기](http://deeplearning4j.org/gettingstarted.html#windows)를 참고하세요.
* 화면 왼쪽의 파일 트리에서 DBNIrisExample.java를 선택하십시오.
* 실행을 누르세요 (소스 파일 위에서 마우스 우클릭 후 나타나는 녹색 버튼을 누르시면 됩니다).

## 환경 설정
만일 Databricks, Domino, Sense.io같은 환경을 사용하신다면 추가적으로 아래의 코드를 터미널에서 실행해야 합니다. 터미널의 예제 디렉토리에서

```
		mvn clean package 
```

를 실행하세요. 그러면 사용중인 환경에 JAR파일을 업로드할 수 있습니다.

## 몇 가지 주의사항
* 예제를 실행하는 경우엔 다른 브랜치나 버전의 git 저장소를 복제하지 않도록 주의하세요. Deeplearning4j 저장소는 지속적으로 업데이트되고 있기 때문에 최신 버전의 코드는 이 예제와 호환되지 않을 수 있습니다.
* 반드시 Maven을 사용해서 모든 필요한 패키지(dependencies)를 다운받아야 합니다. `(rm -rf ls ~/.m2/repository/org/deeplearning4j)`
* 제대로 설치되어 있는지 확인하려면 dl4j-0.4-examples 디렉토리에서 `mvn clean pack clean install -DskipTests=true -Dmaven.javadoc.skip=true`를 실행하십시오.
* TSNE 예제 혹은 다른 예제를 실행하려면 `mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.tsne.TSNEStandardExample" -Dexec.cleanupDaemonThreads=false` 를 입력하십시오. 실행이 실패하거나 Maven 종료 후 데몬 스레드를 멈출 수 없는 경우 마지막 매개변수 `-Dexec.cleanupDaemonThreads=false`가 필요할 수 있습니다.
* TSNE 학습을 1000회 반복한 결과는 `dl4j-0.4-examples/target/archive-tmp/`의 `tsne-standard-coords.csv`를 확인하세요.

Iris 같은 작은 데이터 셋을 사용할 경우 F1-Score가 0.66정도 나와야 정상입니다. Iris 예제에 대한 더 자세한 설명은 [Iris DBN 튜토리얼](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)을 참조하시기 바랍니다.

만일 실행중에 문제가 생기면 우선 여러분의 POM.xml파일을 확인해 보세요. 정상적인 경우라면 [이 파일](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)과 비슷해야 합니다.

## 디펜던시와 백엔드(Dependencies and Backends)

백엔드의 역할은 DL4J 신경망 내부에 필요한 선형 대수 및 행렬 연산을 실제로 수행하는 것 입니다. 백엔드의 성능은 실제로 연산에 사용되는 하드웨어에 따라 달라집니다. CPU를 사용한 연산은 x86 백엔드를, GPU를 사용한 연산은 Jcublas를 사용할 경우 가장 빠르게 작동합니다. [Maven Central 페이지](https://search.maven.org)를 방문하시면 사용 가능한 백엔드 목록을 볼 수 있습니다; “Latest Version" 에 링크된 가장 최신 버전을 클릭하고 그 다음에 나오는 화면의 좌측의 디펜던시 코드를 복사한 뒤 그 코드를 IntelliJ에서 여러분의 프로젝트 루트의 pom.xml 파일에 붙여 넣으십시오.

nd4j-x86 백엔드는 아래과 같이 쓰여있습니다.

```

		 <dependency>
		   <groupId>org.nd4j</groupId>
		   <artifactId>nd4j-java</artifactId>
		   <version>${nd4j.version}</version>
		 </dependency>
```


*nd4j-x86*은 모든 예제들과 작동합니다. 추가적인 디펜던시를 설치하기 위해서, OpenBlas, 윈도우, 리눅스 사용자는 [Deeplearning4j Getting Started page](http://deeplearning4j.org/kr-gettingstarted.html)를 참조하시기 바랍니다.

## 고급: AWS의 커맨드 라인(Command Line) 사용 하기

만약 AWS에서 Linux OS를 사용해 Deeplearning4j를 설치하는 경우 커맨드 라인에서 예제를 실행할 수 있습니다. 우선 위의 설명에 따라 `*git clones*` 및 `*mvn clean installs*`를 실행하십시오. 설치가 완료되면  커맨드 라인에서 아래와 같은 명령어로 예제를 실행할 수 있습니다 (명령어는 repo 버전과 실행하고자 하는 예제에 따라 조금씩 달라질 수 있습니다).

명령어 템플릿은 다음과 같습니다:

```
      java -cp target/nameofjar.jar fully.qualified.class.name
```

예를 들면 아래와 같습니다.

```
      java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample
```

와일드 카드 *를 사용해 표현하면 아래와 같이 표현할 경우 자동으로 예제를 실행합니다.

```
      java -cp target/*.jar org.deeplearning4j.*
```

커맨드 라인을 통해 예제를 수정하고 수정된 파일을 다시 maven-build하여 예제를 실행할 수 있습니다. 예를 들어, `src/main/java/org/deeplearning4j/multiplayer`의 `MLPBackpropIrisExample`를 수정한 뒤 이 코드를 다시 maven-build 하실 수 있습니다.

## 스칼라 (Scala)

[예제의 스칼라 버전은 여기](https://github.com/kogecoo/dl4j-0.4-examples-scala)를 참고하십시오.

## 다음 단계

위의 예제보다 더 자세한 내용은 저희의 [Full Installation 페이지](http://deeplearning4j.org/gettingstarted.html)를 참고하십시오.

## 단계별 Walkthrough
* 여러분의 컴퓨터에 Git이 설치되어 있는지는 아래의 커맨드를 통해 확인할 수 있습니다.

```
     git --version
```

* 만일 Git이 설치되어 있지 않다면, [Git 설치 페이지](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git)를 참고하여 Git을 설치하시기 바랍니다.
* 그리고 [Github 계정](https://github.com/join)을 만드신 뒤 [맥](https://mac.github.com/)이나 [윈도우](https://windows.github.com/)용 GitHub을 다운로드 하십시오. 
* 윈도우에서는 시작 메뉴에서 “Git Bash”를 찾으신 후 열어주십시오. Git Bash 터미널은 윈도우의 커맨드 cmd.exe와 비슷한 모습의 터미널입니다.
* 터미널에서 `cd` 명령어를 이용해 DL4J 예제를 다운받을 디렉토리로 이동하십시오. 그리고 `mkdir dl4j-examples`로 새 디렉토리를 만들고 `cd dl4j-examples`를 입력하여 그 디렉토리로 이동하십시오. 그리고 아래의 명령어를 실행하십시오.

```
			git clone https://github.com/deeplearning4j/dl4j-0.4-examples
```

* `ls` 명령어로 파일이 전부 다운로드 되었는지 확인하십시오.
* 이제 IntelliJ를 실행하십시오.
* “File” 메뉴에서 “Import Project” 혹은 “New Project from Existing Sources”를 클릭하십시오.
* 위에서 만든 DL4J 예제 디렉토리를 이동하여 그 디렉토리를 선택하십시오.
* 이제 여러분이 선택할 수 있는 빌드 도구 목록을 보시게 될 것 입니다. Maven을 선택하십시오.
* “Search for projects recursively”와 “Import Maven projects automatically” 박스들을 체크하신 후, “Next”를 클릭하십시오.
* 여러분의 JDK/SDK가 설정되었는지 확인하십시오.  잘 되어있지 않다면 플러스 창 하단의 (+) 기호를 클릭하여 JDK/SDK를 추가하십시오.
* 그리고 계속 다음을 클릭하면 프로젝트 이름을 정하는 창이 나옵니다. 디폴트로 나온 프로젝트 이름을 사용하셔도 문제 없습니다. 그리고  “Finish”를 누르면 완료됩니다.
