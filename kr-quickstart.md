---
title:
layout: kr-default
---

# 퀵 스타트 가이드(Quick Start Guide)

## 전제 조건 (Prerequisites)

이 퀵 스타트 가이드는 이미 여러분이 아래를 설치하신 것으로 간주합니다:

1. Java 7 혹은 최신
2. IntelliJ (또는 다른 IDE)
3. Maven (자동화된 빌드 도구)
4. Github

위 중 어느 하나를 설치해야 하는 경우 [ND4J Getting Started guide](http://nd4j.org/kr-getstarted.html)를 참조하십시오. (ND4J는 저희가 딥 러닝 작업을 할 때 사용하는 과학 컴퓨팅 엔진이며, 그 설명들이 두 프로젝트들에 적용됩니다.) 예제들을 위해서는 그 페이지에 열거된 모든 것을 설치하지 마시고, 위에서 언급된 소프트웨어만을 설치하십시오.

질문이나 피드백이 있으시다면 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 추천드립니다. 심지어 반사회적이라 느껴지시더라도 잠복해 학습 하실 수 있습니다. 추가적으로 딥 러닝에 완전 초보자 이시다면, [시작하실때 무엇을 배워야 할지를 보여주는 로드맵](http://deeplearning4j.org/deeplearningforbeginners.html)을 포함해두었습니다.

Deeplearning4j는 IntelliJ와 같은 IDE 및 Maven과 같은 자동화 된 빌드 도구인 생산 배포에 익숙한 전문 자바 개발자를 대상으로 하는 오픈 소스 프로젝트 입니다. 만약 여러분이 이미 이 도구들에 익숙하시다면 저희의 도구가 여러분을 최상으로 도와드릴 것 입니다.

## DL4J를 위한 몇 가지 쉬운 절차

위 프로그램들을 설치 후, 이 단계들을 따르실 수 있다면, 여러분은 준비 완료 되셨습니다 (윈도우 사용자들은 아래 [Walkthrough](http://deeplearning4j.org/quickstart.html#walk) 섹션을 봐 주십시오):

* 여러분의 커맨드 라인에 git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git을 입력 합니다. (저희는 현재 예제 버전 0.0.4.x에 있습니다.)
* IntelliJ에서, Maven을 사용하여 새 프로젝트를 만들고, 위 예제들의 루트 디렉터리를 가리키도록 하십시오.
* 여러분의 POM.xml이 [이](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)와 같은 지 확인하기 위해 다음의 코드를 복사해 붙이십시오.
* 추가적인 [윈도우용 설명은 여기에 있습니다](http://deeplearning4j.org/gettingstarted.html#windows).
* Lefthand file tree에서 DBNIrisExample.java를 선택하십시오.
* 실행을 누르십시오! (소스 파일 위에서 오른쪽 마우스 클릭하시면 나타나는 녹색 버튼 입니다…)

## 몇 가지 주의사항
* 혹시 다른 저장소를 자체적으로 복제하지는 않았는지 확인하십시오. 주요 deeplearning4j 저장소는 지속적인 개선 중에 있습니다. 최신의 개선 항목들은 예제들과 완전하게 테스트 되지 않았을 수 있습니다.
* 여러분의 예제들을 위한 모든 디펜던시들이 자체로 찾으신 것이 아닌 Maven으로부터 다운로드 되었는지 확인하십시오. (rm -rf ls ~/.m2/repository/org/deeplearning4j)
* 제대로 설치되어 있는지 확인하기 위해 dl4j-0.4-examples 디렉터리에서 mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true를 실행하십시오.
* TSNE를 위해서는, TSNE 예제나 다른 예제를 실행하기 위해 mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.tsne.TSNEStandardExample" -Dexec.cleanupDaemonThreads=false 를 실행하십시오. 실행이 실패하거나 Maven이 종료 시 데몬 스레드를 중지 수 없는 경우 이 마지막 인수가 필요할 수 있습니다.
* 1000 반복은 tsne-standard-coords.csv가 dl4j-0.4-examples/target/archive-tmp/에 배치되도록 해야 합니다.

여러분은 Iris와 같은 작은 데이터 세트에 유용한 약 0.66의 F1 점수를 얻어야 합니다. 이 예제의 한 단계씩의 설명을 원하시면, 저희의 [Iris DBN 튜토리얼](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)을 참조하시기 바랍니다.

문제가 있을 때 가장 먼저 확인해야 하는 것은 [이](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)와 같이 보여질 여러분의 POM.xml 파일 입니다.

## 디펜던시와 백엔드(Dependencies and Backends)

백엔드는 DL4J 신경망 뒤의 선형 대수학 연산에 동력을 제공하는 것 입니다. 백엔드는 칩에 따라 달라집니다. CPUs는 x86과 함께, GPUs는 Jcublas와 함께 일 때 가장 빠르게 작동합니다. 여러분은 [Maven Central](https://search.maven.org)에서 모든 백엔드들을 찾을 수 있습니다; “Latest Version" 의 아래에 링크된 버전 번호를 클릭하고; 다음에 나오는 화면의 왼쪽에 있는 디펜던시 코드를 복사하고; 그것을 IntelliJ에서 여러분의 프로젝트 루트의 pom.xml 파일에 붙여 넣습니다.

nd4j-x86 백엔드는 아래과 같이 표시될 것입니다:

		 <dependency>
		   <groupId>org.nd4j</groupId>
		   <artifactId>nd4j-java</artifactId>
		   <version>${nd4j.version}</version>
		 </dependency>

*nd4j-x86*은 모든 예제들과 작동합니다. 추가적인 디펜던시를 설치하기 위해서, OpenBlas, 윈도우, 리눅스 사용자께서는 [Deeplearning4j Getting Started page](http://deeplearning4j.org/kr-gettingstarted.html)를 참조하셔야 합니다.

## 고급: AWS의 커맨드 라인(Command Line) 사용 하기

만약 Linux OS를 사용해 AWS 서버 상에서 Deeplearning4j를 설치하는 경우, 첫 번째 예제들을 실행 시 IDE에 의존하기 보다는 커맨드 라인을 사용하시기 바랍니다. 이 경우, 위의 설명에 따라 *git clones* 및 *mvn clean installs*를 실행하십시오. 이 설치가 완료되면, 커맨드 라인에서 한 라인의 코드로 실제 예제를 실행하실 수 있습니다. 이 라인은 repo 버전과 사용자가 선택하는 구체적인 예에 따라 달라질 수 있습니다.

템플릿은 다음과 같습니다:

      java -cp target/nameofjar.jar fully.qualified.class.name

여러분의 커맨드가 어떠한 모습인지 대략적으로 보여드리기 위한 구체적인 예가 여기에 있습니다:

      java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample

보시는 바와 같이 저희가 업데이트 하는대로 변경되고 여러분이 예제를 통해 확인하실 두개의 와일드 카드가 있습니다:

      java -cp target/*.jar org.deeplearning4j.*

커맨드 라인을 통해 예제들에 변경을 주고 그 변경된 파일을 실행하려면, 예를 들어, src/main/java/org/deeplearning4j/multiplayer에서 MLPBackpropIrisExample을 트윅 한 후, 그 예제들을 다시 maven-build 하실 수 있습니다.

## 스칼라 (Scala)

[예제들의 스칼라 버전들이 여기에 있습니다](https://github.com/kogecoo/dl4j-0.4-examples-scala).

## 다음 단계

예제들을 실행하신 후 더 탐구하고 싶으시다면 저희의 [Full Installation 페이지](http://deeplearning4j.org/gettingstarted.html)를 참조하십시오.

## 단계별 Walkthrough
* 여러분께서 Git을 가지고 계신 지 보기 위해 아래를 여러분의 커맨드 라인에 기입 합니다.
     git --version
* 만약 가지고 계시지 않다면, [git](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git)을 설치하시기 바랍니다.
* 추가로, [Github 계정](https://github.com/join)을 생성하시고, [맥](https://mac.github.com/)이나 [윈도우](https://windows.github.com/)용 GitHub을 다운로드 하십시오. 
* 윈도우를 위해서는, 여러분의 시작 메뉴에서 “Git Bash”를 찾으신 후 열어주십시오. 그 Git Bash 터미널은 cmd.exe와 비슷할 것 입니다.
* 여러분이 DL4J 예제들을 위치하기를 원하시는 디렉토리에 cd 하십시오. 여러분은 mkdir dl4j-examples로 새로운 것을 생성하고, 그곳에 cd 하기를 원하실 것 입니다. 그 후 다음을 실행하십시오:
	git clone https://github.com/deeplearning4j/dl4j-0.4-examples
* 그 파일들이 ls를 넣어 다운로드 되었는지 확인하십시오.
* 이제 IntelliJ를 열어주십시오.
* “File” 메뉴를 클릭하고, “Import Project” 혹은 “New Project from Existing Sources”를 클릭하십시오. 이것이 여러분께 로컬 파일 메뉴를 제공할 것 입니다.
* DL4J 예제들을 포함한 디렉토리를 선택하십시오.
* 다음의 창에서 여러분께서는 선택하실 수 있는 빌드 도구들을 보시게 될 것 입니다. Maven을 선택하십시오.
* “Search for projects recursively”와 “Import Maven projects automatically” 박스들을 체크하신 후, “Next”를 클릭하십시오.
* 여러분의 JDK/SDK가 설정되었는지 확인하시고, 안되어 있다면, 추가하기 위해 SDK 창의 바닥에 있는 플러스 기호를 클릭하십시오.
* 그 후 그 프로젝트의 이름을 정하기를 요청받을 때까지 클릭하십시오. 기본 선택된 프로젝트 이름으로 괜찮습니다. “Finish”를 눌러주십시오.
