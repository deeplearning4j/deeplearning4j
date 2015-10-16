---
title:
layout: default
---

# 퀵 스타트 가이드(Quick Start Guide)

## Prerequisites

이 퀵 스타트 가이드는 이미 여러분이 아래를 설치하신 것으로 간주합니다:

1. [Java 7](http://nd4j.org/kr-getstarted.html#java)
2. [IntelliJ (또는 다른 IDE)](http://nd4j.org/kr-getstarted.html#ide)
3. [Maven (자동화된 빌드 도구)](http://nd4j.org/kr-getstarted.html#maven)
4. [Github](http://nd4j.org/kr-getstarted.html#github)

위 중 어느 하나를 설치해야 하는 경우 [ND4J Getting Started guide](http://nd4j.org/kr-getstarted.html)를 참조하십시오. (ND4J는 Deeplearning4j에 동력을 지원하는 선형 대수학 엔진이며, 그 설명들이 두 프로젝트에 적용됩니다.) 그 페이지에 열거된 모든 것을 설치하지 마시고 위에서 언급된 소프트웨어만을 설치하십시오.

질문이나 피드백이 있으시다면 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 강력 추천드립니다. 심지어 일부 Java 프로그래머들도 Maven에 익숙하지 않을 수 있습니다... 비사교적 또는 완전 독립적인 성격이시라면 자율 학습 하실 수 있습니다.

## DL4J를 위한 몇 가지 쉬운 절차

위 프로그램들을 설치 후, 다음의 다섯 절차를 수행할 수 있다면 여러분은 준비 완료 되셨습니다:

1. git clone [nd4j](https://github.com/deeplearning4j/nd4j/), [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/), [canova](https://github.com/deeplearning4j/Canova)와 [예제들](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples). 현재 예제 버전은 0.0.3.3.x 입니다.

2. 여러분의 콘솔(console)로부터 각 디렉터리 상에서 "mvn clean install -DskipTests -Dmaven.javadoc.skip=true"를 실행하십시오.

3. 그 예제들을 하나의 프로젝트로서 Maven과 IntelliJ에 import 하십시오.

4. 예제들의 기본 백엔드(default [backend](http://nd4j.org/kr-dependencies.html))인 POM.xml은 nd4j-jblas로 설정되어 있습니다. (보다  쉬운 설치를 위해, Windows 사용자는 디펜던시들에서 이를 nd4j-java로 변경하십시오.)

5. Lefthand file tree에서 DBNIrisExample.java를 선택하십시오.

6. 실행을 누르십시오. (녹색 버튼!)

여러분은 약 0.66의 F1 점수를 얻어야 하고, 이는 Iris와 같은 작은 데이터 세트(dataset)에 좋습니다. 이 예제의 상세한 설명을 원하시면, 저희의 [Iris DBN 튜토리얼](http://deeplearning4j.org/iris-flower-dataset-tutorial.html)을 참조하시기 바랍니다.

## 디펜던시와 백엔드(Dependencies and Backends)

백엔드는 DL4J 신경망 뒤의 선형 대수학 연산에 동력을 제공합니다. 백엔드는 칩(chip)에 따라 달라집니다. CPU는 Jblas와 Netlib Blas와 함께; GPU는 Jcublas와 함께 가장 빠르게 작동합니다. 여러분은 [Maven Central](https://search.maven.org)에서 모든 백엔드들을 찾을 수 있습니다; “Latest Version" 의 아래에 링크된 버전 번호를 클릭하고; 다음에 나오는 화면의 왼쪽에 있는 디펜던시 코드를 복사하고; 그것을 IntelliJ에서 여러분의 프로젝트 루트의 pom.xml 파일에 붙여 넣습니다.

*nd4j-java* 백엔드는 아래과 같이 표시될 것입니다:

		 <dependency>
		   <groupId>org.nd4j</groupId>
		   <artifactId>nd4j-java</artifactId>
		   <version>${nd4j.version}</version>
		 </dependency>

*nd4j-java*는 Blas를 필요로 하지 않기 때문에 Windows에서 가장 쉬운 설정 입니다. 이는 DBNs 또는 심층 신뢰망(deep-belief nets)의 모든 예제들에 작동하나 그 외의 예제들에는 불가합니다.

*nd4j-jblas* 백엔드는 아래와 같이 표시될 것입니다.

		 <dependency>
		   <groupId>org.nd4j</groupId>
		   <artifactId>nd4j-jblas</artifactId>
		   <version>${nd4j.version}</version>
		 </dependency>

*nd4j-jblas*는 모든 예제들에 작동합니다. Jblas를 설치하려면 Windows 사용자께서는 [Deeplearning4j Getting Started page](http://deeplearning4j.org/kr-gettingstarted.html)를 참조하시기 바랍니다.

## 고급: AWS의 커맨드 라인(Command Line) 사용 하기

만약 Linux OS를 사용해 AWS 서버 상에서 Deeplearning4j를 설치하는 경우, 첫 번째 예제를 실행 시 IDE에 의존하기 보다는 커맨드 라인을 사용하시기 바랍니다. 이 경우, 위의 설명에 따라 git clones 및 mvn clean installs를 실행하십시오. 이 설치가 완료되면, 커맨드 라인에서 한 라인의 코드로 실제 예제를 실행하실 수 있습니다. 이 라인은 repo 버전과 사용자가 선택하는 구체적인 예에 따라 달라질 수 있습니다.

템플릿은 다음과 같습니다:

      java -cp target/nameofjar.jar fully.qualified.class.name

여러분의 커맨드가 어떠한 모습인지 대략적으로 보여드리기 위한 구체적인 예가 여기에 있습니다:

      java -cp target/deeplearning4j-examples-0.4-SNAPSHOT.jar org.deeplearning4j.MLPBackpropIrisExample

보시는 바와 같이 저희가 업데이트 하는대로 변경되고 여러분이 예제를 통해 확인하실 두개의 와일드 카드가 있습니다:

      java -cp target/*.jar org.deeplearning4j.*

커맨드 라인을 통해 예제들에 변경을 주고 그 변경된 파일을 실행하려면, 예를 들어, src/main/java/org/deeplearning4j/multiplayer에서 MLPBackpropIrisExample을 트윅 한 후, 그 예제들을 다시 maven-build 하실 수 있습니다.

## 다음 단계

예제들을 실행하신 후 더 탐구하고 싶으시다면 저희의 [Getting Started page](http://deeplearning4j.org/kr-gettingstarted.html)를 참조하십시오. 그리고 DL4J는 다단계 설치 임을 기억하시기 바랍니다. 질문이나 피드백이 있으시다면 저희가 상세한 설명을 드릴 수 있드록 저희의 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)에 가입하시기를 강력 추천드립니다. 비사교적 또는 완전 독립적인 성격이시라면 자율 학습 하실 수 있습니다.
