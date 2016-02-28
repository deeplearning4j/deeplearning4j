---
title: "Deeplearning 4 j のクイックスタートガイド"
layout: default
---

クイック・スタート・ガイド
=========================================

## 必要なもの

このクイックスタートガイドには、次のものがすでにインストールされていることを前提としています。

1. Java 7、またはそれ以降
2.　IntelliJ （または別の種類のIDE）
3.　Maven （自動ビルドツール）
4.　Github
 
上記のどれかを新たにインストールする必要があれば、ガイドの[ND4Jを「はじめましょう」](http://nd4j.org/getstarted.html)をご参照ください。（ND4Jは、ディープラーニングを実行させるために使う科学的計算エンジンで、上記のガイドは、DL4Jにもお使いいただけるものです。）ガイドにリストされたものをインストールすれば、それで十分でそれ以外をインストールする必要はありません。 


質問やコメントなどございましたら、弊社の[Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)に是非お問合せください。恥ずかしがる必要は全くありません。いつでも気軽にご連絡ください。また、ディープラーニングの初心者の方には、「ディープラーニング初心者ガイド」も[こちら](../deeplearningforbeginners.html)にご用意いたしました。 

Deeplearning4jは、プロのJava開発者向けのオープンソースプロジェクトで、製品展開、Intellijなどの統合開発環境（IDE）、Mavenのような自動ビルドツールなどに精通した方々を対象としています。既にこれらのツールをお持ちの方には、弊社のツールは、非常に役に立ちます。

## DL4Jの簡単な使い方ステップ

上記をインストールした後、以下のステップを踏んでいただくと、すぐにお使いいただけます。（Windowsのユーザーの方は、このページ下方の[ステップごとの手順](#walk)をお読みください。）

* コマンドラインに`git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git`と入力します。（現在使用中のexampleバージョンは0.0.4です。）
* IntelliJを開き、Mavenを使ってメニューツリーの`File/New/Project from Existing Sources`へ行き、新しいプロジェクトを作成します。上記のexampleのルートディレクトリを指定すると、統合開発環境でexampleが開きます。
![Alt text](../img/IntelliJ_New_Project.png) 
* 以下のコードをPOM.xmlにコピー＆ペーストし、[こちら](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)のようにします。 
* 追加の[Windowsユーザー向け手引きは、こちらをお読みください](../gettingstarted.html#windows)。 
* 左側のファイルツリーから`DBNIrisExample.java`を選びます。
* runを押すと、完了です！（ソースファイルを右クリックしたときに表示される緑色のボタンです。)

### 管理された環境

Databricks、Domino、 Sense.ioなどの管理された環境で作業している場合、もう1つすべきことがあります。 上述のローカルセットアップに従った後、exampleのディレクトリ内から以下のコマンドを実行してください。 

		mvn clean package

その後、ご使用の環境にJARファイルをアップロードします。 

### 注意事項

* 他のレポジトリをローカルにクローンしないようにしてください。メインのdeeplearning4jレポジトリは、改善し続けているため、最新のものは様々なexampleを使って完全に検証し終えていない恐れがあります。
* exampleのすべての依存関係は、ローカルでなくMavenからダウンロードするようにしてください。`(rm -rf  ls ~/.m2/repository/org/deeplearning4j)`
* dl4j-0.4-exampleのディレクトリで`mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true`を実行し、正常にインストールされているか確認してください。
* TSNEについては、`mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.tsne.TSNEStandardExample" -Dexec.cleanupDaemonThreads=false`と実行し、TSNE、または他のexampleを実行します。実行に失敗し、 Mavenのデーモンスレッドが終了時に停止しない場合には、最後に引数が必要になる場合があります。
* 1000回のイテレーションは、`dl4j-0.4-examples/target/archive-tmp/`に配置された`tsne-standard-coords.csv`に出力されるはずです。

F１スコアは、約0.66と出るはずですが、Irisのような小さなデータベースでは問題ありません。exampleのステップごとの手順は、弊社の[Iris DBNチュートリアル](../iris-flower-dataset-tutorial.html)をお読みください。

何か問題が発生したら、まずはPOM.xmlファイルが、[こちらの正しい例](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)のようになっているか、確認してください。 

## 依存関係およびバックエンド

バックエンドとは、DL4Jのニューラルネットワークが利用する線形代数ライブラリの処理基盤です。バックエンドはチップによって異なります。CPUはx86で、GPUはJcublasで最も高速に動作します。すべてのバックエンドを[Maven Central](https://search.maven.org)で見つけることができます。 「Latest Version」にある最新バージョン番号をクリックし、次の画面の左側にあるdependencyコードをコピーし、プロジェクトルートのpom.xmlにペーストします。 

nd4j-x86のバックエンドは、以下のようになります。

     <dependency>
       <groupId>org.nd4j</groupId>
       < artifactId > nd 4 j x 86 < / artifactId >
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-x86*はすべてのexampleで動作します。さらに依存関係をインストールするには、OpenBlas、Windows、Linuxのユーザーは[Deepelearining4jをはじめましょう](../gettingstarted.html#open)をお読みください。

## 上級レベル： AWSでのコマンドラインの使用

AWSサーバーでLinux OSにDeeplearningをインストールし、最初のexampleを実行させるためにIDEに頼らず、コマンドラインを使用したい場合は、 上述の指示に従って、*git clone*、*mvn clean install*を実行してください。インストールが完了すると、実際のexampleをコマンドラインに1行のコードで実行できます。コマンドラインは、レポジトリバージョンや特定のexmpleによって異なります。 

以下はテンプレートです。

    java -cp target/nameofjar.jar fully.qualified.class.name

そして具体的にはコマンドは大体以下のようになります。

    java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample

つまり、更新すると変更するワイルドカードが2つあり、以下のようなexampleになります。

    java -cp target/*.jar org.deeplearning4j.*

コマンドラインのexampleを変更して、変更したファイルを実行するには、例えば、*src/main/java/org/deeplearning4j/multilayer*の*MLPBackpropIrisExample*を調整し、examplesを再びMavenで構築します。 

## Scala 

[Scalaバージョンでの例はこちら](https://github.com/kogecoo/dl4j-0.4-examples-scala)。

## 次のステップ

exampleを実行し終えた後は、 [フルインストール・ページ](../gettingstarted.html)をお読みいただくと詳細を知ることができます。 

## <a name="walk">ステップごとの手順</a>

* gitが既にインストールされている場合は、以下のコマンドを入力します。

		git --version 

* gitがまだインストールされていない場合は、[git](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git)をインストールします。 
* また、[Githubのアカウント]( https://github.com/join)を作成し、GitHubの[Mac版](https://mac.github.com/)、または[Windows版](https://windows.github.com/)をダウンロードします。 
* Windowsをご使用の場合、スタートアップメニューで「Git Bash」を探して開きます。Git Bashターミナルは、cmd.exeのようなものです。
* DL4Jのexampleを配置したいディレクトリに`cd`コマンドを実行します。新しいものを`mkdir dl4j-examples`で作成し、`cd`コマンドをそこに入力します。そして以下を実行します。

    `git clone https://github.com/deeplearning4j/dl4j-0.4-examples`
* `ls`コマンドを実行して必ずファイルをダウンロードするようにしてください。 
* 次にIntelliJを開きます。 
* 「File（ファイル）」メニューをクリックし、「Import Project（プロジェクトをインポート）」または「New Project from Existing Sources（既存のソースからの新規プロジェクト）」を選びます。これにより、ローカルのファイルメニューが提供されます。 
* DL4Jのexampleが含まれているディレクトリを選択します。 
* ビルドツールの選択画面が表示されます。Mavenを選択します。 
* 「Search for projects recursively（再帰的にプロジェクトを検索）」と「Import Maven projects automatically（自動的にMavenのプロジェクトをインポート）」にあるチェックボックスにチェックを入れ、「Next（次へ）」をクリックします。 
* JDK/SDKが設定されていることを確認します。これらが設定されていない場合、SDKウィンドウの下方にあるプラス記号（＋）をクリックします。 
* それから、プロジェクト名を指定するよう指示があるまでクリックし続けます。デフォルトのプロジェクト名はそのままで問題ないはずなので、「Finish（終了）」ボタンを押すだけで完了です。
