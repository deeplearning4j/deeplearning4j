---
title: 
layout: ja-default
---

#Video: DL4Jで簡単にサンプルを実行するためには

<iframe width="750" height="560" src="//www.youtube.com/embed/2lwsHKUrXMk" frameborder="0" allowfullscreen></iframe>

#クイックスタート

* まずはじめに、 どのバージョンのJavaを持っているかチェックをします。以下のコードをコマンドラインに入力することで確認することができます。:

		java -version

* もしJava　7を持ていなかった場合は、[コチラ](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html)のサイトからダウンロードを行ってください。ダウンロード方法は、それぞれのオペレーティングシステムによって異なります。最新のMacの方はMac OS Xと記載のあるところから、ダウンロードをしてください。 (この*jdk-7u* という数字に続く部分が、アップグレードの状態を示しております)。以下がその表示内容となります。:

		Mac OS X x64 185.94 MB -  jdk-7u67-macosx-x64.dmg

*　Jblasの信頼性を確保するためのに、Blasの初期設定が必要となります。
		OSX
		Already Installed
		
		Fedora/RHEL
		yum -y install blas

		Ubuntu
		apt-get install libblas* (credit to @sujitpal)

		Windows
		See http://icl.cs.utk.edu/lapack-for-windows/lapack/

* DL4Jはデータの可視化とデバッグにクロスプラットフォームツールを活用しているため、 [Anaconda](http://continuum.io/downloads)が必要となります。一度Anacondaをインストールすると、[Python](https://ja.wikipedia.org/wiki/Python)に以下のコードを入力することで必要なライブラリがコンピュータに入っているか、確認することができます。:

		import numpy
		import pylab as pl

![Alt text](../img/python_shot.png)

これらのツールは、ニューラルネットをデバックする際の可視化を高めることができます。通常の統計分布は正常に機能しているサインとなります。可視化を進めることで、Macをお使いの方のコンピュータには、エラーのリストが表示されることがありますが、これはトレーニングの中断を意味するものではありません。

*次に、DL4Jのサンプルgit cloneしてください。:

		git clone https://github.com/SkymindIO/dl4j-examples

ここでMavenプロジェクトをこれらのIDEへ一つ一つインポートすることができます。
[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html),  [Intellij](https://www.jetbrains.com/idea/help/importing-project-from-maven-model.html) もしくは [Netbeans](http://wiki.netbeans.org/MavenBestPractices).

* IntelliJにあるdl4jのサンプルを開き、 MNISTサンプルを実行してください。もしMNISTのデモがたくさんのレンダリングを表示し、その速度が落ちてしまっている場合、スピードを適切な形でコントロールし、ファイルを保存したうえでリスタートすることができます。

* ここでニューラルネットがコマンドラインのターミナル上で、作業を始めたのが確認できるかと思います。ニューラルネットは計算の過程で、自動的にターミナルウィンドウをスクロールしていきます。(情報のソースによっては、スタートするまでに時間がかかることがございます。) そして、右端から二番目に表示されている数字が、計算を重ねるごとに減っていくことが確認いただけるかと思います。この数字はニューラルネットのエラーの確率を示しており、この数字が減っているということは「学習している」ことを意味します。

![Alt text](../img/learning.png)

* 計算している途中で、デスクトップの左側にポップアップされる、ニューラルイメージをご確認いただけるかと思います。このイメージでニューラルネットが実際に作業を進められているかの確認をすることができます。イメージはこのような形のものが表示されます。

![Alt text](../img/numeral_reconstructions.png)

MNISTのデータセットをニューラルネットが学習しているかどうかは、これらのイメージをもとに判断することができます。これらのイメージは回数を重ねるごとに、手書きの文字に近づいていきます。もし、イメージで実際に進捗が確認することができれば、それ以降もニューラルネットは自動的に学習を続け、精度が高まっていきます。

もし正常に作業が進んでいない場合には、[コチラ](https://groups.google.com/forum/#!forum/deeplearning4j))までお問い合わせください。

**次のステップ**:  [コチラ](../runexample.html)がどのようにサンプルを実行するかの、チュートリアルのページになります。  [コチラ](https://github.com/SkymindIO/dl4j-examples/tree/master/src/main/java/org/deeplearning4j) のページをクリックすることで、次に実行するサンプルも探すことができます。 (これらのサンプルはMNIST, Iris,そしてLabeled Faces in the Wildに対応しております。)

もしすべてのサンプルを実行し終わった際には、[コチラ](../ja-gettingstarted.html)のページで、すべてのコードベースを取得することも可能です。
