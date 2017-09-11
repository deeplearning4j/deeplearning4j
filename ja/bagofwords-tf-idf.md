---
title:Bag of Words - TF-IDF
layout: default
---

# Bag of Words（単語の袋） & TF-IDF 

[Bag of Words（単語の袋）](https://en.wikipedia.org/wiki/Bag-of-words_model)（以下BoW）とは、ある文書における単語の出現回数を数えるアルゴリズムです。単語数を数えることにより、複数の文書を比較したり類似性を測定することができるため、検索、文書の分類、トピックモデリングなどのアプリケーションに役立てることができます。BoWは、ディープラーニングネットワークの入力テキストを準備するための一手段です。 

BoWは、各文書の単語数を一覧表で表示します。この表では、単語と文書がベクトル化という効果的な方法で保存されています。行見出しには単語、列見出しには文書、各セルには単語数が入力されています。コーパス内の文書は、等しい長さの列で表示されます。これらはwordcount（単語の出現回数）ベクトルといって、コンテキストが取り除かれた出力データです。 

![Alt text](../img/wordcount-table.png) 

ニューラルネットワークにそれらが入力される前に、wordcountsの各ベクトルが正規化されます。ベクトルのすべての要素が合計して1になります。このため、各単語の出現頻度はその文書内で単語が出現する確率を表すものへと効果的な形で変換されます。この確率があるレベルを超えると、ネットワークのノードが活性化され、その文書の分類結果に影響が出てきます。 

### 単語の出現頻度 - 逆文書頻度（TF-IDF）

[単語の出現頻度 - 逆文書頻度](https://ja.wikipedia.org/wiki/Tf-idf)（Term-frequency-inverse document frequency。以下、TF-IDF）とは、ある記事のトピックを判断する方法をその記事に含まれる単語を使って行うというものです。TF-IDFでは、単語に重みが付与され、頻度ではなく関連性を測定します。つまり、データセット全体のwordcountがTF-IDFのスコアに置き換えられます。 

まず最初に、TF-IDFはある文書に単語が出現する回数を測定しますが、これがTF（単語の出現頻度）に当たります。しかし、「and」や「the」などの単語はどの文書にも頻繁に出現するため、系統的に除外されます。この部分が、IDF（逆文書頻度）に当たります。ある単語の出現頻度が高ければ高いほど、その単語の信号としての価値は低下します。このように処理されるのは、頻繁に出現し、「かつ」特徴的な単語だけマーカー語として残すことを目的としているからです。各単語のTF-IDFの関連性は、正規化されたデータ形態で合計で1になります。 

![Alt text](../img/tfidf.png) 

そして、これらのマーカー語を含む文書のトピックを決めるために、これらの単語がニューラルネットワークに入力されます。

BoWのセットアップは以下のようなものになります。 

``` java
    public class BagOfWordsVectorizer extends BaseTextVectorizer {
      public BagOfWordsVectorizer(){}
      protected BagOfWordsVectorizer(VocabCache cache,
             TokenizerFactory tokenizerFactory,
             List<String> stopWords,
             int minWordFrequency,
             DocumentIterator docIter,
             SentenceIterator sentenceIterator,
             List<String> labels,
             InvertedIndex index,
             int batchSize,
             double sample,
             boolean stem,
             boolean cleanup) {
          super(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator,
              labels,index,batchSize,sample,stem,cleanup);
    }
```

TF-IDFは、そのシンプルさにも関わらず驚くほど強力で、Google検索などの有名で便利なツールに役立てられています。 



さて、ここでBoWと[Word2vec](./word2vec.html)との違いをご説明しましょう。Word2vecは1単語につきベクトル１つを産出しますが、BoWは数字（単語数）1つを産出するというのが主な違いです。Word2vecは文書内のコンテンツやコンテンツのサブセットを見極めるのに優れています。そのベクトルは各単語のコンテキストを表しており、nグラムがその一部です。BoWは、文書全体を分類するのに適しています。 

### <a name="beginner">その他のDeeplearning4jのチュートリアル</a>
* [Word2vec](https://deeplearning4j.org/ja/word2vec)
* [ディープニューラルネットワークについて](https://deeplearning4j.org/ja/neuralnet-overview)
* [制限付きボルツマン・マシンの初心者ガイド](https://deeplearning4j.org/ja/restrictedboltzmannmachine)
* [固有ベクトル、主成分分析、共分散、エントロピー入門](https://deeplearning4j.org/ja/eigenvector)
* [再帰型ネットワークと長・短期記憶についての初心者ガイド](https://deeplearning4j.org/ja/lstm)
* [回帰を使ったニューラルネットワーク](https://deeplearning4j.org/ja/linear-regression)
* [画像向けの畳み込みネットワーク](https://deeplearning4j.org/ja/convolutionalnets)

