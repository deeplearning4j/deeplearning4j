---
title: 
layout: zh-default
---

# Word2vec

本网站正在更新中，如想要获得最新的信息，[请参考](../word2vec.html) 

目录

* <a href="#intro">简介</a>
* <a href="#anatomy">Word2Vec分析</a>
* <a href="#code">训练</a>
* <a href="#windows">滑动窗口</a>
* <a href="#grams">N-grams 和 Skip-grams</a>
* <a href="#load">加载你的数据</a>
* <a href="#trouble">疑难解答和调节Word2Vec</a>
* <a href="#dbn">细调DBNs (有代码示例)</a>
* <a href="#next">接下来</a>

## <a name="intro">Word2Vec简介</a>

Deeplearning4j 使用Java实现了一个分布式的Word2Vec，并且可以在GPU上运行。 Word2vec 最先是在Google由Tomas Mikolov带领的一个研究团队提出的。 

Word2vec是一个神经网络，它用来在使用深度学习算法之前预处理文本。它本身并没有实现深度学习，但是Word2Vec把文本变成深度学习能够理解的向量形式。

Word2vec在不需要人工干预的情况下创建特征，包括词的上下文特征。这些上下文来自于多个词的窗口。如果有足够多的数据，用法和上下文，Word2Vec能够基于这个词的出现情况高度精确的预测一个词的词义（对于深度学习来说，一个词的词义只是一个简单的信号，这个信号能用来对更大的实体分类；比如把一个文档分类到一个类别中）。

Word2vec 需要一串句子做为其输入。每个句子，也就是一个词的数组，被转换成n维向量空间中的一个向量并且可以和其它句子（词的数组）所转换成向量进行比较。在这个向量空间里，相关的词语和词组会出现在一起。把它们变成向量之后，我们可以一定程度的计算它们的相似度并且对其进行聚类。这些类别可以作为搜索，情感分析和推荐的基础。

Word2vec神经网络的输出是一个词表，每个词由一个向量来表示，这个向量可以做为深度神经网络的输入来进行分类。

DL4J采用了Mikolov提出而流行的skip-gram表示方法，这种表示方法因为更通用的上下文生成能力因而更加准确。

一般来说，我们通过余弦相似度来计算词语之间的相似度, 这个距离用来估计两个词向量之间的距离。90度代表完全相同，比如France（法国）和France（法国），而Spain（西班牙）和France（法国）的距离是0.678515，这是和France（法国）相似度最高的国家。

下面是使用Word2vec计算得出的和China（中国）相似的一些词画的图:

![Alt text](../img/word2vec.png) 

## <a name="anatomy">Word2vec分析</a>

当我们说到Word2Vec时候我们知道是什么呢？我们指的是Deeplearning4j 的自然语言处理组件，它包括:
* SentenceIterator/DocumentIterator: 用来迭代一个数据集. SentenceIterator返回一些字符串而DocumentIterator处理inputstreams. 如果可能尽量使用SentenceIterator.
* Tokenizer/TokenizerFactory: 用来分词。在NLP的术语，一个句子就是用一系列Token来表示。TokenizerFactory用来创建分词用的Tokenizer。
* VocabCache: 用来追踪元数据，包括词语数量，在文档中的出现，Token的集合（不是vocab，而是出现过的token），vocab（特征既包括Bag of word，也包括词向量查找表）
* 倒排索引: 关于词在哪里出现的元数据。可以用来帮助理解数据集。Lucene实现的索引会自动被创建。

[简单来说，使用梯度下降算法训练一个两层神经网络](http://deeplearning4j.org/glossary.html#downpoursgd)。神经网络连接权重的大小是特殊设计的。Word2vec术语中的syn0是wordvector查找表, syn1是激活，两层神经网络上的层级softmax训练用来计算不同词之间的相似度。Word2vec使用[skipgrams](http://deeplearning4j.org/glossary.html#skipgram)来实现。

## <a name="code">训练</a>

Word2Vec使用原始的文本进行训练。然后它会使用词向量来记录每个词的上下文，或者说是用法。训练结束后，它被当做一个查找表来使用，在很多自然语言处理任务中这个被查找表用来构成训练文本的窗口。

词形还原之后，Word2vec会根据你的句子数据自动进行多线程的训练。之后你可以保存模型。Word2vec有很多组件。其中之一就是vocab缓存。在 deeplearning4j里保存数据的常用方法是通过SerializationUtils (类似python的pickling的Java序列化方法)

        SerializationUtils.saveObject(vec, new File("mypath"));

上面的代码将把Word2vec保存到mypath。你也可以这样把模型重新加载到内存中：

        Word2Vec vec = SerializationUtils.readObject(new File("mypath"));

你可以像这样把Word2vec当成查找表来使用：

        INDArray wordVector = vec.getWordVectorMatrix("myword");
        double[] wordVector = vec.getWordVector("myword");

如果词表中没有这个词语，那么它会返回每项都是零的数组——没有别的信息。

## <a name="windows">窗口</a>

Word2Vec使用神经网络，利用滑动窗口模型通过词的共现来训练。下面是得到文本的窗口的两种方法：

        List<Window> windows = Windows.windows("some text");

上面的代码将从文本中选择5个token的滑动窗口（窗口的每个元素就是一个token）。

你也可以使用自定义的tokenizer(分词器)：

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
      List<Window> windows = Windows.windows("text",tokenizerFactory);

它会为文本创建一个tokenizer，并且基于这个分词器来滑动窗口。

      List<Window> windows = Windows.windows("text",tokenizerFactory);

下面的代码也会使用我们自定义的tokenizer，并且使用我们指定的windowSize来滑动窗口。

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
      List<Window> windows = Windows.windows("text",tokenizerFactory,windowSize);

[训练词序列模型是通过维特比算法来实现的](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)。

基本的思路是使用滑动窗口来训练Word2Vec然后把每个窗口（通过窗口的中心词）打上特定的标签。这可以通过词性(part-of-speech)标注，语义角色(semantic-role)标注，命名实体(named-entity)识别和一些其它任务来完成。

给定转移矩阵（从一个状态到另一个状态的跳转概率），维特比算法能计算最可能的事件(标签)序列。下面是一段示例代码：

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowExample.java?slice=112:121"></script>

在这里，需要处理的一行可能是这样样子：

        <ORGANIZATION> IBM </ORGANIZATION> invented a question-answering robot called <ROBOT>Watson</ROBOT>.

给定一些文本，Windows.windows会从尖括号包含的大写文本中自动的推测标签。

如果调用如下代码：

        String label = window.getLabel();

于包含那个窗口的任何东西上, 它都会自动包含那个标签。这被用来初始化(bootstrapping)一个训练数据上的标签集合的先验分布。

下面的代码将会保持你的维特比算法结果以备后用：

        SerializationUtils.saveObject(viterbi, new File("mypath"));

## <a name="grams">N-grams 和 Skip-grams</a>

词语被依次的读入向量，然后在一定范围内来回扫描，很像n-gram。（一个n-gram是一个语言学序列的n个元素组成的连续序列。n取不同的值就形成了一元语法，二元语法，三元语法，四元语法和五元语法）。

n-gram然后会被输入到神经网络中来学习给定词向量的重要性；重要性被定义成把它当成更大的语义或者标签的指示器的作用。

![enter image description here](http://i.imgur.com/SikQtsk.png)

Word2vec 使用两种窗口：连续的 n-grams和skip-grams。

考虑下面的句子：

    How’s the weather up there?

这个句子可以被分成如下一系列连续的三元语法。

    {“How’s”, “the”, “weather”}
    {“the”, “weather”, “up”}
    {“weather”, “up”, “there”}

它也可以变成一系列的skip-gram。

    {“How’s”, “the”, “up”}
    {“the”, “weather”, “there”}
    {“How’s”, “weather”, “up”}
    {“How’s”, “weather”, “there”}
    ...

skip-gram，就像你看到的，是一种不连续的n-gram。
在这里，你会经常看到“上下文窗口”的说法。在上面的例子中，上下文窗口是3。许多人也使用5作为上下文窗口。

## <a name="load">数据集</a>

在这个例子中，我们将使用路透新闻的一个很小的数据集。

在DL4J，你可以使用[UimaSentenceIterator](https://uima.apache.org/)来只能的加载你的数据。简单起见，我们将使用FileSentenceIterator。

### 加载你的数据

DL4J可以帮你轻松的加载一个文档语料库。在这个例子里，我们在用户的家目录下有一个叫 reuters的目录，包括许多文章。

考虑下面的代码：


    String reuters= System.getProperty("user.home") +             
    new String("/reuters/");
    File file = new File(reuters);
    
    SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
    @Override
    public String preProcess(String sentence) {
        return new 
        InputHomogenization(sentence).transform();
        }
    },file);

在第一行和第二行，我们得到一个指向‘reuters’的File对象。然后我们把它传给FileSentenceIterator。SentenceIterator是DL4J的Word2Vec非常重要的一个组件。它可以让我们轻松的扫描我们的数据，一次一行。

从第4-8行，我们预处理我们的数据（比如所有的词变成小写以及移除标点符号），这使得后续的处理更加容易。

## 准备创建Word2Vec对象

接下来我们需要如下代码

        TokenizerFactory t = new UimaTokenizerFactory();

概况来说，分词器(tokenizer)的输入是原始的没有处理的文本流，输出离散的，干净整洁的表示。我们把它叫做token，其实就是词。比如输入是：

    the|brown|fox   jumped|over####spider-man.

分词器会返回我们一个词或者token的列表，我们可以看到如下的列表。

1. the
2. brown
3. fox
4. jumped
5. over
6. spider-man

一个智能的分词器要能够识别spider-man中的-符号，知道它是名字中的一部分。

Uima指的是一个Apache项目——非结构化(Unstructured) 信息(Information) 管理(Management) 程序(Application)——它们帮我们处理非结构化的数据，就像tokenizer那样。事实上它就是一个智能的分词器。

### 创建 Word2Vec 对象

现在我们可以开始写代码来创建Word2Vec对象了。考虑下面的代码：

    Word2Vec vec = new Word2Vec.Builder().windowSize(5).layerSize(300).iterate(iter).tokenizerFactory(t).build();

这里我们可以通过一些参数创建一个word2Vec

    windowSize: 知道n-gram的n。 5是一个不错的默认值。
    iterate: 我们之前创建的SentenceIterator对象
    tokenizerFactory： 我们之前创建的UimaTokenizerFactory对象

在这之后设置其它你需要的参数也是不错的选择。
最后，我们可以开始通过我们的数据来训练我们的word2Vec。

    vec.fit();

大功告成。fit方法可能会花一段时间，当完成之后，你就可以随意的查询word2vec对象了。

    String oil = new String("oil");
    System.out.printf("%f\n", vec.similarity(oil, oil));

在上面的例子，你应该得到相似度的值为1。Word2Vec 使用余弦距离，相同向量的余弦距总是1。

下面是你可以调用的一些函数：

1. similarity(String, String) - 查询两个词的余弦相似度
2. analogyWords(String A, String B, String x) - 查询一个词，使得A和B的关系就像x和这个词的关系
3. wordsNearest(String A, int n) - 查询和 A 最相似的n个词。

## <a name="trouble">疑难解答和调节Word2Vec</a>

问：我的代码输出大量如下的stack trace：

       java.lang.StackOverflowError: null
       at java.lang.ref.Reference.<init>(Reference.java:254) ~[na:1.8.0_11]
       at java.lang.ref.WeakReference.<init>(WeakReference.java:69) ~[na:1.8.0_11]
       at java.io.ObjectStreamClass$WeakClassKey.<init>(ObjectStreamClass.java:2306) [na:1.8.0_11]
       at java.io.ObjectStreamClass.lookup(ObjectStreamClass.java:322) ~[na:1.8.0_11]
       at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1134) ~[na:1.8.0_11]
       at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548) ~[na:1.8.0_11]

答：看看你启动 Word2vec 程序的目录。这可能是一个IntelliJ项目的家目录或者是你在命令输入Java命令所在的目录。它应该有一些这样的文件：

       ehcache_auto_created2810726831714447871diskstore  
       ehcache_auto_created4727787669919058795diskstore
       ehcache_auto_created3883187579728988119diskstore  
       ehcache_auto_created9101229611634051478diskstore

你应该停止你的Word2Vec程序然后删除它们。

问：不是原始文本数据中的所有词都出现在我的word2vec对象里...

答：试着增加层的大小，调用word2vec的layerSize方法。

        Word2Vec vec = new Word2Vec.Builder().layerSize(300).windowSize(5)
                .layerSize(300).iterate(iter).tokenizerFactory(t).build();

微调 <a name="dbn">DBNs</a>

现在你应该有了一些构建Word2Vec的基本概念了，下面是如何使用DL4J API的一个例子：

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/word2vec/Word2VecRawTextExample.java?slice=28:97"></script>

这里有一些参数须要特殊注意。第一个就是窗口中须要向量化的词的数量，也就是WindowSize。第二个就是神经网络隐层包含的数据元的数量，也就是LayerSize。这两个参数会乘以输入的大小。

Word2Vec对于信息检索和QA系统的预处理数据尤其有用。这些系统使用DJ4J的 [deep autoencoders](http://deeplearning4j.org/deepautoencoder.html). 对于句子的parsing和其它一些NLP任务，我们也实现了[recursive neural tensor networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html).

## <a name="next">下一步</a>

我们最近仍然在测试[Doc2Vec](../doc2vec.html)和GloVe的实现，请多关注这里！
