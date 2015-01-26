---
title: 
layout: zh-default
---

# 使用Word2Vec ,DBNS和RNTNs来进行电影评论情感分析

在这篇文章中,我们教您一步一步使用烂番茄数据集([Rotten Tomatoes](http://www.rottentomatoes.com/))来进行电影评论的情感分析。

您需要从Kaggle (需要注册)那下载数据集:

    https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

这数据集已经被分成训练集和测试集,这使我们的在所有过程中更方便。让我们来下载这数据并加载到我们的神经网络。

这数据集已经被分成训练集和测试集,这使我们的在所有过程中更方便。让我们来下载这数据并加载到我们的神经网络。

您可能会想要收集这些文件在一个新的文件夹里, 下载并解压这训练集:

    unzip train.tsv.zip

在新的文件夹中,我们会看到了一个train.tsv文件。如果想要知道这数据是什么样子,只要输入这个以下命令:

    head train.tsv

命令“头”应该会输出以下图表:

<table id="first_table" class="display">
    <thead>
        <tr>
            <th>PhraseId</th>
            <th>SentenceId</th>
            <th>Phrase</th>
            <th>Sentiment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .</td>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1</td>
            <td>A series of escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1</td>
            <td>A series</td>
            <td>2</td>
        </tr>
        <tr>
            <td>4</td>
            <td>1</td>
            <td>A</td>
            <td>2</td>
        </tr>
        <tr>
            <td>5</td>
            <td>1</td>
            <td>series</td>
            <td>2</td>
        </tr>
        <tr>
            <td>6</td>
            <td>1</td>
            <td>of escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>7</td>
            <td>1</td>
            <td>of</td>
            <td>2</td>
        </tr>
        <tr>
            <td>8</td>
            <td>1</td>
            <td>escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>9</td>
            <td>1</td>
            <td>escapades</td>
            <td>2</td>
        </tr>
    </tbody>
</table>

让我们一步一步为您讲解。

上述图表两列描述该句子: PhraseID和SentenceID。PhraseID是一块较大的通道中的一个子窗口,这使得每一个句子中的每个子窗口成为一个”上下文”例子;比如:通常是组在一起的词。

在上表中,在每一行都有一个SentenceI我们在整个过程中处理同一个句子。这整个句子都呈现在第2行3列的一个短语。在第3列中显示的每个后续词组是原句的一个子集:子窗口, 使我们在更细化的级别里衡量情感。

我们的表只宣示句子中一部分的集,在结束前将会呈现构成后半部分的短语。

这是一个监督数据集,每个子窗口已被真人分配了一个情感标记。这里有一个表映射情感数字标签:

| Label |  Sentiment |
|:----------:|:-------------:|
|  0 |  negative |
|  1 |    somewhat negative   |
| 2 | neutral |
| 3 |    somewhat positive   |
| 4 | positive |

这个标签系统是相当细致入微:很多情感分析的问题是二元分类;即 1或0,正或负,没有更精细的等级了。

在我们的句子1宣示,这句话本身已经分配到“有点儿负面”(somewhat negative),这是适当的标签因为下半部的句子是电影的关键问题。上半部不是关键,但是,它的子短语都被标为“中性”,或2 。

## 来自Kaggle

该数据集拥有在烂番茄(Rotten Tomatoes)数据里的制表符分隔文件。分裂训练/测试一直保存着,目的是要将它作为标杆,但这句子一直拖带在原来的顺序。每个句子被由斯坦福解析器解析成许多短语。每个短语都有PhraseId 。每个句子都有SentenceId 。

重复(如短或常用词)的短语只输入一次到数据中。
* 这train.tsv包含了短语及其相关的情绪标签。同时,我们也提供了一个SentenceId让你可以跟踪哪些词组属于哪一个单句。
* test.tsv只包含词组。您必须指定一个情绪标签给每个短语。

到下一个步骤。

## 数据集:如何解决问题
* 我们想要建设的工具,是一个能把句子分解成评论,再分解成子窗口或上下文,然后由SentenceID将它们分组。对于深度信念网:deep-belief networks( DBNS ),我们将构建一个简单的移动窗口,然后在句子中的每个窗口标注一个情感。我们将使用一个“序列移动窗口(sequence moving window)”的做法与Viterbi的算法来标注词组。这类似于Ronan Collobert等在[Natural Language Processing (Almost) From Scratch](https://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/35671.pdf)的论文上使用的方法 自然语言处理(几乎)从头开始的做法。其特点是字向量,这将在下面进行说明。
* 递归神经张量网络(Recursive Neural Tensor Network: RNTNs ) ,即另一种类型的神经网络,我们将在这里演示如何标记树为子上下文。树的每个子节点将匹配它的跨度,或子窗口,还具有特定的标签。由Richart Socher等人创建的算法:RNTNs是一个树分析算法([tree-parsing algorithm](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)), 即就是这神经网络连接到各个二进制的树的节点上。树的叶子是字向量。

我们会专注在子上下的文情感分析上,主要是因为这上下文和子上下文(不孤立的文本)最能捕捉情感信号。一个词组里的负面的标签,例如含有“没有”或“否”的上下文本,将抵消正面的字。我们需要能够在我们的功能空间捕捉到这种情绪,否则几乎是没有用处。

捕获上下文的量化的代码已经编写完成,这使们能够专注于将高层次逻辑的矢量字匹配子上下文,然后我们会将我们的注意力放在标杆模型的准确性。

## 定义

在我们进一步深入讲解这些方法之前,让我们来为您确认一些术语。

### 词袋(Bag of Words - BoW)

Bag of Words是许多自然语言处理任务中使用的基线代表。它在“文件”的水平是有用的,不是在句子和其子集。

使用Bag of Words,一个文件将会是文本的基本单位或原子单位。BoW不会潜入更深层。它会保留在没有上下文的单词或短语。Bag of Words基本上是一个在向量里的字计数。

一个较复杂的 BoW版本就是“词频-逆文档频率法(term frequency–inverse document frequency),”或TF-IDF。TF -IDF在同个指定的文档内借重量给一个单术语的频率,而扣除术语(discounting terms)是一般在文件中通用的的文字(例如:the,and,等)。列的特征向量的数量会随着词汇量的大小而有所不同,但您将会在每一列有一个字。这产生了一个非常稀疏的功能集,有很多没有出现在文档中的0单词,以及一些在文档中出现的正实数。

### 自然语言处理( NLP )管道

NLP管线将预处理文本转,然后使用神经网络转换成可以进行分类的格式。我们会做两件事情:将数据集分解成文件,然后计算一个词汇。这将用于Bag of Words(单词计数向量)以及字向量,这其中包括上下文。

字计数向量和字向量是两个完全不同的东西,这代表了两种不同的方法来使用NLP ,所以不要混淆这两个词,即使他们是相似的。字计数向量只包含每个出现的单词计数;而字向量捕捉上下文;字周围的词。在机器学习,一个字的意思通常是通过观察周围的词而决定的。

### 词汇计算

一个文本数据集也被称为语料库(corpus),而语料库的词汇拥有在语料库里一套独特的文档。Deeplearning4j拥有一些工具以不同的方法来计算词汇。

### 打破了语料库

在Kaggle数据,每个短语会保持在一个CSV。我们需要解析文本成不同的形式。我们将使用DatasetIterator来解析,这将让您拥有一个可重复和可检验的数据管道。

### DataSetIterator

当您创建一个机器学习模型时,您需要向量化这些数据,因为机器学习算法只能理解向量。矢量是分解非结构化数据到一组功能的过程,它们能分开各方面的原数据。再一次,这两个特征代表是字向量(上下文,每个字都有一个向量来捕获其周围邻近的字)和字计数矢量(文档级别,没有上下文,捕获词语的存在或不存在)。

### DataFetcher

当我们遍历数据集时,我们将根据一定的顺序获取数据。该数据通常取自多个位置。因此,我们希望使用DataFetcher来隔离并能测试数据管道在检索过程中的每一个潜在的端点。

正如其名,DataFetcher能处理数据检索。数据可能潜在Amazon Web Services ,您的本地文件

系统,或MySQL 。DataFetcher能使用一个特定或每个数据源来处理特征向量的组成。考虑到这一点,让我们从CSV获取数据并解析它。正如其名,DataFetcher能处理数据检索。数据可能潜在Amazon Web Services ,您的本地文件系统,或MySQL 。DataFetcher能使用一个特定或每个数据源来处理特征向量的组成。考虑到这一点,让我们从CSV获取数据并解析它。

## CSV处理

Deeplearning4j有一个CSV库,使我们能够使用一下的方法处理好的CSV值:

        CSV csv1 = CSV.separator('\t')
                .ignoreLeadingWhiteSpace().skipLines(1)
                .create();
        csv1.read(csv,new CSVReadProc() {
            @Override
            public void procRow(int rowIndex, String... values) {
                
            }
        });

在上面的代码片段, CSV是一个文件对象。

这回调让我们能获取数据。我们的目标是收集文字和创造一个丰富的语料库。在回调的时候,也就是我们会传递CSVReadProc(作为参数)给读取方法,我们要抓住文本并把每一行作为一个文档(这是procRow将会运行的,或者叫作“流程排” ) 。基于该数据集的性质,我们只能创建文档的列表。

由于Kaggle主要是为短语分类,我们将在列表中储存每一个短语为排。

我们会被带到一个称为TextRetriever的抽象数据类型,它将包含以下内容:

    Map<String,Pair<String,String>> //mapping the phraseID to the content and the associated label.

我们在procRow的CSV解析器 ,在上文有提到,随后将:

    pair.put(values[0],new Pair<>(values[2],values[3]));

这就是我们如何映射短语和标签,我们可以不在链接和使用任何特定的分类器下使用它。现在让我们来测试一下。

## 测试

鉴于我们分离的担忧和测试,我们将构建一个单元测试。由于我们的数据是在我们的类路径(classpath)中,我们将使用一些巧妙的技巧来获取它。Deeplearning4j使用Spring为反射功能以及使用更高级的类路径版本来寻获组件。怀着这个概念,我们的测试就如下面:

    @Test
    public void testRetrieval() throws  Exception {
         Map<String,Pair<String,String>> data = retriever.data();
        Pair<String,String> phrase2 = data.get("2");
        assertEquals("A series of escapades demonstrating the adage that what is good for the goose",phrase2.getFirst());
        assertEquals("2",phrase2.getSecond());
    }

我们的目标是建立一套我们可以认为是组件的抽象体,而不是个别的步骤。 (我们要以后可能做的一件事是仅获得词组,或只是标签)为了简便起见,我将让您看看相关的测试:


    @Test
    public void testNumPhrasesAndLabels() {
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.phrases().size());
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.labels().size());
    }

如果很多问题,您要使用不同分类的方法测试以确定哪个效果最好(或使用集成学习,这会比任何一种方法更好)的多种方法。

## 句子迭代器

现在,我们有了基本的CSV处理,让我们把它变成这一个数据管道的开端。一个句子迭代器将有助于创造一个奠定往后进一步处理的基础的语料库。在这个特殊的情况下,每个在语料库的句子将作为一个“文件” 。那么,这语料库到底是什么个样子?

我们的文件都有标签,因为这数据是被监督的,我们正在实施一个标签识别句子迭代器。一个句子迭代器的主要概念是:它知道它在语料库的位置,我们可以随时检索下一个句子,而当它完成时它会告诉我们。这是一个很大的责任,所以我们要隔离这些功能。逻辑的核心在这里:

        private List<String> phrases;
        private List<String> labels;
        private int currRecord;
        private SentencePreProcessor preProcessor;

我们的目标是跟踪我们当前句子与标签。我们使用currRecord位置和我们之前建立的文本检索来检索数据。这将分开数据检索和迭代的职责。

这里是代码的丰富部分:

        try {
            TextRetriever retriever = new TextRetriever();
            this.phrases = retriever.phrases();
            this.labels = retriever.labels();
        } catch (IOException e) {
            e.printStackTrace();
        }

这相当的方便。它给我们的正是我们需要遍历的,而我们也已经把它装上一个接口-这接口是你使用Deeplearning4j所建立的任何数据管道的标准。

测试:


    @Test
    public void testIter() {
        LabelAwareSentenceIterator iter = new RottenTomatoesLabelAwareSentenceIterator();
        assertTrue(iter.hasNext());
        //due to the nature of a hashmap internally, this may not be the same everytime
        String sentence = iter.nextSentence();
        assertTrue(!sentence.isEmpty());
    }

这将验证我们的一个组成部分是能行的,所以现在我们可以不再担心像文字向量和Bag of Words更高层次的概念。让我们先建立一个BoW的DataFetcher先 。(它比你想象中更容易)

    public class RottenTomatoesBagOfWordsDataFetcher extends BaseDataFetcher {
    private LabelAwareSentenceIterator iter;
    private BagOfWordsVectorizer countVectorizer;
    private TokenizerFactory factory = new DefaultTokenizerFactory();
    private DataSet data;
        
    public RottenTomatoesBagOfWordsDataFetcher() {
        iter = new RottenTomatoesLabelAwareSentenceIterator();
        countVectorizer = new BagOfWordsVectorizer(iter, factory, Arrays.asList("0", "1", "2", "3", "4"));
        data = countVectorizer.vectorize();
    }
        
    @Override
    public void fetch(int numExamples) {
        //set the current dataset
        curr = data.get(ArrayUtil.range(cursor, numExamples));
    }    
    }

短短几行,但我们会把它分解成组件
* 迭代器。我们早些时候建立了它,目的是要在我们遍历数据时进行跟踪。它与一个字符串和数据表有关联。
* CountVectorizer。这是Bag of Words的主力。让我们使用矢量把数据加载到内存,并在必要时进行迭代。

注:以上过程是RAM密集型的,所以只能在一个相当强大的服务器上运行。使用TF-IDF的词汇来修剪字会给您一个很好的数据,但在这里我们会跳过这一步。( ND4J现在只支持密集矩阵,我们正在努力发展其他的方式来处理稀疏格式 )由于ND4J是以Blas为重点的框架,这就是我们将支持的。

那么我们究竟做了什么?我们编写了代码来解析CSV ,采取文字,把它映射到标签,并迭代它来产生一个矩阵。在这个过程中,我们建立了一个关键组成部分:词汇。它拥有约17500字和15万句子。对于Bag of Words的词矩阵,这将产生17,000列(每字一列),5万行的稀疏表示。

不是玩具矩阵...

这就是说,我们并不需要花很多时间在运行Bag of Words,因为我们只要点击空格来确定有没有这个字。当我们探讨了Word2vec ,我们将看到DBN如何分类既它们的表现。

## 词向量

让我们设置单词数量向量放在一边,考虑字向量。请记住,文字载体有助于特征化的文本上下文。我们使用[Viterbi](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)算法有表决权的移动窗口进行文档分类。

由于这是一个以字向量为基础的方法,我们将使用Word2vec,看看它如何训练。它与Bag of Words不同的是,这些特点是确定的(根据规则), Word2vec本身是个神经网络,同时它也处理其他神经网络的数据,这意味着我们正在处理的概率和培训系数。请记住, Word2vec代表词的用法;用量是概率的问题,而不是因循守旧规则。

由于眼看就会理解,我们就用D3形象化这16000字的词汇。我们使用一种称为t-SNE算法来衡量字与字之间的亲近度。这样做让我们确保词簇本身是一致的。

字向量对连续运用文本非常有用。它们可以使用适当的整体方法来分类文档(投票),以及优化窗口和标签的似然估计。

那么,Word2vec在代码是什么样的呢?关键代码段是在这里:

     Word2vec vec = new Word2Vec.Builder().iterate(iter).tokenizerFactory(factory)
        .learningRate(1e-3).vocabCache(new InMemoryLookupCache(300))
        .layerSize(300).windowSize(5).build();
     vec.fit();

你会发现我们指定的文件迭代器,一个标记生成器工厂,学习率,层和窗口大小等等。在此演练的第二部分,我们讲讲解这些参数,因为它们会在深信念网络使用:
* 迭代(iter):这DocumentIterator是我们的原始文本的管道
* 工厂(factory):我们工厂标记生成器,处理标记化文本
* 学习率(learning rate):步长
* 缓存(cache):这就是我们所有关于词汇的元数据的储存,包括文字载体, TFIDF分数,文档频率,以及文件。
* 层尺寸(layer size):这是每个字特征的数量
* 窗口大小(window size):窗口大小用于遍历文本,这是训练上下文的长短。

## 一些结果

你如何评价这特征向量的执行能力?不像分类网,它没有f1得分给无监督,生成的学习。一个快速和旁门左道的技术是使用最接近的字。第一个字是搜索项,在阵列中的那些词语是已被Word2Vec确认为最接近的含义。
* 有趣的(amusing) [有时候(sometimes),人物(characters),投(cast),经常(often),搞笑(funny),平出(flat-out),懒虫(slackers),多(many),聪明(clever),战争(wars),要么(either)]
* 寒心(chilling) [好运(luck),有效(effectively),渗出(oozing),严重(severely),成长(grew),内疚(guilty),有才华(talented),快乐(pleasure),伙计们(guys),冰(ice),张口结舌(tongue-tied)]

另 外 , 在 上 述 的 例 子 中 , 有 趣 的 (amusing) 具 有 积 极 的 含 义 和 它 涉 及 到 性 能 , 而 寒 心(chilling)是有一部分是负的,一部分是物理。如果我们反映每个关键字域名的语义,如果人物(character)和搞笑(funny)是与有趣的(amusing)接近是有道理的。

您应该自己练习使用这些词汇和搜索相似之处。它的结果可以使用t- SNE呈现。我们在另一篇文章给予更详细的解释。
