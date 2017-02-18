---
title: 语句迭代器
layout: cn-default
---

# 语句迭代器

[Word2vec](./word2vec.html)和[词袋法](./bagofwords-tf-idf.html)都会用到一个[语句迭代器](./doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html)。

它会将文本片段以向量的形式输入神经网络，同时也决定了文本处理中的文档概念。

在自然语言处理中，通常用一份文档或一个句子来封装算法所应学习的一种上下文。

这方面的例子包括推特推文和整篇新闻报道的分析。[语句迭代器](./doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html)的作用是将文本划分为可以处理的片段。请注意，语句迭代器与输入的类型无关。所以文本（文档）片段可以来自一个文件系统、推特API或者Hadoop。

语句迭代器的输出会传递给一个[分词器](./org/deeplearning4j/word2vec/tokenizer/Tokenizer.html)，由分词器来处理具体的词例（token）。词例通常是词语，但也有可能是n-gram、skipgram或其他单位，具体取决于输入的处理方式。分词器是由[分词器工厂](./doc/org/deeplearning4j/word2vec/tokenizer/TokenizerFactory.html)逐句创建的。分词器工厂的输出会传递给处理文本的向量生成器。 

以下是一些典型示例：

         SentenceIterator iter = new LineSentenceIterator(new File("your file"));

上述代码假设文件中的每一行都是一个句子。

也可以用以下方法来处理语句字符串列表。

	     Collection<String> sentences = ...;
	     SentenceIterator iter = new CollectionSentenceIterator(sentences);

上述代码假设每个字符串都是一个句子（文档）。记住，这可以是推文列表，也可以是新闻文章列表，两者都适用。

您可以用如下方式对文件进行迭代：
          
          SentenceIterator iter = new FileSentenceIterator(new File("your dir or file"));

上述代码将逐行解析文件，返回每个文件中的各个句子。

对于任何复杂情况，我们建议使用机器学习级别的数据加工管道，比较常见的代表是[UimaSentenceIterator](./doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html)（UIMA语句迭代器）。

UimaSentenceIterator的功能包括分词、词性标注、词形还原等。UimaSentenceIterator对一组文件进行迭代，可以切分语句。您也可以根据为其输入数据的AnalysisEngine（分析引擎）来自定义UimaSentenceIterator的行为。

AnalysisEngine是[UIMA](http://uima.apache.org/)的文本处理数据加工管道概念。DeepLearning4j配有对应所有通用任务的标准分析引擎，您可以自行定义输入哪些文本以及如何界定句子。AnalysisEngines是[opennlp](http://opennlp.apache.org/)数据加工管道的线程安全版本。我们还提供基于[cleartk](http://cleartk.googlecode.com/)的数据加工管道，用以处理通用任务。

UIMA用户或对其感兴趣的读者请注意，这一加工管道使用cleartk类型系统来处理词例、语句和类型系统内的其他注解。

UimaSentenceItrator的创建方式如下：

            SentenceIterator iter = UimaSentenceIterator.create("path/to/your/text/documents");

您也可以直接实例化：

			SentenceIterator iter = new UimaSentenceIterator(path,AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(TokenizerAnnotator.getDescription(), SentenceAnnotator.getDescription())));

熟悉UIMA的用户请注意，上述方法会大量使用Uimafit来创建分析引擎。您也可以通过扩展SentenceIterator来创建自定义的语句迭代器。
