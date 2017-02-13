---
title: Deeplearning4j中的Doc2Vec与段落向量
layout: cn-default
---

## Deeplearning4j中的Doc2Vec与段落向量

Doc2Vec的主要作用是在任意文档与标签之间建立联系，所以标签是必需的。Doc2Vec是Word2Vec的一项扩展，而Word2Vec的功能也是学习词与标签之间的关联，而非词与其他词之间的关联。Deeplearning4j中的Doc2Vec实现旨在服务Java、[Scala](./scala.html)和Clojure社区。 

我们首先要获得表示一份文档的“意义”的向量，然后再将这个向量输入有监督的机器学习算法，确定与文档有关的标签。

在ParagraphVectors构建器模式中，`labels()`方法指明了定型中涉及到的标签。以下的示例中给出了与情感分析相关的标签：

``` java
    .labels(Arrays.asList("negative", "neutral","positive"))
```

以下是一个[用段落向量进行分类](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java)的完整实用示例：

``` java
    public void testDifferentLabels() throws Exception {
        ClassPathResource resource = new ClassPathResource("/labeled");
        File file = resource.getFile();
        LabelAwareSentenceIterator iter = LabelAwareUimaSentenceIterator.createWithPath(file.getAbsolutePath());

        TokenizerFactory t = new UimaTokenizerFactory();

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5).labels(Arrays.asList("negative", "neutral","positive"))
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        vec.fit();

        assertNotEquals(vec.lookupTable().vector("UNK"), vec.lookupTable().vector("negative"));
        assertNotEquals(vec.lookupTable().vector("UNK"),vec.lookupTable().vector("positive"));
        assertNotEquals(vec.lookupTable().vector("UNK"),vec.lookupTable().vector("neutral"));}
```

### 扩展阅读

* [语句与文档的分布式表示](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
* [Word2vec教程](./word2vec)
