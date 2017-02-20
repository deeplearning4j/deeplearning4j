---
title: 分词
layout: cn-default
---

# 分词

分词（tokenization）是将文本分解为单个词语的过程。词窗口也是由词例（token）组成的。[Word2Vec](./word2vec.html)可以输出文本窗口，作为定型样例输入神经网络，如下文所示。

以下是用DL4J工具进行分词的示例：
                 
         //采用词形还原、词性标注、语句切分的分词
         TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
         Tokenizer tokenizer = tokenizerFactory.tokenize("mystring");

          //对词例进行迭代
          while(tokenizer.hasMoreTokens()) {
          	   String token = tokenizer.nextToken();
          }
          
          //获得整个词例列表
          List<String> tokens = tokenizer.getTokens();

上述代码创建了能够进行词干提取的分词器。

我们推荐在Word2Vec中采用这种方式生成词汇表，如此可以避免词汇表出现异常，比如同一个名词的单数和复数形式被记为两个不同的词。
