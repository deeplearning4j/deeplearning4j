---
title: Doc2Vec, or Paragraph Vectors, in Deeplearning4j
layout: default
---

## Doc2Vec, or Paragraph Vectors, in Deeplearning4j

The main purpose of Doc2Vec is associating arbitrary documents with labels, so labels are required. Doc2vec is an extension of word2vec that learns to correlate labels and words, rather than words with other words. Deeplearning4j's implentation is intended to serve the Java, [Scala](../scala.html) and Clojure communities. 

The first step is coming up with a vector that represents the "meaning" of a document, which can then be used as input to a supervised machine learning algorithm to associate documents with labels.

In the ParagraphVectors builder pattern, the `labels()` method points to the labels to train on. In the example below, you can see labels related to sentiment analysis:

``` java
    .labels(Arrays.asList("negative", "neutral","positive"))
```

Here's a full working example of [classification with paragraph vectors](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/ParagraphVectorsClassifierExample.java):

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

### Further Reading

* [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
* [Word2vec: A Tutorial](../word2vec)
