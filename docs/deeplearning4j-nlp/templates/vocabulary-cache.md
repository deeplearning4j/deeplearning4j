---
title: Vocabulary Cache
short_title: Vocab Cache
description: Mechanism for handling general NLP tasks in DL4J.
category: Language Processing
weight: 10
---

# How the Vocab Cache Works

The vocabulary cache, or vocab cache, is a mechanism for handling general-purpose natural-language tasks in Deeplearning4j, including normal TF-IDF, word vectors and certain information-retrieval techniques. The goal of the vocab cache is to be a one-stop shop for text vectorization, encapsulating techniques common to bag of words and word vectors, among others.

Vocab cache handles storage of tokens, word-count frequencies, inverse-document frequencies and document occurrences via an inverted index. The InMemoryLookupCache is the reference implementation.

In order to use a vocab cache as you iterate over text and index tokens, you need to figure out if the tokens should be included in the vocab. The criterion is usually if tokens occur with more than a certain pre-configured frequency in the corpus. Below that frequency, an individual token isn't a vocab word, and it remains just a token. 

We track tokens as well. In order to track tokens, do the following:

        addToken(new VocabWord(1.0,"myword"));

When you want to add a vocab word, do the following:

        addWordToIndex(0, Word2Vec.UNK);
        putVocabWord(Word2Vec.UNK);

Adding the word to the index sets the index. Then you declare it as a vocab word. (Declaring it as a vocab word will pull the word from the index.)