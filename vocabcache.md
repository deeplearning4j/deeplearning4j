---
title:
layout: default
---

# How the Vocab Cache Works


The vocab cache is a mechanism for handling general purpose natural language tasks in deeplearning4j including normal tfidf, word vectors, and certain information retrieval techniques.

The goal of the vocab cache is to be a one stop shop for vectorization of text with encapsulation of common techniques across bag of words, word vectors, among other things.

It handles storage of tokens, word count frequencies, inverse document frequencies and document occurrences via an inverted index.

The InMemoryLookupCache is the reference implementation.

In order to use a vocab cache, we have to make the following considerations when iterating over text and indexing tokens.

First of all, we need to figure out if the tokens should be included in the vocab. This usually happens by occurring more than a certain pre configured frequency in the corpus.

Otherwise, an individual token isn't a vocab word, but just a token.

We track tokens as well. In order to track tokens, do the following:


addToken(new VocabWord(1.0,"myword"));

When you want to add a vocab word:

        addWordToIndex(0, Word2Vec.UNK);
        putVocabWord(Word2Vec.UNK);


Add the word to the index (sets the index), and then declare it as a vocab word. Declaring it as a vocab word will pull the word from the index.