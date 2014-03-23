---
title: 
layout: default
---

# tokenization

Tokenization is the process of breaking text down into individual words. Word windows are also composed of tokens. [Word2Vec](../word2vec.html) can output text windows that comprise training examples for input into neural nets, as seen here.

Here's an example of tokenization done with DL4J tools:
                 
         //tokenization with lemmatization,part of speech taggin,sentence segmentation
         TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
         Tokenizer tokenizer = tokenizerFactory.tokenize("mystring");

          //iterate over the tokens
          while(tokenizer.hasMoreTokens()) {
          	   String token = tokenizer.nextToken();
          }
          
          //get the whole list of tokens
          List<String> tokens = tokenizer.getTokens();

The above snippet creates a tokenizer capable of stemming.

In Word2Vec, this is recommended a way of creating a vocabulary. This will avert various quirks in your vocabulary, such as plurals and singulars being counted as 2 different words.


