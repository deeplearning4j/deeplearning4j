---
title: 
layout: default
---


#Tokenization


Tokenization is the process of isolating text in to individual words. Word windows of a certain length

are also composed of tokens. Word2Vec can output windows of text which comprise training examples

for input in to neural nets as seen here.


Below is an example of how to do tokenization with tools contained in dl4j.


                 
                 //tokenization with lemmatization,part of speech taggin,sentence segmentation
                 TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
                 Tokenizer tokenizer = tokenizerFactory.tokenize("mystring");

                  //iterate over the tokens
                  while(tokenizer.hasMoreTokens()) {
                  	   String token = tokenizer.nextToken();
                  }
                  
                  //get the whole list of tokens
                  List<String> tokens = tokenizer.getTokens();



The above snippet creates a tokenizer capable of doing stemming.

In Word2Vec It is reccomended this as a way of doing vocabulary. This will prevent odd things in your vocabulary such as plurals

and other things that are semantically being the same being counted as 2 different words.


