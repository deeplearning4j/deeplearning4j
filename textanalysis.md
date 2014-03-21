---
title: 
layout: default
---

# textual analysis and deep learning

Laborious, manual feature extraction is the main disadvantage of applying machine learning to textual analysis. Data scientists need to spend a great deal of time telling the algorithm what to pay attention to. 

A part-of-speech tag on a single word might be one feature they select, the fact that the word occurred might be another, and the number of times it appeared in a given text would be a third, each carefully selected by hand. 

One of the chief advantages of deep learning is that feature creation is largely automated. To describe what it does exactly, we’ll first describe feature extraction in more depth.

A text fed into a neural network passes through several stages of analysis. The first is sentence segementation, in which the software finds the sentence boundaries within the text. The second is tokenization, in which the software finds individual words. In the third stage, parts-of-speech tags are attached to those words, and in the fourth, they are grouped according to their stems or concepts, in a process known as lemmatization. That is, words such as be, been and is will be grouped since they represent the same verb idea.

The neural net called [Word2vec](../word2vec.html) goes as far as lemmatization. Lemmas simply extend features based on stems, which is a process deep learning does in other ways automatically.

Word2vec creates features without human intervention, and some of those features include the context of individual words; that is, it retains context in the form of multiword windows. In deep learning, the meaning of a word is essentially the words that surround it. Given enough data, usage and context, Word2vec can make highly accurate guesses as to a word’s meaning based on its past appearances. 

The output of the Word2vec neural is a vocabulary with a vector attached to it, which can be fed into a deep-learning net for classification/labeling. 

The other method of preparing text for input to a deep-learning net is called Bag of Words (BoW). BoW produces a vocabulary with word counts associated to each element of the text; however, it does not retain context, and therefore is not useful in a granular analysis of those words' meaning. 