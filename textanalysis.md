---
title: Textual analysis and deep learning
layout: default
---

# Textual analysis and deep learning

While images are inherently ambiguous, words belong to a set of semi-structured data known as language, which contains information about itself. 

One way to view language is as a form of data compression, in which knowledge of the world is consolidated into a symbolic set. Like a lossy file or a serialized dataset, words are a compact rendering of something larger. You could argue, therefore, that words are a more promising field for deep learning than images, because you can get to the essence of them. 

That said, textual analysis has presented many challenges for machine learning. Laborious, manual feature extraction is the main disadvantage of applying three-layer neural nets to textual analysis. In those cases, data scientists need to spend a great deal of time telling the algorithm what to pay attention to. 

A part-of-speech tag on a single word might be one feature they select, the fact that the word occurred might be another, and the number of times it appeared in a given text would be a third, each rule carefully and deliberately generated. That process leads to a high ratio of features per word, many of which can be redundant. 

One of the chief advantages of deep learning is that feature creation is largely automated. To describe what it does exactly, we’ll first describe feature extraction in more depth.

A text fed into a neural network passes through several stages of analysis. The first is sentence segementation, in which the software finds the sentence boundaries within the text. The second is [tokenization](../tokenization.html), in which the software finds individual words. In the third stage, parts-of-speech tags are attached to those words, and in the fourth, they are grouped according to their stems or concepts, in a process known as lemmatization. That is, words such as be, been and is will be grouped since they represent the same verb idea. 

The neural net called [Word2vec](../word2vec.html) goes as far as lemmatization. Lemmas simply extend features based on stems, which is a process deep learning does in other ways automatically.

Before we turn to Word2vec, however, we'll cover a slightly simpler algorithm, [Bag of Words](../bagofwords-tf-idf.html).