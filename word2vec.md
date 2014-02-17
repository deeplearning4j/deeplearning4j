---
title: 
layout: default
---

# word2vec

One of deep learning's applications is text analysis, and at the heart of text analysis is [word2vec](https://code.google.com/p/word2vec/), which includes a bag-or-words feature representation tool. Bag-of-words allows us to represent word counts while retaining words' context; i.e. their neighbors. 

There is also a skip gram representation which is used in the dl4j implementation. This was proven to be more accurate due to the more generalizable contexts generated.

The way we measure words' proximity to each other is through [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), a measure of similarity between two word vectors, in which a perfect 90 degree angle represents identity; i.e. France equals France, while Spain has a cosine distance of  0.678515 from France, the highest of any other country.
<!---
Word2Vec is implemented [here]({{ site.baseurl }}/doc/com/ccc/deeplearning/word2vec/Word2Vec.html).
-->
