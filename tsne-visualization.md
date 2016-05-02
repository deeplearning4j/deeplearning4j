---
title: t-SNE's Data Visualization
layout: default
---

# t-SNE's Data Visualization

[t-Distributed Stochastic Neighbor Embedding](http://homepage.tudelft.nl/19j49/t-SNE.html) (t-SNE) is a data-visualization tool created by Laurens van der Maaten at Delft University of Technology. 

While it can be used for any data, t-SNE (pronounced Tee-Snee) is only really meaningful with labeled data, which clarify how the input is clustering. Below, you can see the kind of graphic you can generate in DL4J with t-SNE working on [MNIST data](http://deeplearning4j.org/deepbeliefnetwork.html). 

![Alt text](../img/tsne.png)

Look closely and you can see the numerals clustered near their likes, alongside the dots. 

Here's how t-SNE appears in Deeplearning4j code. 

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/tsne/TsneExample.java?slice=14:27"></script>
