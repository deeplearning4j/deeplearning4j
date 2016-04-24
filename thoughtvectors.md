---
title: Thought Vectors, Deep Learning & the Future of AI
layout: default
---

# Thought Vectors, Deep Learning & the Future of AI

“Thought vector” is a term popularized by Geoffrey Hinton, the prominent deep-learning researcher now at Google, which is using [vectors based on natural language](http://www.bloomberg.com/news/articles/2015-10-26/google-turning-its-lucrative-web-search-over-to-ai-machines) to improve its search results. 

A thought vector is like a [*word vector*](http://deeplearning4j.org/word2vec.html#embed), which is typically a vector of 300-500 numbers that represent a word. A word vector represents a word’s meaning as it relates to other words (its context) with a single column of numbers. 

That is, the word is embedded in a vector space using a shallow neural network like [word2vec](http://deeplearning4j.org/word2vec.html), which learns to generate the word's context through repeated guesses. 

A thought vector, therefore, is a vectorized thought, and the vector represents one thought’s relations to others. A thought vector is trained to generate a thought's context. Just as a words are linked by grammar (a sentence is just a path drawn across words), so thoughts are linked by a chain of reasoning, a logical path of sorts. 

So training an algorithm to represent any thought in its relation to others might be called the artificial construction of common sense. Given one thought, a neural network might predict the thoughts that are likely to follow, much like [recurrent neural networks](http://deeplearning4j.org/recurrentnetwork) do with characters and words. Conversation as search. 

Hinton, in a 2015 [speech to the Royal Society in London](https://www.youtube.com/watch?v=IcOMKXAw5VA), said this: 

		"The implications of this for document processing are very important. If we convert a sentence into a vector that captures the meaning of the sentence, then Google can do much better searches; they can search based on what's being said in a document.
		Also, if you can convert each sentence in a document into a vector, then you can take that sequence of vectors and [try to model] natural reasoning. And that was something that old fashioned AI could never do.
		If we can read every English document on the web, and turn each sentence into a thought vector, you've got plenty of data for training a system that can reason like people do. 
		Now, you might not want it to reason like people do, but at least we can see what they would think.
		What I think is going to happen over the next few years is this ability to turn sentences into thought vectors is going to rapidly change the level at which we can understand documents. 
		To understand it at a human level, we're probably going to need human level resources and we have trillions of connections [in our brains], but the biggest networks we have built so far only have billions of connections. So we're a few orders of magnitude off, but I'm sure the hardware people will fix that." 

Let’s pause for a moment and consider what Hinton is saying. 

Traditional, rules-based AI, a pile of if-then statements locking brittle symbols into hard-coded relationships with others, is not flexible enough to represent the world without near infinite amounts of human intervention. Symbolic logic and knowledge graphs may establish strict relations between entities, but those relations are unlikely to adapt quickly to the new.

Hinton is saying that, rather than hard-code the logical leaps that lead an AI from one thought to another, we can simply feed neural nets enough text – enough trains of thought – that they will eventually be able to mimic the thoughts expressed there, and generate their own thought trains, the context of the thoughts they've been fed. 

This affects how well algorithms will understand natural-language queries at search engines like Google, and it will also go beyond pure search. 

With the ability to associate thoughts comes the ability to converse. Thought vectors could serve as the basis for chatbots, personal assistants, and other agents whose purpose is to augment and entertain human beings. That’s the good side. The bad side is that, [on the Internet, you really won’t know who’s a dog](https://upload.wikimedia.org/wikipedia/en/f/f8/Internet_dog.jpg), or in this case, a bot. 

If we define thought vectors loosely, we could say they are already being used to represent similar sentences in different languages, which is useful in [machine translation](http://arxiv.org/pdf/1409.3215). (In fact, improving Google Translate was one the goals that brought thought vectors about.) They are therefore independent of any particular language. 

Thought vectors can also [represent images](http://arxiv.org/abs/1411.4555), which makes them more general than, and independent of, language alone. Thus the term *thought*, a concept more general that the textual or visual mediums by which it is expressed. 
The problem with thought vectors, even if we limit ourselves to words, is that their number increases exponentially with the words used to express them. Thoughts are combinatorial. What's more, one sentence may contain many states, or discrete elements of thought; e.g. x is-a y, or b has-a c. So every sentence might contain and mingle several thoughts.

This is important, because when we vectorize words, we index those words in a lookup table. In the massive matrix of all words, each word is a vector, and that vector is a row in the matrix. (Each column represents a feature of the word, which in a low-dimensional space would be 300-500 columns.) 

Given that neural networks are already taxing current hardware to its limits, the exponentially larger costs of manipulating a dense matrix containing all thought vectors looks impractical. For now.  

The future of this branch of AI will depend on advances in hardware, as well as advances in thought vectorization, or capturing thoughts with numbers in novel ways. (How do we discretize sentences? What are the fundamental units of thought?)

A word should be said about semantic structure. It's possible to embed dependency and constituency based parsing in vectors. In fact, interesting work is being done at [Stanford](http://nlp.stanford.edu/), [Cornell](https://confluence.cornell.edu/display/NLP/Home) and [University of Texas](http://www.katrinerk.com/home/research/publications), among other schools. 

Advances in theory and hardware, in turn, will give us other tools to tackle natural language processing and machine conceptualization, the missing link between symbolic logic, which is abstract, and machine percetion via deep learning, which is processing concrete instances of, say, images or sounds. 

Here are a few of the approaches that are being made to thought vectorization: 

* Doc2vec: [Doc2Vec](http://deeplearning4j.org/doc2vec.html), paragraph vectors and sentence vectors are broadly synonymous. It doesn't necessarily account for word order and it is generally used in associating word groups with labels (in sentiment analysis, for example)
* [Seq2seq bilingual translation](http://arxiv.org/pdf/1409.3215) and [skip-thought vectors](http://arxiv.org/abs/1506.06726).

### <a name="beginner">Other Deeplearning4j Tutorials</a>
* [Restricted Boltzmann Machines](../restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](../eigenvector)
* [LSTMs and Recurrent Networks](../lstm)
* [Neural Networks](../neuralnet-overview)
* [Neural Networks and Regression](../linear-regression)
* [Convolutional Networks](../convolutionalnets)
* [Word2vec](../word2vec)
