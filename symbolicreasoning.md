---
title: Deep Learning & Symbolic Reasoning
layout: default
redirect_from: deepsymboliclearning
---

# Deep Learning & Symbolic Reasoning

The two biggest flaws of deep learning are its lack of model interpretability (why did my model make that prediction?) and the amount of data that deep neural networks require in order to learn. 

Geoff Hinton himself has [expressed scepticism](https://www.axios.com/artificial-intelligence-pioneer-says-we-need-to-start-over-1513305524-f619efbd-9db0-4947-a9b2-7a4c310a28fe.html) about whether backpropagation, the workhorse of deep neural nets, will be the way forward for AI.<sup>[1](#one)</sup>   

Research into so-called one-shot learning could address deep learning's data hunger, while deep symbolic learning, or enabling deep neural networks to generate and manipulate concepts, could help solve explainability.<sup>[2](#two)</sup> 

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH AI</a>
</p>

## Symbolic AI

[Symbolic artificial intelligence](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence), also known as GOFAI, was the dominant paradigm in the AI community from the post-War era until the late 1980s. Implementations of symbolic reasoning are called rules engines or expert systems; they are essentially piles of nested if-then statements drawing conclusions based on human-readable concepts. Those concepts are encoded into the system by its programmers. 

One of the main differences between machine learning and traditional symbolic reasoning is where the learning happens. In machine- and deep-learning, the algorithm learns rules as it establishes correlations between inputs and outputs. In symbolic reasoning, the rules are created through human intervention. That is, to build a symbolic reasoning system, first humans must learn the rules by which two phenomena relate, and then hard-code those relationships into a static program. This difference is the subject of a well-known [hacker koan](https://simple.wikipedia.org/wiki/Hacker_koan):

```
In the days when Sussman was a novice, Minsky once came to him as he sat hacking at the PDP-6.
"What are you doing?", asked Minsky.
"I am training a randomly wired neural net to play Tic-tac-toe", Sussman replied.
"Why is the net wired randomly?", asked Minsky.
"I do not want it to have any preconceptions of how to play", Sussman said.
Minsky then shut his eyes.
"Why do you close your eyes?" Sussman asked his teacher.
"So that the room will be empty."
At that moment, Sussman was enlightened.
```

A hard-coded rule is a preconception. It is one form of assumption, and a strong one, while deep neural architectures contain other assumptions, usually about *how* they should learn, rather than what conclusion they should reach. The ideal, obviously, is to choose assumptions that allow a system to learn flexibly and produce accurate decisions about their inputs. 

How can we fuse the ability of deep neural nets to learn probabilistic correlations from scratch alongside abstract and higher-order concepts, which are useful in compressing data and combining it in new ways? How can we learn to attach new meanings to concepts, and to use [atomic concepts as elements in more complex and composable thoughts](https://medium.com/@GaryMarcus/in-defense-of-skepticism-about-deep-learning-6e8bfd5ae0f1) such as language allows us to express in all its natural plasticity? 

Combining symbolic reasoning with deep neural networks and deep reinforcement learning may help us address the fundamental challenges of reasoning, hierarchical representations, transfer learning, robustness in the face of adversarial examples, and interpretability (or explanatory power).

## <a name="footnote">Footnotes</a>

<a name="one">1)</a> Hinton, Yann LeCun and Andrew Ng have all suggested that work on unsupervised learning (learning from unlabeled data) will lead to our next breakthroughs.

<a name="two">2)</a> The two problems may overlap, and solving one could lead to solving the other, since a concept that helps explain a model will also help it recognize certain patterns in data using fewer examples. 

## Resources for Deep Learning and Symbolic Reasoning

This page includes some recent, notable research that attempts to combine deep learning with symbolic learning to answer those questions. 

* [Towards Deep Symbolic Reinforcement Learning](https://arxiv.org/abs/1609.05518) (2016)
by Marta Garnelo, Kai Arulkumaran and Murray Shanahan

*Deep reinforcement learning (DRL) brings the power of deep neural networks to bear on the generic task of trial-and-error learning, and its effectiveness has been convincingly demonstrated on tasks such as Atari video games and the game of Go. However, contemporary DRL systems inherit a number of shortcomings from the current generation of deep learning techniques. For example, they require very large datasets to work effectively, entailing that they are slow to learn even when such datasets are available. Moreover, they lack the ability to reason on an abstract level, which makes it difficult to implement high-level cognitive functions such as transfer learning, analogical reasoning, and hypothesis-based reasoning. Finally, their operation is largely opaque to humans, rendering them unsuitable for domains in which verifiability is important. In this paper, we propose an end-to-end reinforcement learning architecture comprising a neural back end and a symbolic front end with the potential to overcome each of these shortcomings. As proof-of-concept, we present a preliminary implementation of the architecture and apply it to several variants of a simple video game. We show that the resulting system -- though just a prototype -- learns effectively, and, by acquiring a set of symbolic rules that are easily comprehensible to humans, dramatically outperforms a conventional, fully neural DRL system on a stochastic variant of the game.*

* [Learning explanatory rules from noisy data](https://deepmind.com/blog/learning-explanatory-rules-noisy-data/) (2018)
by Richard Evans and Edward Grefenstette 

*Artificial Neural Networks are powerful function approximators capable of modelling solutions to a wide variety of problems, both supervised and unsupervised. As their size andexpressivity increases, so too does the variance of the model, yielding a nearly ubiquitous overfitting problem. Although mitigated by a variety of model regularisation methods, the common cure is to seek large amounts of training data—which is not necessarily easily obtained—that sufficiently approximates the data distribution of the domain we wish to test on. In contrast, logic programming methods such as Inductive Logic Programming offer an extremely data-efficient process by which models can be trained to reason on symbolic domains. However, these methods are unable to deal with the variety of domains neural networks can be applied to: they are not robust to noise in or mislabelling of inputs, and perhaps more importantly, cannot be applied to non-symbolic domains where the data is ambiguous, such as operating on raw pixels. In this paper, we propose a Differentiable Inductive Logic framework, which can not only solve tasks which traditional ILP systems are suited for, but shows a robustness to noise and error in the training data which ILP cannot cope with. Furthermore, as it is trained by backpropagation against a likelihood objective, it can be hybridised by connecting it with neural networks over ambiguous data in order to be applied to domains which ILP cannot address, while providing data efficiency and generalisation beyond what neural networks on their own can achieve.*

* [Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics](https://arxiv.org/abs/1706.04317) (2017)

*The recent adaptation of deep neural network-based methods to reinforcement learning and planning domains has yielded remarkable progress on individual tasks. Nonetheless, progress on task-to-task transfer remains limited. In pursuit of efficient and robust generalization, we introduce the Schema Network, an object-oriented generative physics simulator capable of disentangling multiple causes of events and reasoning backward through causes to achieve goals. The richly structured architecture of the Schema Network can learn the dynamics of an environment directly from data. We compare Schema Networks with Asynchronous Advantage Actor-Critic and Progressive Networks on a suite of Breakout variations, reporting results on training efficiency and zero-shot generalization, consistently demonstrating faster, more robust learning and better transfer. We argue that generalizing from limited data and learning causal relationships are essential abilities on the path toward generally intelligent systems.*

* [Learning like humans with Deep Symbolic Networks](https://arxiv.org/abs/1707.03377) (2017)
by Qunzhi Zhang and Didier Sornette

*We introduce the Deep Symbolic Network (DSN) model, which aims at becoming the white-box version of Deep Neural Networks (DNN). The DSN model provides a simple, universal yet powerful structure, similar to DNN, to represent any knowledge of the world, which is transparent to humans. The conjecture behind the DSN model is that any type of real world objects sharing enough common features are mapped into human brains as a symbol. Those symbols are connected by links, representing the composition, correlation, causality, or other relationships between them, forming a deep, hierarchical symbolic network structure. Powered by such a structure, the DSN model is expected to learn like humans, because of its unique characteristics. First, it is universal, using the same structure to store any knowledge. Second, it can learn symbols from the world and construct the deep symbolic networks automatically, by utilizing the fact that real world objects have been naturally separated by singularities. Third, it is symbolic, with the capacity of performing causal deduction and generalization. Fourth, the symbols and the links between them are transparent to us, and thus we will know what it has learned or not - which is the key for the security of an AI system. Fifth, its transparency enables it to learn with relatively small data. Sixth, its knowledge can be accumulated. Last but not least, it is more friendly to unsupervised learning than DNN. We present the details of the model, the algorithm powering its automatic learning ability, and describe its usefulness in different use cases. The purpose of this paper is to generate broad interest to develop it within an open source project centered on the Deep Symbolic Network (DSN) model towards the development of general AI.*

* [Deep Learning: A Critical Appraisal](https://arxiv.org/abs/1801.00631) (2018)
by Gary Marcus

*Although deep learning has historical roots going back decades, neither the term "deep learning" nor the approach was popular just over five years ago, when the field was reignited by papers such as Krizhevsky, Sutskever and Hinton's now classic (2012) deep network model of Imagenet. What has the field discovered in the five subsequent years? Against a background of considerable progress in areas such as speech recognition, image recognition, and game playing, and considerable enthusiasm in the popular press, I present ten concerns for deep learning, and suggest that deep learning must be supplemented by other techniques if we are to reach artificial general intelligence.*

* [Composable Planning with Attributes](https://openreview.net/forum?id=r154_g-Rb) (2018)
by Amy Zhang, Adam Lerer, Sainbayar Sukhbaatar, Rob Fergus and Arthur Szlam

*The tasks that an agent will need to solve often aren’t known during training. However, if the agent knows which properties of the environment we consider im- portant, then after learning how its actions affect those properties the agent may be able to use this knowledge to solve complex tasks without training specifi- cally for them. Towards this end, we consider a setup in which an environment is augmented with a set of user defined attributes that parameterize the features of interest. We propose a model that learns a policy for transitioning between “nearby” sets of attributes, and maintains a graph of possible transitions. Given a task at test time that can be expressed in terms of a target set of attributes, and a current state, our model infers the attributes of the current state and searches over paths through attribute space to get a high level plan, and then uses its low level policy to execute the plan. We show in grid-world games and 3D block stacking that our model is able to generalize to longer, more complex tasks at test time even when it only sees short, simple tasks at train time.
TL;DR: Compositional attribute-based planning that generalizes to long test tasks, despite being trained on short & simple tasks.*

* [Object-Oriented Deep Learning](https://cbmm.mit.edu/publications/object-oriented-deep-learning) (2018)

by Q. Liao and T.A. Poggio

*We investigate an unconventional direction of research that aims at converting neural networks, a class of distributed, connectionist, sub-symbolic models into a symbolic level with the ultimate goal of achieving AI interpretability and safety. To that end, we propose Object-Oriented Deep Learning, a novel computational paradigm of deep learning that adopts interpretable “objects/symbols” as a basic representational atom instead of N-dimensional tensors (as in traditional “feature-oriented” deep learning). For visual processing, each “object/symbol” can explicitly package common properties of visual objects like its position, pose, scale, probability of being an object, pointers to parts, etc., providing a full spectrum of interpretable visual knowledge throughout all layers. It achieves a form of “symbolic disentanglement”, offering one solution to the important problem of disentangled representations and invariance. Basic computations of the network include predicting high-level objects and their properties from low-level objects and binding/aggregating relevant objects together. These computations operate at a more fundamental level than convolutions, capturing convolution as a special case while being significantly more general than it. All operations are executed in an input-driven fashion, thus sparsity and dynamic computation per sample are naturally supported, complementing recent popular ideas of dynamic networks and may enable new types of hardware accelerations. We experimentally show on CIFAR-10 that it can perform flexible visual processing, rivaling the performance of ConvNet, but without using any convolution. Furthermore, it can generalize to novel rotations of images that it was not trained for.*

### <a name="beginner">Other Deep Learning Tutorials</a>
* [LSTMs and Recurrent Networks](./lstm)
* [Introduction to Deep Neural Networks](./neuralnet-overview)
* [Convolutional Networks](./convolutionalnets)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [Deep Reinforcement Learning](./deepreinforcementlearning)
* [Deeplearning4j Quickstart Examples](./quickstart)
* [ND4J: A Tensor Library for the JVM](http://nd4j.org)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network)
* [Word2vec and Neural Embeddings for Natural-Language Processing](./word2vec.html)


