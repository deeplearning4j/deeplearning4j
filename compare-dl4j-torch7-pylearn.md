---
title: "Deep Learning Comp Sheet: Deeplearning4j vs. Torch vs. Caffe vs. TensorFlow vs. MxNet vs. CNTK"
layout: default
---

# Comparing Top Deep Learning Frameworks: Deeplearning4j, PyTorch, TensorFlow, Caffe, Keras, MxNet & CNTK

Deeplearning4j is distinguished from other frameworks in its API languages, intent and integrations. DL4J is a JVM-based, industry-focused, commercially supported, **distributed deep-learning framework** that solves problems involving massive amounts of data in a reasonable amount of time. It integrates with Kafka, Hadoop and [Spark](./spark) using an arbitrary number of [GPUs](./gpu) or [CPUs](./native), and it has [a number you can call](http://www.skymind.io/contact) if anything breaks. 

DL4J is portable and platform neutral, rather than being optimized on a specific cloud service such as AWS, Azure or Google Cloud. In speed, its [performance is equal to Caffe](https://github.com/deeplearning4j/dl4j-benchmark) on non-trivial image-processing tasks on multiple GPUs, and better than Tensorflow or Torch. For more information on benchmarking Deeplearning4j, please see this [benchmarks page](https://deeplearning4j.org/benchmark) to optimize its performance by adjusting the JVM's heap space, garbage collection algorithm, memory management and DL4J's ETL pipeline. Deeplearning4j has Java, [Scala](https://github.com/deeplearning4j/scalnet) and [Python APIs, the latter using Keras](./keras).

<p align="center">
<a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH DEEPLEARNING4J</a>
</p>

### Content

Python Frameworks

* <a href="#torch">Pytorch & Torch</a>
* <a href="#tensorflow">TensorFlow</a>
* <a href="#caffe">Caffe</a>
* <a href="#theano">RIP: Theano & Ecosystem</a>
* <a href="#caffe2">Caffe2</a>
* <a href="#chainer">Chainer</a>
* <a href="#cntk">CNTK</a>
* <a href="#dsstne">DSSTNE</a>
* <a href="#dsstne">DyNet</a>
* <a href="#gluon">Gluon</a>
* <a href="#keras">Keras</a>
* <a href="#mxnet">Mxnet</a>
* <a href="#paddle">Paddle</a>
* <a href="#bigdl">BigDL</a>
* <a href="#licensing">Licensing</a>

JVM Considerations

* <a href="#speed">Speed</a>
* <a href="#java">DL4J: Why the JVM?</a>
* <a href="#ecosystem">DL4J: Ecosystem</a>
* <a href="#scala">DL4S: Deep Learning in Scala</a>
* <a href="#ml">Machine-Learning Frameworks</a>
* <a href="#tutorial">Further Reading</a>

### <a name="torch">Pytorch & Torch</a>

A Python version of Torch, known as [Pytorch](https://github.com/pytorch/pytorch), was open-sourced by Facebook in January 2017. PyTorch offers dynamic computation graphs, which let you process variable-length inputs and outputs, which is useful when working with RNNs, for example. In September 2017, Jeremy Howard's and Rachael Thomas's well-known deep-learning course fast.ai adopted [Pytorch](http://www.fast.ai/2017/09/08/introducing-pytorch-for-fastai/). Since it's introduction, PyTorch has quickly become the favorite among machine-learning researchers, because it allows certain complex architectures to be built easily. Other frameworks that support dynamic computation graphs are CMU's DyNet and PFN's Chainer. 

[**Torch**](http://torch.ch/) is a computational framework with an API written in Lua that supports machine-learning algorithms. Some version of it is used by large tech companies such as Facebook and Twitter, which devote in-house teams to customizing their deep learning platforms. Lua is a multi-paradigm scripting language that was developed in Brazil in the early 1990s. 

Torch, while powerful, [was not designed to be widely accessible](https://news.ycombinator.com/item?id=7929216) to the Python-based academic community, nor to corporate software engineers, whose lingua franca is Java. Deeplearning4j was written in Java to reflect our focus on industry and ease of use. We believe usability is the limiting parameter that inhibits more widespread deep-learning implementations. We believe scalability ought to be automated with open-source distributed run-times like Hadoop and Spark. And we believe that a commercially supported open-source framework is the appropriate solution to ensure working tools and building a community.

* [Comparing PyTorch and TensorFlow](https://awni.github.io/pytorch-tensorflow/)

Pros and Cons:

* (+) Lots of modular pieces that are easy to combine
* (+) Easy to write your own layer types and run on GPU
* (+) Lots of pretrained models
* (-) You usually write your own training code (Less plug and play)
* (-) No commercial support
* (-) Spotty documentation

### <a name="tensorflow">TensorFlow</a>

* Google created TensorFlow to replace Theano. The two libraries are in fact quite similar. Some of the creators of Theano, such as Ian Goodfellow, went on to create Tensorflow at Google before leaving for OpenAI. 
* For the moment, **TensorFlow** does not support so-called “inline” matrix operations, but forces you to copy a matrix in order to perform an operation on it. Copying very large matrices is costly in every sense. TF takes 4x as long as the state of the art deep learning tools. Google says it’s working on the problem. 
* Like most deep-learning frameworks, TensorFlow is written with a Python API over a C/C++ engine that makes it run faster. Although there is experimental support for a Java API it is not currently considered stable, we do not consider this a solution for the Java and Scala communities. 
* TensorFlow runs dramatically [slower than other frameworks](https://arxiv.org/pdf/1608.07249v7.pdf) such as CNTK and MxNet. 
* TensorFlow is about more than deep learning. TensorFlow actually has tools to support reinforcement learning and other algos.
* Google's acknowledged goal with Tensorflow seems to be recruiting, making their researchers' code shareable, standardizing how software engineers approach deep learning, and creating an additional draw to Google Cloud services, on which TensorFlow is optimized. 
* TensorFlow is not commercially supported, and it’s unlikely that Google will go into the business of supporting open-source enterprise software. It's giving a new tool to researchers. 
* Like Theano, TensforFlow generates a computational graph (e.g. a series of matrix operations such as z = sigmoid(x) where x and z are matrices) and performs automatic differentiation. Automatic differentiation is important because you don't want to have to hand-code a new variation of backpropagation every time you're experimenting with a new arrangement of neural networks. In Google's ecosystem, the computational graph is then used by Google Brain for the heavy lifting, but Google hasn’t open-sourced those tools yet. TensorFlow is one half of Google's in-house DL solution. 
* From an enterprise perspective, the question some companies will need to answer is whether they want to depend upon Google for these tools. 
* Caveat: Not all operations in Tensorflow work as they do in Numpy. 
* [A Critique of Tensorflow](http://nicodjimenez.github.io/2017/10/08/tensorflow.html)
* [Keras shoot-out: TensorFlow vs MXNet](https://medium.com/@julsimon/keras-shoot-out-tensorflow-vs-mxnet-51ae2b30a9c0)
* [PyTorch vs. TensorFlow](https://medium.com/towards-data-science/pytorch-vs-tensorflow-1-month-summary-35d138590f9)

Pros and Cons

* (+) Python + Numpy
* (+) Computational graph abstraction, like Theano
* (+) Faster compile times than Theano
* (+) TensorBoard for visualization
* (+) Data and model parallelism
* (-) Slower than other frameworks
* (-) Much “fatter” than Torch; more magic
* (-) Not many pretrained models
* (-) Computational graph is pure Python, therefore slow
* (-) No commercial support
* (-) Drops out to Python to load each new training batch
* (-) Not very toolable
* (-) Dynamic typing is error-prone on large software projects

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/) is a well-known and widely used machine-vision library that ported Matlab's implementation of fast convolutional nets to C and C++ ([see Steve Yegge's rant about porting C++ from chip to chip if you want to consider the tradeoffs between speed and this particular form of technical debt](https://sites.google.com/site/steveyegge2/google-at-delphi)). Caffe is not intended for other deep-learning applications such as text, sound or time series data. Like other frameworks mentioned here, Caffe has chosen Python for its API. 

Both Deeplearning4j and Caffe perform image classification with convolutional nets, which represent the state of the art. In contrast to Caffe, Deeplearning4j offers parallel GPU *support* for an arbitrary number of chips, as well as many, seemingly trivial, features that make deep learning run more smoothly on multiple GPU clusters in parallel. While it is widely cited in papers, Caffe is chiefly used as a source of pre-trained models hosted on its Model Zoo site. 

* [Caffe's GitHub repository](https://github.com/BVLC/caffe)
* [Caffe's GitHub issues](https://github.com/BVLC/caffe/issues)

Pros and Cons:

* (+) Good for feedforward networks and image processing
* (+) Good for finetuning existing networks
* (+) Train models without writing any code
* (+) Python interface is pretty useful
* (-) Need to write C++ / CUDA for new GPU layers
* (-) Not good for recurrent networks
* (-) Cumbersome for big networks (GoogLeNet, ResNet)
* (-) Not extensible, bit of a hairball
* (-) No commercial support
* (-) Probably dying; slow development

### <a name="theano">RIP: Theano and Ecosystem</a>

Yoshua Bengio announced on Sept. 28, 2017, that [development on Theano would cease](https://groups.google.com/d/msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ). Theano is effectively dead.

Many academic researchers in the field of deep learning rely on [**Theano**](http://deeplearning.net/software/theano/), the grand-daddy of deep-learning frameworks, which is written in [Python](http://darkf.github.io/posts/problems-i-have-with-python.html). Theano is a library that handles multidimensional arrays, like Numpy. Used with other libs, it is well suited to data exploration and intended for research. 

Numerous open-source deep-libraries have been built on top of Theano, including [Keras](https://github.com/fchollet/keras),  [Lasagne](https://lasagne.readthedocs.org/en/latest/) and [Blocks](https://github.com/mila-udem/blocks). These libs attempt to layer an easier to use API on top of Theano's occasionally non-intuitive interface. (As of March 2016, another Theano-related library, [Pylearn2, appears to be dead](https://github.com/lisa-lab/pylearn2).)

In contrast, Deeplearning4j brings deep learning to production environment to create solutions in JVM languages like Java and Scala. It aims to automate as many knobs as possible in a scalable fashion on parallel GPUs or CPUs, integrating as needed with Hadoop and [Spark](./spark.html).

Pros and Cons

* (+) Python + Numpy
* (+) Computational graph is nice abstraction
* (+) RNNs fit nicely in computational graph
* (-) Raw Theano is somewhat low-level
* (+) High level wrappers (Keras, Lasagne) ease the pain
* (-) Error messages can be unhelpful
* (-) Large models can have long compile times
* (-) Much “fatter” than Torch
* (-) Patchy support for pretrained models
* (-) Buggy on AWS
* (-) Single GPU


### <a name="caffe2">Caffe2</a>

[**Caffe2**](https://caffe2.ai/) is the long-awaited successor to the original Caffe, whose creator now works at Facebook. Caffe2 is the second deep-learning framework to be [backed by Facebook](https://techcrunch.com/2017/04/18/facebook-open-sources-caffe2-its-flexible-deep-learning-framework-of-choice/) after Torch/PyTorch. The main difference seems to be the claim that Caffe2 is more scalable and light-weight. Like Caffe and PyTorch, Caffe2 offers a Python API running on a C++ engine. 

* [Caffe2's GitHub repository](https://github.com/caffe2/caffe2)

Pros and Cons:

* (+) BSD License
* (-) No commercial support

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK) is Microsoft's open-source deep-learning framework. The acronym stands for "Computational Network Toolkit." The library includes feed-forward DNNs, convolutional nets and recurrent networks. CNTK offers a Python API over C++ code. While CNTK appears to have a [permissive license](https://github.com/Microsoft/CNTK/blob/master/LICENSE.md), it has not adopted one of the more conventional licenses, such as ASF 2.0, BSD or MIT. This license does not apply to the method by which CNTK makes distributed training easy -- one-bit SGD -- which is not licensed for commercial use. 

### <a name="chainer">Chainer</a>

Chainer is an open-source neural network framework with a Python API, whose core team of developers work at [Preferred Networks](https://www.crunchbase.com/organization/preferred-networks#/entity), a machine-learning startup based in Tokyo drawing its engineers largely from the University of Tokyo. Until the advent of DyNet at CMU, and PyTorch at Facebook, Chainer was the leading neural network framework for dynamic computation graphs, or nets that allowed for input of varying length, a popular feature for NLP tasks. By its own [benchmarks](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html), Chainer is notably faster than other Python-oriented frameworks, with TensorFlow the slowest of a test group that includes MxNet and CNTK. 

### <a name="dsstne">DSSTNE</a>

Amazon's Deep Scalable Sparse Tensor Network Engine, or [DSSTNE](https://github.com/amznlabs/amazon-dsstne), is a library for building models for machine- and deep learning. It is one of the more recent of many open-source deep-learning libraries to be released, after Tensorflow and CNTK, and Amazon has since backed MxNet with AWS, so its future is not clear. Written largely in C++, DSSTNE appears to be fast, although it has not attracted as large a community as the other libraries. 

* (+) Handles Sparse encoding
* (-) Amazon may not be sharing [all information necessary to obtain the best results with its examples](https://github.com/amznlabs/amazon-dsstne/issues/24)
* (-) Amazon has chosen another framework for use on AWS.

### <a name="dynet">DyNet</a>

[DyNet](https://github.com/clab/dynet), the [Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980), came out of Carnegie Mellon University and used to be called cnn. Its notable feature is the dynamic computation graph, which allows for inputs of varying length, which is great for NLP. PyTorch and Chainer offer the same. 

* (+) Dynamic computation graph
* (-) Small user community

### <a name="gluon">Gluon</a>

Named after a subatomic particle, Gluon is an API over Amazon's MxNet that was [introduced](http://www.businesswire.com/news/home/20171012005742/en/AWS-Microsoft-Announce-Gluon-Making-Deep-Learning) by Amazon and Microsoft in October 2017. It will also integrate with Microsoft's CNTK. While it is similar to Keras in its intent and place in the stack, it is distinguished by its dynamic computation graph, similar to Pytorch and Chainer, and unlike TensorFlow or Caffe. On a business level, Gluon is an attempt by Amazon and Microsoft to carve out a user base separate from TensorFlow and Keras, as both camps seek to control the API that mediates UX and neural net training.    

* [Gluon API on GitHub](https://github.com/gluon-api/gluon-api/)
* Zachary Lipton's [MxNet Gluon Tutorial](https://github.com/zackchase/gluon-slides/blob/master/sept18-gluon.pdf) (PDF)

### <a name="keras">Keras</a>

[Keras](keras.io) is a deep-learning library that sits atop Theano and TensorFlow, providing an intuitive API inspired by Torch. Perhaps the best Python API in existence. Deeplearning4j relies on Keras as its [Python API](./keras) and [imports models from Keras and through Keras from Theano and TensorFlow](./model-import-keras). It was created by [Francois Chollet](https://twitter.com/fchollet), a software engineer at Google. 

* (+) Intuitive API inspired by Torch
* (+) Works with Theano, TensorFlow and Deeplearning4j backends (CNTK backend to come)
* (+) Fast growing framework
* (+) Likely to become standard Python API for NNs

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet) is a machine-learning framework with APIs is languages such as R, Python and Julia which has been [adopted by Amazon Web Services](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html). Parts of Apple are also rumored to use it after the company's acquisition of Graphlab/Dato/Turi in 2016. A fast and flexible library, MxNet involves Pedro Domingos and a team of researchers at the University of Washington. A [comparison](https://deeplearning4j.org/mxnet) between MxNet and some aspects of Deeplearning4j can be found here. 

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle) is a deep-learning framework [created and supported by Baidu](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html). Its name stands for PArallel Distributed Deep LEarning. Paddle is the most recent major framework to be released, and like most others, it offers a Python API. 

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL), a new deep learning framework with a focus on Apache Spark, only works on Intel chips. 

### <a name="licensing">Licensing</a>

Licensing is another distinction among these open-source projects: Theano, Torch and Caffe employ a BSD License, which does not address patents or patent disputes. Deeplearning4j and ND4J are distributed under an **[Apache 2.0 License](http://en.swpat.org/wiki/Patent_clauses_in_software_licences#Apache_License_2.0)**, which contains both a patent grant and a litigation retaliation clause. That is, anyone is free to make and patent derivative works based on Apache 2.0-licensed code, but if they sue someone else over patent claims regarding the original code (DL4J in this case), they immediately lose all patent claim to it. (In other words, you are given resources to defend yourself in litigation, and discouraged from attacking others.) BSD doesn't typically address this issue. 

## JVM Considerations

### <a name="speed">Speed</a>

Deeplearning4j's underlying linear algebra computations, performed with ND4J, have been shown to run [at least twice as fast as Numpy](http://nd4j.org/benchmarking) on very large matrix multiplies. That's one reasons why we've been adopted by teams at NASA's Jet Propulsion Laboratory. Moreover, Deeplearning4j has been optimized to run on various chips including x86 and GPUs with CUDA C.

While both Torch7 and DL4J employ parallelism, DL4J's **parallelism is automatic**. That is, we automate the setting up of worker nodes and connections, allowing users to bypass libs while creating a massively parallel network on [Spark](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark), [Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn), or with [Akka and AWS](http://deeplearning4j.org/scaleout.html). Deeplearning4j is best suited for solving specific problems, and doing so quickly. 

For a full list of Deeplearning4j's features, please see our [features page](./features.html).

### <a name="java">Why the JVM?</a>

We're often asked why we chose to implement an open-source deep-learning project for the JVM, when so much of the deep-learning community is focused on Python. After all, Python has great syntactic elements that allow you to add matrices together without creating explicit classes, as Java requires you to do. Likewise, Python has an extensive scientific computing environment with native extensions like Theano and Numpy.

Yet the JVM and its main languages - Java and Scala - have several advantages. 

First, most major companies and large government organizations rely heavily on Java or a JVM-based system. They have made a huge investment, which they can leverage with JVM-based AI. Java remains the most widely used language in enterprise. It is the language of Hadoop, ElasticSearch, Hive, Lucene and Pig, which happen to be useful for machine learning problems. Spark and Kafka are written in Scala, another JVM language. That is, many programmers solving real-world problems could benefit from deep learning, but they are separated from it by a language barrier. We want to make deep learning more usable to a large new audience that can put it to immediate use. With 10 million developers, Java is the world's largest programming language. 

Second, Java and Scala are inherently faster than Python. Anything written in Python by itself, disregarding its reliance on Cython, will be slower. Admittedly, most computationally expensive operations are written in C or C++. (When we talk about operations, we also consider things like strings and other tasks involved with higher-level machine learning processes.) Most deep-learning projects that are initially written in Python will have to be rewritten if they are to be put in production. Deeplearning4j relies on [JavaCPP](https://github.com/bytedeco/javacpp) to call pre-compiled native C++ from Java, substantially accelerating the speed of training. Many Python programmers opt to do deep learning in Scala because they prefer static typing and functional programming when working with others on a shared code base. 

Third, Java's lack of robust scientific computing libraries can be solved by writing them, which we've done with [ND4J](http://nd4j.org). ND4J runs on distributed GPUs or GPUs, and can be interfaced via a Java or Scala API.

Finally, Java is a secure, network language that inherently works cross-platform on Linux servers, Windows and OSX desktops, Android phones and in the low-memory sensors of the Internet of Things via embedded Java. While Torch and Pylearn2 optimize via C++, which presents difficulties for those who try to optimize and maintain it, Java is a "write once, run anywhere" language suitable for companies who need to use deep learning on many platforms. 

### <a name="ecosystem">Ecosystem</a>

Java's popularity is only strengthened by its ecosystem. [Hadoop](https://hadoop.apache.org/) is implemented in Java; [Spark](https://spark.apache.org/) runs within Hadoop's Yarn run-time; libraries like [Akka](https://www.typesafe.com/community/core-projects/akka) made building distributed systems for Deeplearning4j feasible. In sum, Java boasts a highly tested infrastructure for pretty much any application, and deep-learning nets written in Java can live close to the data, which makes programmers' lives easier. Deeplearning4j can be run and provisioned as a YARN app.

Java can also be used natively from other popular languages like Scala, Clojure, Python and Ruby. By choosing Java, we excluded the fewest major programming communities possible. 

While Java is not as fast as C or C++, it is much faster than many believe, and we've built a distributed system that can accelerate with the addition of more nodes, whether they are GPUs or CPUs. That is, if you want speed, just throw more boxes at it. 

Finally, we are building the basic applications of Numpy, including ND-Array, in Java for DL4J. We believe that many of Java's shortcomings can be solved quickly, and many of its advantages will continue for some time. 

### <a name="scala">Scala</a>

We have paid special attention to [Scala](./scala) in building Deeplearning4j and ND4J, because we believe Scala has the potential to become the dominant language in data science. Writing numerical computing, vectorization and deep-learning libraries for the JVM with a [Scala API](http://nd4j.org/scala.html) moves the community toward that goal. 

To really understand the differences between DL4J and other frameworks, you may just have to [try us out](./quickstart).

### <a name="ml">Machine-learning frameworks</a>

The deep-learning frameworks listed above are more specialized than general machine-learning frameworks, of which there are many. We'll list the major ones here:

* [sci-kit learn](http://scikit-learn.org/stable/) - the default open-source machine-learning framework for Python. 
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html) - The flagship machine-learning framework on Apache. Mahout does classifications, clustering and recommendations.
* [SystemML](https://sparktc.github.io/systemml/quick-start-guide.html) - IBM's machine-learning framework, which performs Descriptive Statistics, Classification, Clustering, Regression, Matrix Factorization and Survival Analysis, and includes support-vector machines. 
* [Microsoft DMTK](http://www.dmtk.io/) - Microsoft's distributed machine-learning toolkit. Distributed word embeddings and LDA. 

### <a name="tutorial">Deeplearning4j Tutorials</a>

* [Introduction to Deep Neural Networks](./neuralnet-overview)
* [Convolutional Networks Tutorial](./convolutionalnets)
* [LSTM and Recurrent Network Tutorial](./lstm)
* [Using Recurrent Nets With DL4J](./usingrnns)
* [Deep-Belief Networks With MNIST](./deepbeliefnetwork)
* [Customizing Data Pipelines With Canova](./image-data-pipeline)
* [Restricted Boltzmann machines](./restrictedboltzmannmachine)
* [Eigenvectors, PCA and Entropy](./eigenvector.html)
* [A Glossary of Deep-Learning Terms](./glossary.html)
* [Word2vec, Doc2vec & GloVe](./word2vec)
