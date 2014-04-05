Deep Learning for Java
=====================================

Deep learning is a form of state-of-the-art machine learning that can learn to recognize patterns in data unsupervised.

Unsupervised pattern recognition saves time during data analysis, trend discovery and labeling of certain types of data, such as images, text, sound and time series.

[![Build Status](https://api.travis-ci.org/agibsonccc/java-deeplearning.png)](https://api.travis-ci.org/agibsonccc/java-deeplearning).

See [Deeplearning4j.org](http://deeplearning4j.org/) for applications, tutorials, definitions and other resources on the discipline.


Feature set summary
======================

1. Distributed Deep Learning via akka clustering and distributed coordination of jobs via hazelcast with confs stored in zookeeper

2. DBNs for both continuous and binary data

3. Native matrices via jblas

4. Automatic cluster provisioning for EC2

5. Baseline ability to read from a variety of input providers including S3, local file systems

6. Text processing via Word2Vec as well as a TFIDF vectorizer
          
  - Special tokenizers/stemmers and a SentenceIterator interface to make handling text input agnostic
  - Ability to do moving window operations via a Window encoding - optimized by Viterbi

7. Different data preprocessing tools such as an Image loader that allows for binarization, scaling of pixels, normalization via zero unit 
   mean and standard deviation



Coming up
=============================

Recursive Neural nets, Convolutional Neural nets Possibly Recursive Neural Tensor

Matrix provider agnostic: 

A matrix abstraction layer that sits on top of various matrix providers that will allow for 
distributed gpu deep learning via either AMD, NVIDIA, or native with BLAS, 
as well as bindings for colt for plain old java
Abstraction layers for different tasks such as 
face detection, named entity recognition, 
sentiment analysis.



# Maven coordinates



## Singular neural nets
       
       <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
         <version>0.0.3.1</version>
      </dependency>





## Scaleout for multithreaded methods and clustering
       
        <dependency>
         <groupId>org.deeplearning4j</groupId>
           <artifactId>deeplearning4j-scaleout-akka</artifactId>
         <version>0.0.3.1</version>
        </dependency>






## Text analysis

         <dependency>
           <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-scaleout-akka-word2vec</artifactId>
             <version>0.0.3.1</version>
          </dependency>


