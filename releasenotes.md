---
title: Release Notes
layout: default
---

# <a name="zeroseventwo">Release Notes for Version 0.7.2</a>
- Added variational autoencoder [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/variational/VariationalAutoEncoderExample.java)
- Activation function refactor
    - Activation functions are now an interface [Link](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/IActivation.java)
    - Configuration now via enumeration, not via String (see examples - [Link](https://github.com/deeplearning4j/dl4j-examples))
	- Custom activation functions now supported [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/activationfunctions/CustomActivationExample.java)
	- New activation functions added: hard sigmoid, randomized leaky rectified linear units (RReLU)
- Multiple fixes/improvements for Keras model import
- Added P-norm pooling for CNNs (option as part of SubsamplingLayer configuration)
- Iteration count persistence: stored/persisted properly in model configuration + fixes to learning rate schedules for Spark network training
- LSTM: gate activation function can now be configured (previously: hard-coded to sigmoid)
- UI:
    - Added Chinese translation
	- Fixes for UI + pretrain layers
    - Added Java 7 compatible stats collection compatibility [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface/UIStorageExample_Java7.java)
	- Improvements in front-end for handling NaNs
	- Added UIServer.stop() method
	- Fixed score vs. iteration moving average line (with subsampling)
- Solved Jaxb/Jackson issue with Spring Boot based applications
- RecordReaderDataSetIterator now supports NDArrayWritable for the labels (set regression == true; used for multi-label classification + images, etc)

### 0.7.1 -> 0.7.2 Transition Notes

- Activation functions (built-in): now specified using Activation enumeration, not String (String-based configuration has been deprecated)


# <a name="zerosevenone">Release Notes for Version 0.7.1</a>
* RBM and AutoEncoder key fixes: 
    - Ensured visual bias updated and applied during pretraining. 
    - RBM HiddenUnit is the activation function for this layer; thus, established derivative calculations for backprop according to respective HiddenUnit.
* RNG performance issues fixed for CUDA backend
* OpenBLAS issues fixed for macOS, powerpc, linux.
* DataVec is back to Java 7 now.
* Multiple minor bugs fixed for ND4J/DL4J

# <a name="zerosevenzero">Release Notes for Version 0.7.0</a>

* UI overhaul: new training UI has considerably more information, supports persistence (saving info and loading later), Japanese/Korean/Russian support. Replaced Dropwizard with Play framework. [Link](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface)
* Import of models configured and trained using [Keras](http://keras.io)
    - Imports both _Keras_ model [configurations](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/ModelConfiguration.java) and [stored weights](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/Model.java#L59)
    - Supported models: [Sequential](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/ModelConfiguration.java#L41) models
    - Supported [layers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/LayerConfiguration.java#L85): _Dense, Dropout, Activation, Convolution2D, MaxPooling2D, LSTM_
* Added ‘Same’ padding more for CNNs (ConvolutionMode network configuration option) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/ConvolutionMode.java)
* Weighted loss functions: Loss functions now support a per-output weight array (row vector)
* ROC and AUC added for binary classifiers [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROC.java)
* Improved error messages on invalid configuration or data; improved validation on both
* Added metadata functionality: track source of data (file, line number, etc) from data import to evaluation. Loading a subset of examples/data from this metadata is now supported. [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/CSVExampleEvaluationMetaData.java)
* Removed Jackson as core dependency (shaded); users can now use any version of Jackson without issue
* Added LossLayer: version of OutputLayer that only applies loss function (unlike OutputLayer: it has no weights/biases)
* Functionality required to build triplet embedding model (L2 vertex, LossLayer, Stack/Unstack vertices etc)
* Reduced DL4J and ND4J ‘cold start’ initialization/start-up time
* Pretrain default changed to false and backprop default changed to true. No longer needed to set these when setting up a network configuration unless defaults need to be changed.
* Added TrainingListener interface (extends IterationListener). Provides access to more information/state as network training occurs [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/api/TrainingListener.java)
* Numerous bug fixes across DL4J and ND4J
* Performance improvements for nd4j-native & nd4j-cuda backends
* Standalone Word2Vec/ParagraphVectors overhaul:
    - Performance improvements
    - ParaVec inference available for both PV-DM & PV-DBOW
    - Parallel tokenization support was added, to address computation-heavy tokenizers.
* Native RNG introduced for better reproducibility within multi-threaded execution environment.
* Additional RNG calls added: Nd4j.choice(), and BernoulliDistribution op.
* Off-gpu storage introduced, to keep large things, like Word2Vec model in host memory. Available via WordVectorSerializer.loadStaticModel()
* Two new options for performance tuning on nd4j-native backend: setTADThreshold(int) & setElementThreshold(int)

### 0.6.0 -> 0.7.0 Transition Notes
Notable changes for upgrading codebases based on 0.6.0 to 0.7.0:

* UI: new UI package name is deeplearning4j-ui_2.10 or deeplearning4j-ui_2.11 (previously: deeplearning4j-ui). Scala version suffix is necessary due to Play framework (written in Scala) being used now.
* Histogram and Flow iteration listeners deprecated. They are still functional, but using new UI is recommended [Link](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface)
* DataVec ImageRecordReader: labels are now sorted alphabetically by default before assigning an integer class index to each - previously (0.6.0 and earlier) they were according to file iteration order. Use .setLabels(List<String>) to manually specify the order if required.
* CNNs: configuration validation is now less strict. With new ConvolutionMode option, 0.6.0 was equivalent to ‘Strict’ mode, but new default is ‘Truncate’
    - See ConvolutionMode javadoc for more details: [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/ConvolutionMode.java)
* Xavier weight initialization change for CNNs and LSTMs: Xavier now aligns better with original Glorot paper and other libraries. Xavier weight init. equivalent to 0.6.0 is available as XAVIER_LEGACY
* DataVec: Custom RecordReader and SequenceRecordReader classes require additional methods, for the new metadata functionality. Refer to existing record reader implementations for how to implement these methods.
* Word2Vec/ParagraphVectors: 
    - Few new builder methods:
        - allowParallelTokenization(boolean)
        - useHierarchicSoftmax(boolean)
    - Behaviour change: batchSize: now batch size is ALSO used as threshold to execute number of computational batches for sg/cbow


# <a name="six">Release Notes for Version 0.6.0</a> 

* Custom layer support
* Support for custom loss functions
* Support for compressed INDArrays, for memory saving on huge data
* Native support for BooleanIndexing where applicable
* Initial support for combined operations on CUDA
* Significant performance improvements on CPU & CUDA backends
* Better support for Spark environments using CUDA & cuDNN with multi-gpu clusters
* New UI tools: FlowIterationListener and ConvolutionIterationListener, for better insights of processes within NN.
* Special IterationListener implementation for performance tracking: PerformanceListener
* Inference implementation added for ParagraphVectors, together with option to use existing Word2Vec model
* Severely decreased file size on the deeplearnning4j api
* `nd4j-cuda-8.0` backend is available now for cuda 8 RC
* Added multiple new built-in loss functions
* Custom preprocessor support
* Performance improvements to Spark training implementation
* Improved network configuration validation using InputType functionality

# <a name="five">Release Notes for Version 0.5.0</a> 

* FP16 support for CUDA
* [Better performance for multi-gpu}(http://deeplearning4j.org/gpu)
* Including optional P2P memory access support
* Normalization support for time series and images
* Normalization support for labels
* Removal of Canova and shift to DataVec: [Javadoc](http://deeplearning4j.org/datavecdoc/), [Github Repo](https://github.com/deeplearning4j/datavec)
* Numerous bug fixes
* Spark improvements

## <a name="four">Release Notes for version 0.4.0</a> 

* Initial multi-GPU support viable for standalone and Spark. 
* Refactored the Spark API significantly
* Added CuDNN wrapper 
* Performance improvements for ND4J
* Introducing [DataVec](https://github.com/deeplearning4j/datavec): Lots of new functionality for transforming, preprocessing, cleaning data. (This replaces Canova)
* New DataSetIterators for feeding neural nets with existing data: ExistingDataSetIterator, Floats(Double)DataSetIterator, IteratorDataSetIterator
* New learning algorithms for word2vec and paravec: CBOW and PV-DM respectively
* New native ops for better performance: DropOut, DropOutInverted, CompareAndSet, ReplaceNaNs
* Shadow asynchronous datasets prefetch enabled by default for both MultiLayerNetwork and ComputationGraph
* Better memory handling with JVM GC and CUDA backend, resulting in significantly lower memory footprint

## Resources

* [Deeplearning4j on Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
* [Deeplearning4j Source Code](https://github.com/deeplearning4j/deeplearning4j/)
* [ND4J Source Code](https://github.com/deeplearning4j/nd4j/)
* [libnd4j Source Code](https://github.com/deeplearning4j/libnd4j/)

## Roadmap for Fall 2016

* [ScalNet Scala API](https://github.com/deeplearning4j/scalnet) (WIP!)
* Standard NN configuration file shared with Keras
* CGANs
* Model interpretability
