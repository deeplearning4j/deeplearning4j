---
title: Release Notes
layout: default
---

**Contents**

* <a href="#zeronineone">Version 0.9.1</a>
* <a href="#zeroninezero">Version 0.9.0</a>
* <a href="#zeroeightzero">Version 0.8.0</a>
* <a href="#zeroseventwo">Version 0.7.2</a>
* <a href="#six">Version 0.6.0</a>
* <a href="#five">Version 0.5.0</a>
* <a href="#four">Version 0.4.0</a>

# <a name="zeronineone">Release Notes for Version 0.9.1</a>

**Deeplearning4J**

- Fixed issue with incorrect version dependencies in 0.9.0
- Added EmnistDataSetIterator [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/iterator/impl/EmnistDataSetIterator.java#L32)
- Numerical stability improvements to LossMCXENT / LossNegativeLogLikelihood with softmax (should reduce NaNs with very large activations)

**ND4J**

- Added runtime version checking for ND4J, DL4J, RL4J, Arbiter, DataVec [Link](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/versioncheck/VersionCheck.java)


**Known Issues**

- Deeplearning4j: Use of Evaluation class no-arg constructor (i.e., new Evaluation()) can result in accuracy/stats being reported as 0.0. Other Evaluation class constructors, and ComputationGraph/MultiLayerNetwork.evaluate(DataSetIterator) methods work as expected.



# <a name="zeroninezero">Release Notes for Version 0.9.0</a>

**Deeplearning4J**

- Workspaces feature added (faster training performance + less memory) [Link](https://deeplearning4j.org/workspaces)
- SharedTrainingMaster added for Spark network training (improved performance) [Link 1](https://deeplearning4j.org/distributed), [Link 2](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/mlp/MnistMLPDistributedExample.java)
- ParallelInference added - wrapper that server inference requests using internal batching and queues  [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/inference/ParallelInferenceExample.java)
- ParallelWrapper now able to work with gradients sharing, in addition to existing parameters averaging mode [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-cuda-specific-examples/src/main/java/org/deeplearning4j/examples/multigpu/GradientsSharingLenetMnistExample.java)
- VPTree performance significantly improved
- CacheMode network configuration option added - improved CNN and LSTM performance at the expense of additional memory use [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/CacheMode.java)
- LSTM layer added, with CuDNN support [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java) (Note that the existing GravesLSTM implementation does not support CuDNN)
- New native model zoo with pretrained ImageNet, MNIST, and VGG-Face weights [Link](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo)
- Convolution performance improvements, including activation caching
- Custom/user defined updaters are now supported [Link](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/IUpdater.java)
- Evaluation improvements
    - EvaluationBinary, ROCBinary classes added: for evaluation of binary multi-class networks (sigmoid + xent output layers) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationBinary.java)
    - Evaluation and others now have G-Measure and Matthews Correlation Coefficient support; also macro + micro-averaging support for Evaluation class metrics [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationAveraging.java)
    - ComputationGraph and SparkComputationGraph evaluation convenience methods added (evaluateROC, etc)
    - ROC and ROCMultiClass support exact calculation (previous: thresholded calculation was used) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROC.java#L78-L81)
    - ROC classes now support area under precision-recall curve calculation; getting precision/recall/confusion matrix at specified thresholds (via PrecisionRecallCurve class) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/curves/PrecisionRecallCurve.java#L102-L193)
    - RegressionEvaluation, ROCBinary etc now support per-output masking (in addition to per-example/per-time-step masking)
    - EvaluationCalibration added (residual plots, reliability diagrams, histogram of probabilities) [Link 1](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/EvaluationCalibration.java) [Link 2](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java#L180-L188)
    - Evaluation and EvaluationBinary: now supports custom classification threshold or cost array [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/Evaluation.java#L150-L170)
- Optimizations: updaters, bias calculation
- Network memory estimation functionality added. Memory requirements can be estimated from configuration without instantiating networks [Link 1](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/MultiLayerConfiguration.java#L310-L317) [Link 2](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.java#L451-L458)
- New loss functions:
    - Mixture density loss function [Link](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/lossfunctions/impl/LossMixtureDensity.java)
    - F-Measure loss function [Link](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/lossfunctions/impl/LossFMeasure.java)


**ND4J**
- Workspaces feature added [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/nd4j-examples/src/main/java/org/nd4j/examples/Nd4jEx15_Workspaces.java)
- Native parallel sort was added
- New ops added: SELU/SELUDerivative, TAD-based comparisons, percentile/median, Reverse, Tan/TanDerivative, SinH, CosH, Entropy, ShannonEntropy, LogEntropy, AbsoluteMin/AbsoluteMax/AbsoluteSum, Atan2
- New distance functions added: CosineDistance, HammingDistance, JaccardDistance



**DataVec**
- MapFileRecordReader and MapFileSequenceRecordReader added [Link 1](https://github.com/deeplearning4j/DataVec/blob/master/datavec-hadoop/src/main/java/org/datavec/hadoop/records/reader/mapfile/MapFileRecordReader.java) [Link 2](https://github.com/deeplearning4j/DataVec/blob/master/datavec-hadoop/src/main/java/org/datavec/hadoop/records/reader/mapfile/MapFileSequenceRecordReader.java)
- Spark: Utilities to save and load ```JavaRDD<List<Writable>>``` and ```JavaRDD<List<List<Writable>>``` data to Hadoop MapFile and SequenceFile formats [Link](https://github.com/deeplearning4j/DataVec/blob/master/datavec-spark/src/main/java/org/datavec/spark/storage/SparkStorageUtils.java)
- TransformProcess and Transforms now support NDArrayWritables and NDArrayWritable columns
- Multiple new Transform classes


**Arbiter**
- Arbiter UI: [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/arbiter/BasicHyperparameterOptimizationExample.java)
    - UI now uses Play framework, integrates with DL4J UI (replaces Dropwizard backend). Dependency issues/clashing versions fixed.
    - Supports DL4J StatsStorage and StatsStorageRouter mechanisms (FileStatsStorage, Remote UI via RemoveUIStatsStorageRouter)
    - General UI improvements (additional information, formatting fixes)



### 0.8.0 -> 0.9.0 Transition Notes

**Deeplearning4j**
- Updater configuration methods such as .momentum(double) and .epsilon(double) have been deprecated. Instead: use ```.updater(new Nesterovs(0.9))``` and ```.updater(Adam.builder().beta1(0.9).beta2(0.999).build())``` etc to configure

**DataVec**
- CsvRecordReader constructors: now uses characters for delimiters, instead of Strings (i.e., ',' instead of ",")

**Arbiter**
- Arbiter UI is now a separate module, with Scala version suffixes: ```arbiter-ui_2.10``` and ```arbiter-ui_2.11```


# <a name="zeroeightzero">Release Notes for Version 0.8.0</a>

- Added transfer learning API [Link](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16)
- Spark 2.0 support (DL4J and DataVec; see transition notes below)
- New layers
    - Global pooling (aka "pooling over time"; usable with both RNNs and CNNs) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/pooling/GlobalPoolingLayer.java)
    - Center loss output layer [Link](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/centerloss/CenterLossLenetMnistExample.java)
    - 1D Convolution and subsampling layers [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/Convolution1DLayer.java) [Link2](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/subsampling/Subsampling1DLayer.java)
    - ZeroPaddingLayer [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.java)
- New ComputationGraph vertices
    - L2 distance vertex
    - L2 normalization vertex
- Per-output masking is now supported for most loss functions (for per output masking, use a mask array equal in size/shape to the labels array; previous masking functionality was per-example for RNNs)
- L1 and L2 regularization can now be configured for biases (via l1Bias and l2Bias configuration options)
- Evaluation improvements:
    - DL4J now has an IEvaluation class (that Evaluation, RegressionEvaluation, etc all implement. Also allows custom evaluation on Spark) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/IEvaluation.java)
    - Added multi-class (one vs. all) ROC: ROCMultiClass [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ROCMultiClass.java)
    - For both MultiLayerNetwork and SparkDl4jMultiLayer: added evaluateRegression, evaluateROC, evaluateROCMultiClass convenience methods
    - HTML export functionality added for ROC charts [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java#L93)
    - TSNE re-added to new UI
    - Training UI: now usable without an internet connection (no longer relies on externally hosted fonts)
    - UI: improvements to error handling for ‘no data’ condition
- Epsilon configuration now used for Adam and RMSProp updaters
- Fix for bidirectional LSTMs + variable-length time series (using masking)
- Added CnnSentenceDataSetIterator (for use with ‘CNN for Sentence Classification’ architecture) [Link](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/iterator/CnnSentenceDataSetIterator.java) [Link2](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/sentenceClassification/CnnSentenceClassificationExample.java)
- Spark + Kryo: now test serialization + throw exception if misconfigured (instead of logging an error that can be missed)
- MultiLayerNetwork now adds default layer names if no name is specified
- DataVec: 
    - JSON/YAML support for DataAnalysis, custom Transforms etc
    - ImageRecordReader refactored to reduce garbage collection load (hence improve performance with large training sets)
    - Faster quality analysis.
- Arbiter: added new layer types to match DL4J
    - Performance improvement for Word2Vec/ParagraphVectors tokenization & training.
- Batched inference introduced for ParagraphVectors
- Nd4j improvements
    - New native operations available for ND4j: firstIndex, lastIndex, remainder, fmod, or, and, xor.
    - OpProfiler NAN_PANIC & INF_PANIC now also checks result of BLAS calls.
    - Nd4.getMemoryManager() now provides methods to tweak GC behavior.
- Alpha version of parameter server for Word2Vec/ParagraphVectors were introduced for Spark. Please note: It’s not recommended for production use yet.
- Performance improvements for CNN inference

### 0.7.2 -> 0.8.0 Transition Notes

- Spark versioning schemes: with the addition of Spark 2 support, the versions for Deeplearning4j and DataVec Spark modules has changed
    - For Spark 1: use ```<version>0.8.0_spark_1</version>```
    - For Spark 2: use ```<version>0.8.0_spark_2</version>```
    - Also note: Modules with Spark 2 support are released with Scala 2.11 support only. Spark 1 modules are released with both Scala 2.10 and 2.11 support

### 0.8.0 Known Issues (At Launch)

- UI/CUDA/Linux issue: [Link](https://github.com/deeplearning4j/deeplearning4j/issues/3026)
- Dirty shutdown on JVM exit is possible for CUDA backend sometimes: [Link](https://github.com/deeplearning4j/deeplearning4j/issues/3028)
- Issues with RBM implementation [Link](https://github.com/deeplearning4j/deeplearning4j/issues/3049)
- Keras 1D convolutional and pooling layers cannot be imported yet. Will be supported in forthcoming release.
- Keras v2 model configurations cannot be imported yet. Will be supported in forthcoming release.

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
