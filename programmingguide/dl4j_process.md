title: DeepLearning4j Process Overview
layout: default 

------

# DL4J Process Overview

Building a neural network requires the following phases. DeepLearning4j plays a central role in each phase. 

1. Data Preparation
  - Extract raw data from the source such as text files, images or video.
  - Transform raw data to arrays of numeric values, a format neural networks can understand
  - Apply additional transformations, scaling, or normalization to data
2. Creating a Neural Network
  - Define neural network architecture
  - Train the neural network
  - Tune neural network hyperparamters
3. Evaluation and Deployment of Neural Network
  - Evaluate neural network on new data
  - Deploy neural network if satisfied

# DeepLearning4j ETL

## Data Type Overview

To build a neural network, it is important to first identify the source data. Raw data can come from such diverse sources as text, tabular files, images, and video. The particular source and format of the data should guide your decision on which neural network architecture to use. For instance, convolutional neural networks work well with images, and recurrent neural networks are suitable for time series data. 

Here, we will outline some common data types:

The simplest data type are *general numeric vectors*. Each observation is represented by a vector of features and every observation is represented by the same number of features; that is, each feature is a numeric element of the vector, and the vectors are of equal length. For this type of data, feed forward neural networks are used. Feed forward neural networks consist of one or more layers of neurons. The first layer will take a vector of features as the inputs. In subsequent layers, each neuron in the layer receives a signal from every feature and computes an output to then send to the next layer of neurons. To architect, or create the structure, of a feed forward neural network, you must decide upon the number of hidden layers and the number of neurons in each hidden layer. 

The next data type are *images*. Images are represented as matrices, where each value in the matrix represents a pixel value in the image. For colored images, the standard RBG color model is used. Here, an image is represented by 3 matrices, which contain values for the red, green, and blue colors. To understand why this works, we note that any color can be represented by some combination of the red, green, and blue colors. Thus each pixel is represented by 3 values, representing the intensity for red, green, and blue. For grayscale images, images are represented by only one matrix, which contain values that control how white or black the pixel appears. The neural network architecture used for image data are generally convolutional neural networks (CNN's). CNN's take image data as matrices and process the matrices block by block. Thus, CNN's learn to identify relevant features of images used for the specified task whether it is image reconstruction or classification.

Neural networks also work well with *sequential or time series data*. In this case, an observation is represented by a sequence of numeric values or vectors. Although observations have the same number of features, the sequences can be of different lengths between observations. The neural network architecture used for this type of data are recurrent neural networks (RNNs). At each timestep of the data, an RNN takes the numeric value or vector from the data as input, but also takes the output from the neural network at the last timestep. The output of the current timestep is then passed onto the next time step of the neural network as an input. The output from the last timestep is then used to assign a prediction about the observation. Note that if the output of an observation is also time series in nature, then each timestep of the neural network can output a prediction, and the predictions can then be aggregated into time series output format.

Lastly, *text data* can also be used as input for neural networks. Using text for deep learning is a challenge, because representing text as numeric data is challenging. Text will generally first be converted to a list of documents, such as a tweet or a newspaper article. These documents can then be broken down into tokens at the level of single words or alternatelively, n-grams, a contiguous sequence of letters, words, syllables. These tokens are then embedded into a vector space, generally with an algorithm called [Word2Vec](https://deeplearning4j.org/word2vec). This vectorization of text is necessary because neural networks can only understand data in an array like format of numeric values. Once this process is finished, the final vectorized representation of text can be used to train a variety of neural networks such as FFNN's and RNN's. 

Once data source for neural network training is identified, the next step is the challenge of converting the raw data into a format that neural networks can read. This is one of the hardest challenges in deep learning. Raw data that can vary between text files, images, and video must be converted to numeric values formatted into an array-like structure in order for a neural network to understand them. Thus, data pipeline work must be done to transform the raw data into the suitable format. In DL4J, the data will typically need to be represented by a `DataSet` object. The details of this processing phase will be explored in Chapter 4.

## Processing Data

Once data is in the appropriate format, further processing is usually needed. There are several reasons for this: 

* Data will often need to be randomly shuffled before feeding into a neural network. This is done to avoid training batches with highly correlated observations, which can negatively impact neural network training. 
* Data will also need to be split into different sets. The process of training and testing a neural network usually requires at least two sets of non-overlapping data: the training set and the testing set. The training set is exactly what it sounds like; it is the portion of the data used to train a neural network. Usually, the training set will comprise the majority of the observations in the original dataset. Once network training is finished, the testing set is used to evaluate how well the neural network does on a task. An additional split of the data is often used in practice as well. This split of the data is called the validation set which is used to set the hyperparameters of a neural network, such as the learning rate, number of hidden layers, and weight initialization. These splits of the data are needed in order to teach the neural network to generalize well on data it has not seen before.
* Data will also need to be further processed if [Spark](https://deeplearning4j.org/spark) is used to train a neural network. Spark is often used for this purpose because training a neural network can be computationally heavy. Thus, Spark can be used to train a neural network in parallel. The details of using Spark with DL4J to train neural networks will be covered in a later chapter. For now, it is important to know that the data will also need to be converted into a format that is compatible with Spark if Spark is used.

To complete the processes outlined above and more, DataVec should be used. DataVec is a vectorization and ETL (Extract, Transform, and Load) library that was built by the Skymind team to make these outlined tasks as user-friendly as possible. DataVec will be explored in more detail in Chapter 3.

# Building and Training the Neural Network

Once the data is finally ready to be used for neural network training, we must set up the neural network. To start building a neural network, we need to first select a type of neural network architecture appropriate to the data and target output. These are done with the `MultiLayerNetwork` class or the `ComputationGraph` class in DL4J. 

## Building Neural Networks

Common neural networks include feed forward neural networks, convolutional neural networks, and recurrent neural networks. As we learned earlier, these neural networks are useful for processing general numeric vector data, image data, and time series data, respectively. Each of these neural networks can be built using either of two classes in DL4J: `MultiLayerNetwork` and `ComputationGraph`. These classes are used to specify the architecture of the network such as the number of hidden layers, the learning rate, the number of hidden nodes, and essentially any other hyperparameter of the neural network. 

The difference between `MultiLayerNetwork` and `ComputationGraph` is that `ComputationGraph` is more flexible than `MultiLayerNetwork`. `MultiLayerNetwork` can essentially be thought of a stack of neural network layers with a single input and output layer. However, with `ComputationGraph`, neural networks with multiple inputs and outputs can be created as well as a network in the form of a directed acylic graph. In other words, `ComputationGraph` can be used to build everything a `MultiLayerNetwork` can build and more.  However, the tradeoff is that the configuration using `ComputationGraph` is more complex. Thus, when building more typical neural network structures, `MultiLayerNetwork` should be used whenever possible. 

<!–– We should link to the above-mentioned classes and examples of their use. ––>

## Training Neural Networks

Neural networks are typically trained using batches of the training data. Updates to the weights and biases of the neural network, which affect the outputs of each node of the network, are done batch by batch. Usually, multiple passes through the training data will be used and each pass is called an *epoch*. With DL4J, the `MultiLayerNetwork` or `ComputationGraph` and the `DataSet` classes will handle the details behind batch training. All the user needs to do is create loops to iterate through the number of epochs and call on functions from either `MultiLayerNetwork` or `ComputationGraph` using the `DataSet` object. 

Dl4J provides a user interface (UI) for monitoring the neural network training process. This will allow you to visualize the neural network training status in real time on your browser. The UI is mainly used to help tune neural networks or find an optimal or near optimal combination of hyperparameter values for a specific task. 

The first thing to set up the UI is to add the following dependency in the project `pom.xml` file. 

	<dependency>
        	<groupId>org.deeplearning4j</groupId>
        	<artifactId>deeplearning4j-ui_2.10</artifactId>
        	<version>${dl4j.version}</version>
    	</dependency>


It is also relatively straightforward to set up the UI in your code:

    //Initialize the user interface backend
    UIServer uiServer = UIServer.getInstance();

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
    
    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage);

    //Then add the StatsListener to collect this information from the network, as it trains
    net.setListeners(new StatsListener(statsStorage));

To access the UI, just go to `http://localhost:9000/train` in your browser. You can set the port by using the `org.deeplearning4j.ui.port` system property: i.e., to use port 9001, pass the following to the JVM on launch: 

	-Dorg.deeplearning4j.ui.port=9001

The neural network will then collect information and route it to the UI as you train your neural network. The UI provides information such as the loss value after each iteration and epoch, as well as neural network information such as number of parameters and layers, and more. 

The UI is useful because it can help you train the neural network properly. For instance, if the training process is going well, you should be seeing a downward trend in the loss value over time. If not, you should adjust hyperparameters like the learning rate or make sure the data was normalized correctly. On the other hand, if the loss value is going downward but extremely slowly, the learning rate may be too small. Furthermore, the UI provides the mean layer activation values over time, which can help you to detect vanishing or exploding activations. More help hints and further reading on this topic can be found [here](https://deeplearning4j.org/visualization).

## Evaluating Neural Networks
After the training phase, you will need to evaluate how well a neural network performs on the testing set. Depending on the task, different metrics will be used, such as AUC (area under ROC curve) or MSE (mean squared error). Furthermore, these metrics can be used to evaluate the network after each epoch on the validation set to help set hyperparameters of the network and perform early stopping. These operations can easily be done using the `DataSet` object and the `ComputationGraph` or `MultiLayerNetwork`, whether or not the specific architecture is a feed forward network, convolutional network, or recurrent network. 

DL4J provides many evaluation functions as part of the [eval](https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html) class in deeplearning4j. The functionality includes basic accuracy, f1 score using false negative and true and false positive rates, and more. 
