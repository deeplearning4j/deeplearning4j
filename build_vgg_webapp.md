---
title: Building an Image Classification Web Application Using VGG-16
layout: default
---

# How to Build an Image Classification Web App With VGG-16

Neural networks are setting new accuracy records for image recognition. This page describes how to build a web-based application to use a well-known network, VGG-16, for inference to classify images uploaded by the app's users. 

**Contents**

* [What is VGG-16?](#VGG-16)
* [Using VGG-16 for Web Applications](#Using VGG-16)
* [Loading Pre-Trained Models](#Loading Pre-Trained Models)
* [Configuring Data Pipelines for Testing](#Configure Data Pipelines for Testing)
* [Testing Pre-Trained Models](#Testing Pre-Trained Models)
* [Saving Models with ModelSerializer](#Save Model with ModelSerializer)
* [Building Web Apps to Take Input Images](#Build Web App to Take Input Image)
* [Tying Web App Front-End to Neural Net Backend](#Tie Web App Front End to Neural Net Backend)
* [Just Show Me the Code](#code)
* [Example Predictions (Cats! Dogs!)](#example)

## <a name="VGG-16"> What is VGG-16?</a>

Since 2010, [ImageNet](http://image-net.org/) has hosted an annual [challenge](http://www.image-net.org/challenges/LSVRC/) where research teams present solutions to image classification and other tasks by training on the ImageNet dataset. ImageNet currently has millions of labeled images; it's one of the largest high-quality image datasets in the world. The Visual Geometry group at the University of Oxford did really well in 2014 with two network architectures: VGG-16, a 16-layer convolutional Neural Network, and VGG-19, a 19-layer Convolutional Neural Network. 

Here are the results:

* [VGG-16 ILSVRC-14](http://www.image-net.org/challenges/LSVRC/2014/results#clsloc) 
* VGG-16 also performed well on two other image classification benchmarks: VOC and Caltech. Those results are [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">BUILD APPS WITH DEEP LEARNING</a>
</p>

## <a name="Using VGG-16">Using VGG-16 in Web Applications</a>

The first step in working with neural networks is training. During training, the network is fed input data (images, in this case), and the net's output, or guesses, are compared to the expected results (the images' labels). With each run through the data, the network's weights are modified to decrease the error rate of the guesses; that is, they are adjusted so that the guesses better match the proper label of an image. (Training a large network on millions of images can take a lot of computing resources, thus the interest of distributing pre-trained nets like this.)

Once trained, the network can be used for inference, or making predictions about the data it sees. Inference happens to be a much less compute-intensive process. For VGG-16 and other architectures, developers can download and use pre-trained models without having to master the skills necessary to tune and train those models. Loading the pre-trained models and using the model for prediction is relatively straightforward and described here. 

Loading the pre-trained models for further training is a process we will describe in a later tutorial.

## <a name="Loading Pre-Trained Models">Load the Pre-Trained Model </a>

As of 0.9.0 (0.8.1-SNAPSHOT) Deeplearning4j has a new native model zoo. Read about the [deeplearning4j-zoo](/model-zoo) module for more information on using pretrained models. Here, we load a pretrained VGG-16 model initialized with weights trained on ImageNet:

```
ZooModel zooModel = new VGG16();
ComputationGraph vgg16 = zooModel.initPretrained(PretrainedType.IMAGENET);
```

### Alternative Using Keras

Deeplearning4j also comes with a model import tool for [Keras](http://keras.io/). Our [model importer](/model-import-keras) will convert your Keras configuration and weights into the Deeplearning4j format. DL4J supports importation of both Sequential and standard functional Model classes. Code to import a model may look like:


```
ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelJsonFilename, weightsHdf5Filename, enforceTrainingConfig);

```

If you want to import a pre-trained model *for inference only*, then you should set `enforceTrainingConfig=false`. Unsupported training-only configurations generate warnings, but model import will proceed.

## <a name=">Configure Data Pipelines for Testing">Configure a Data Pipeline for Testing</a>

With data ingest and pre-processing, you can choose a manual process or the helper functions. The helper functions for VGG-16 style image processing are `TrainedModels.VGG16.getPreProcessor` and `VGG16ImagePreProcessor()`. (Remember that the images must be processed in the same way for inference as they were processed for training.) 

### VGG-16 image-processing pipeline steps

1. Scale to 224 * 224 3 layers (RGB images)
2. Mean scaling, subtract the mean pixel value from each pixel

Scaling can be handled with DataVec's native image loader. 

Mean subtraction can be done manually or with the helper functions. 

### Code Examples for scaling images to 224 height 224 width, 3 layers

If reading a directory of images, use DataVec's ImageRecordReader

```
ImageRecordReader rr = new ImageRecordReader(224,224,3);
```

If loading a single image, use DataVec's NativeImageLoader

```
NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
INDArray image = loader.asMatrix(file);
```


### Code Examples for mean subtraction

If loading a directory through DataVec's RecordReader

```
 DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
 dataIter.setPreProcessor(preProcessor);
```

If loading an image using NativeImageLoader

```
DataNormalization scaler = new VGG16ImagePreProcessor();
scaler.transform(image);
```

## <a name="Testing Pre-Trained Models">Test Pre-Trained Model</a>

Once your network is loaded, you should verify that it works as expected. Note that ImageNet was not designed for face recognition. It is better to test with a picture of an elephant, a dog, or a cat. 

If you want to compare the results with Keras output, load the model in Keras and DeepLearning4J and compare the output of each. It should be quite similar.  If Keras outputs a 35.00094% likelihood that the image is Elephant while DeepLearning4j outputs a 35.00104% likelihood that the image is Elephant, that is probably a rounding error rather than a true divergence in models.

### Code to test a directory of images

```

while (dataIter.hasNext()) {
            //prediction array
            DataSet next = dataIter.next();
            INDArray features = next.getFeatures();
            INDArray[] outputA = vgg16.output(false,features);
            INDArray output = Nd4j.concat(0,outputA);

            //print top 5 predictions for each image in the dataset
            List<RecordMetaData> trainMetaData = next.getExampleMetaData(RecordMetaData.class);
            int batch = 0;
            for(RecordMetaData recordMetaData : trainMetaData){
                System.out.println(recordMetaData.getLocation());
                System.out.println(TrainedModels.VGG16.decodePredictions(output.getRow(batch)));
                batch++;
            }

```
 

### Code to test image input from command-line prompt


```

//Buffered stream reader to provide prompt requesting file path of 
// image to be tested
InputStreamReader r = new InputStreamReader(System.in);
BufferedReader br = new BufferedReader(r);
for (; ; ){
    System.out.println("type EXIT to close");
    System.out.println("Enter Image Path to predict with VGG16");
    System.out.print("File Path: ");
    String path = br.readLine();
    if ("EXIT".equals(path))
        break;
    System.out.println("You typed" + path);


	// Code to convert submitted image to INDArray
	// apply mean subtraction and run inference
    File file = new File(path);
    NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
    INDArray image = loader.asMatrix(file);
    DataNormalization scaler = new VGG16ImagePreProcessor();
    scaler.transform(image);
    INDArray[] output = vgg16.output(false,image);
    System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));

```

## <a name="Save Model with ModelSerializer">Save model with ModelSerializer</a>

Once you've loaded and tested the model, save it using DeepLearning4J's `ModelSerializer`. Loading a model from ModelSerializer is less resource intensive than loading from Keras. Our advice is to load once, then save in the DeepLearning4J format for later re-use. 

### Code to save Model to file

```
File locationToSave = new File("vgg16.zip");
ModelSerializer.writeModel(model,locationToSave,saveUpdater);
```

### Code to Load saved model from file

```
File locationToSave = new File("/Users/tomhanlon/SkyMind/java/Class_Labs/vgg16.zip");
ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(locationToSave);

```

## <a name="Build Web App to Take Input Image">Build Web App to take input images</a>

The following HTML for a form element will present the user with a page to select and upload or "post" an image to our server. This one itself is not hooked. (WIP!) 
```
<pre>
<form method='post' action='getPredictions' enctype='multipart/form-data'>
    <input type='file' name='uploaded_file'>
    <button>Upload picture</button>
</form>
</pre>
```
The action attribute of the form element is the URL that the user-selected image will be posted to. 

We used [Spark Java](http://sparkjava.com/) for the web application as it was straightforward. Just add the Spark Java code to a class that's already written. There are many other choices available. 

Whichever Java web framework you choose, the following steps are the same. 

1. Build a form 
2. Test the form 
3. Make sure the file upload works, write some Java that tests for access to the file as a Java File object, or the string for the Path. 
4. Connect the web app functionality as input into the neural network

## <a name="Tie Web App Front End to Neural Net Backend">Tie Webapp Front End to Neural Net Backend</a>

The final working code is below. 

When this class is running, it will launch a Jetty webserver listening on port 4567. 

Starting the web app will take as much time as it takes to load the neural network. VGG-16 takes about four minutes to load. 

Once running, it uses incrementally more RAM in about 60MB chunks until it hits 4G and garbage collection cleans things up. We ran VGG-16 on an AWS t2-large instance, testing it for about a week. It was stable. It may be possible to use an even smaller AMI.  

## <a name="code">Full code example</a>


```
package org.deeplearning4j.VGGwebDemo;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import static spark.Spark.*;

/**
 * Created by tomhanlon on 1/25/17.
 */
public class VGG16SparkJavaWebApp {
    public static void main(String[] args) throws Exception {

        // Set locations for certificates for https encryption
        String keyStoreLocation = "clientkeystore";
        String keyStorePassword = "skymind";
        secure(keyStoreLocation, keyStorePassword, null,null );

        // Load the trained model
        File locationToSave = new File("vgg16.zip");
        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(locationToSave);


        // make upload directory for user submitted images
        // Images are uploaded, processed and then deleted
        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist

        // form this string displays an html form to select and upload an image
        String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
                "    <input type='file' name='uploaded_file'>\n" +
                "    <button>Upload picture</button>\n" +
                "</form>";


        // Spark Java configuration to handle requests
        // test request, the url /hello should return "hello world"
        get("/hello", (req, res) -> "Hello World");

        // Request for VGGpredict returns the form to submit an image
        get("VGGpredict", (req, res) -> form);

        // a Post request (note the form uses http post) for
        // getPredictions (note the action attribute or the form)
        // Page prints results and then another form

        post("/getPredictions", (req, res) -> {

            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }


            // The user submitted file is tempFile, convert to Java File "file"
            File file = tempFile.toFile();

            // Convert file to INDArray
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(file);

            // delete the physical file, if left our drive would fill up over time
            file.delete();

            // Mean subtraction pre-processing step for VGG
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);

            //Inference returns array of INDArray, index[0] has the predictions
            INDArray[] output = vgg16.output(false,image);

            // convert 1000 length numeric index of probabilities per label
            // to sorted return top 5 convert to string using helper function VGG16.decodePredictions
            // "predictions" is string of our results
            String predictions = TrainedModels.VGG16.decodePredictions(output[0]);

            // return results along with form to run another inference
            return "<h4> '" + predictions  + "' </h4>" +
                    "Would you like to try another" +
                    form;
            //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";

        });


    }

}

```

## <a name="example">Example predictions</a>

Here are the results given on a photo of one of the Skymind cats, which VGG-16 has probably never seen before. (He's very shy is why.) Note that the predictions below are from a 1920x1440 jpeg image. When imported to the model it is scaled to 244x244 to match the input the model was trained on. The image on this web page is a scaled version to 240x80, if you send the small image through for inference you will get different results than the results printed below. 

![a cat for inference](./../img/cat.jpeg)

	Tabby 16.7%  Tiger cat 7.6%  Toilet tissue 6.4%  Egyptian cat 5.4%  Paper towel 5.2%

For this dog found on the internet, which VGG-16 may have seen during training, the results are quite precise.

![a dog for inference](./../img/dog_320x240.png)

	53.441956%, bluetick 17.103373%, English_setter 5.808368%, kelpie 3.517581%, Greater_Swiss_Mountain_dog 2.263778%, German_short-haired_pointer'

## <a name="resources">Other Beginner's Guides for Machine Learning</a>

* [Introduction to Deep Neural Networks](./neuralnet-overview)
* [Regression & Neural Networks](./logistic-regression.html)
* [Word2vec: Neural Embeddings for Natural Language Processing](./word2vec.html)
* [Convolutional Networks](./convolutionalnets)
* [Restricted Boltzmann Machines: The Building Blocks of Deep-Belief Networks](./restrictedboltzmannmachine.html)
* [Recurrent Networks and Long Short-Term Memory Units (LSTMs)](./lstm.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network)
* [Inference: Machine Learning Model Server](./machine-learning-modelserver)
* [Beginner's Guide to Reinforcement Learning](./deepreinforcementlearning)
* [Eigenvectors, Eigenvalues, PCA & Entropy](./eigenvector)
* [Deep Reinforcement Learning](./deepreinforcementlearning)
* [Symbolic Reasoning & Deep Learning](./symbolicreasoning)
* [Graph Data & Deep Learning](./graphdata)
* [Open Data Sets for Machine Learning](./opendata)
* [ETL Data Pipelines for Machine Learning](./datavec)
* [A Glossary of Deep-Learning Terms](./glossary.html)
* [Inference: Machine Learning Model Server](./modelserver)
