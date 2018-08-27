---
title: Android for Deep Learning
short_title: Android Overview
description: Using Deep Learning and Neural Networks in Android Applications
category: Mobile
weight: 0
---

## Using Deep Learning & Neural Networks in Android Applications

Contents

* [Prerequisites](#head_link1)
* [Configuring Your Android Studio Project](#head_link2)
* [Starting an Asynchronous Task](#head_link7)
* [Creating a Neural Network](#head_link3)
* [Creating Training Data](#head_link5)
* [Conclusion](#head_link6)

Generally speaking, training a neural network is a task best suited for powerful computers with multiple GPUs. But what if you want to do it on your humble Android phone or tablet? Well, it’s definitely possible. Considering an average Android device’s specifications, however, it will most likely be quite slow. If that’s not a problem for you, keep reading.

In this tutorial, I’ll show you how to use [Deeplearning4J](https://deeplearning4j.org/quickstart), a popular Java-based deep learning library, to create and train a neural network on an Android device.


## <a name="head_link1">Prerequisites</a>

For best results, you’ll need the following:

* An Android device or emulator that runs API level 21 or higher, and has about 200 MB of internal storage space free. I strongly suggest you use an emulator first because you can quickly tweak it in case you run out of memory or storage space.
* Android Studio 2.2 or newer
* A more in-depth look at using DL4J in Android Applications can be found here. This guide covers dependencies, memory management, saving device-trained models, and loading pre-trained models in the application.


## <a name="head_link2">Configuring Your Android Studio Project</a>

To be able to use Deeplearning4J in your project, add the following compile dependencies to your app module’s build.gradle file:

``` groovy
compile (group: 'org.deeplearning4j', name: 'deeplearning4j-core', version: '{{page.version}}') {
    exclude group: 'org.bytedeco.javacpp-presets', module: 'opencv-platform'
    exclude group: 'org.bytedeco.javacpp-presets', module: 'leptonica-platform'
    exclude group: 'org.bytedeco.javacpp-presets', module: 'hdf5-platform'
}
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}'
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-arm"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-arm64"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-x86"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-x86_64"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.3.0-1.4.2'
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.3.0-1.4.2', classifier: "android-arm"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.3.0-1.4.2', classifier: "android-arm64"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.3.0-1.4.2', classifier: "android-x86"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.3.0-1.4.2', classifier: "android-x86_64"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.2-1.4.2'
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.2-1.4.2', classifier: "android-arm"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.2-1.4.2', classifier: "android-arm64"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.2-1.4.2', classifier: "android-x86"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.2-1.4.2', classifier: "android-x86_64"
compile group: 'org.bytedeco.javacpp-presets', name: 'leptonica', version: '1.76.0-1.4.2'
compile group: 'org.bytedeco.javacpp-presets', name: 'leptonica', version: '1.76.0-1.4.2', classifier: "android-arm"
compile group: 'org.bytedeco.javacpp-presets', name: 'leptonica', version: '1.76.0-1.4.2', classifier: "android-arm64"
compile group: 'org.bytedeco.javacpp-presets', name: 'leptonica', version: '1.76.0-1.4.2', classifier: "android-x86"
compile group: 'org.bytedeco.javacpp-presets', name: 'leptonica', version: '1.76.0-1.4.2', classifier: "android-x86_64"

```

If you choose to use a SNAPSHOT version of the dependencies with gradle, you will need to create the a pom.xml file in the root directory and run run ``` mvn -U compile ``` on it from the terminal. You will also need to include ``` mavenLocal() ``` in the ```  repository {} ``` block of the build.gradle file. An example pom.xml file is provided below.

``` xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>snapshots</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <dependencies>
       <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native-platform</artifactId>
            <version>1.0.0-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>1.0.0-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <repositories>
        <repository>
            <id>sonatype-nexus-snapshots</id>
            <url>https://oss.sonatype.org/content/repositories/snapshots</url>
            <releases>
                <enabled>false</enabled>
            </releases>
            <snapshots>
                <enabled>true</enabled>
                <updatePolicy>always</updatePolicy>
            </snapshots>
        </repository>
    </repositories>
</project>

```
Android Studio 3.0 introduced new Gradle, now annotationProcessors should be defined too If you are using it, add following code to gradle dependencies:

```java
NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
nncBuilder.updater(Updater.ADAM);
```
As you can see, DL4J depends on ND4J, short for N-Dimensions for Java, which is a library that offers fast n-dimensional arrays. ND4J internally depends on a library called OpenBLAS, which contains platform-specific native code. Therefore, you must load a version of OpenBLAS and ND4J that matches the architecture of your Android device. 

Dependencies of DL4J and ND4J have several files with identical names. In order to avoid build errors, add the following exclude parameters to your packagingOptions.

```groovy
packagingOptions {
    exclude 'META-INF/DEPENDENCIES'
    exclude 'META-INF/DEPENDENCIES.txt'
    exclude 'META-INF/LICENSE'
    exclude 'META-INF/LICENSE.txt'
    exclude 'META-INF/license.txt'
    exclude 'META-INF/NOTICE'
    exclude 'META-INF/NOTICE.txt'
    exclude 'META-INF/notice.txt'
    exclude 'META-INF/INDEX.LIST'
}
```
Your compiled code will have well over 65,536 methods. To be able to handle this condition, add the following option in the defaultConfig:

```groovy
multiDexEnabled true
```
And now, press Sync Now to update the project. Finally, make sure that your APK doesn't contain both lib/armeabi and lib/armeabi-v7a subdirectories. If it does, move all files to one or the other as some Android devices will have problems with both present.


## <a name="head_link7">Starting an Asynchronous Task</a>

Training a neural network is CPU-intensive, which is why you wouldn’t want to do it in your application’s UI thread. I’m not too sure if DL4J trains its networks asynchronously by default. Just to be safe, I’ll spawn a separate thread now using the AsyncTask class.

```java
AsyncTask.execute(new Runnable() {
    @Override
    public void run() {
        createAndUseNetwork();
    }
});
```

Because the method createAndUseNetwork() doesn’t exist yet, create it.

```java
private void createAndUseNetwork() {
}
```


## <a name="head_link3">Creating a Neural Network</a>

DL4J has a very intuitive API. Let us now use it to create a simple multi-layer perceptron with hidden layers. It will take two input values, and spit out one output value. To create the layers, we’ll use the DenseLayer and OutputLayer classes. Accordingly, add the following code to the createAndUseNetwork() method you created in the previous step:
``` java
DenseLayer inputLayer = new DenseLayer.Builder()
        .nIn(2)
        .nOut(3)
        .name("Input")
        .build();
DenseLayer hiddenLayer = new DenseLayer.Builder()
        .nIn(3)
        .nOut(2)
        .name("Hidden")
        .build();
OutputLayer outputLayer = new OutputLayer.Builder()
        .nIn(2)
        .nOut(1)
        .name("Output")
        .build();
```
Now that our layers are ready, let’s create a NeuralNetConfiguration.Builder object to configure our neural network.
``` java
NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
nncBuilder.updater(Updater.ADAM);
```
We must now create a NeuralNetConfiguration.ListBuilder object to actually connect our layers and specify their order.
``` java
NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
listBuilder.layer(0, inputLayer);
listBuilder.layer(1, hiddenLayer);
listBuilder.layer(2, outputLayer);
```
Additionally, enable backpropagation by adding the following code:
``` java
listBuilder.backprop(true);
```
At this point, we can generate and initialize our neural network as an instance of the MultiLayerNetwork class.

``` java
MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
myNetwork.init();
```

## <a name="head_link5">Creating Training Data</a>
To create our training data, we’ll be using the INDArray class, which is provided by ND4J. Here’s what our training data will look like:
```
INPUTS      EXPECTED OUTPUTS
------      ----------------
0,0         0
0,1         1
1,0         1
1,1         0

```
As you might have guessed, our neural network will behave like an XOR gate. The training data has four samples, and you must mention it in your code.

``` java
final int NUM_SAMPLES = 4;
```
And now, create two INDArray objects for the inputs and expected outputs, and initialize them with zeroes.

``` java
INDArray trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
INDArray trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());
```
Note that the number of columns in the inputs array is equal to the number of neurons in the input layer. Similarly, the number of columns in the outputs array is equal to the number of neurons in the output layer.

Filling those arrays with the training data is easy. Just use the putScalar() method:


``` java
// If 0,0 show 0
trainingInputs.putScalar(new int[]{0, 0}, 0);
trainingInputs.putScalar(new int[]{0, 1}, 0);
trainingOutputs.putScalar(new int[]{0, 0}, 0);
// If 0,1 show 1
trainingInputs.putScalar(new int[]{1, 0}, 0);
trainingInputs.putScalar(new int[]{1, 1}, 1);
trainingOutputs.putScalar(new int[]{1, 0}, 1);
// If 1,0 show 1
trainingInputs.putScalar(new int[]{2, 0}, 1);
trainingInputs.putScalar(new int[]{2, 1}, 0);
trainingOutputs.putScalar(new int[]{2, 0}, 1);
// If 1,1 show 0
trainingInputs.putScalar(new int[]{3, 0}, 1);
trainingInputs.putScalar(new int[]{3, 1}, 1);
trainingOutputs.putScalar(new int[]{3, 0}, 0);
```

We won’t be using the INDArray objects directly. Instead, we’ll convert them into a DataSet.
```java
DataSet myData = new DataSet(trainingInputs, trainingOutputs);
```

At this point, we can start the training by calling the ``` fit() ``` method of the neural network and passing the data set to it. The ``` for ``` loop controls the iterations of the data set through the network. It is set to 1000 iterations in this example.

```java
for(int l=0; l<=1000; l++) {
    myNetwork.fit(myData);
}
```

And that’s all there is to it. Your neural network is ready to be used.


## <a name="head_link6">Conclusion</a>

In this tutorial, you saw how easy it is to create and train a neural network using the Deeplearning4J library in an Android Studio project. I’d like to warn you, however, that training a neural network on a low-powered, battery operated device might not always be a good idea.

A second example DL4J Android Application which includes a user interface can be found [here](./deeplearning4j-android-linear-classifier). This example trains a neural network on the device using Anderson’s iris data set for iris flower type classification. The application includes user input for the measurements and returns the probability that these measurements belong to one of three iris types (*Iris serosa, Iris versicolor,* and *Iris virginica*).

The limitations of processing power and battery life on mobile devices make training robust, multi-layer networks unfeasible. As an alternative to training a network on the device, the neural network being used by your application can be trained on the desktop, saved via ModelSerializer, and then loaded as a pre-trained model in the application. A third example DL4J Android Application can be found [here](./deeplearning4j-android-image-classification) which loads a pre-trained Mnist network and uses it to classify user drawn numbers.
