---
title: Android Classifier with DL4J
short_title: Android Classifier
description: How to create an IRIS classifier on Android using Eclipse Deeplearning4j.
category: Mobile
weight: 2
---

# IRIS Classifier Demo

The example application trains a small neural network on the device using Anderson’s Iris data set for iris flower type classification. For a more indepth look at optimizing android for DL4J, please see the Prerequisites and Configuration documentation [here](./deeplearning4j-android-prerequisites). This application has a simple UI to take measurements of petal length, petal width, sepal length, and sepal width from the user and returns the probability that the measurements belong to one of three types of Iris (*Iris serosa*, *Iris versicolor*, and *Iris virginica*). A data set includes 150 measurement values (50 for each iris type) and training the model takes anywhere from 5-20 seconds, depending on the device.

Contents

* [Setting the Dependencies](#head_link1)
* [Setting up the neural network on a background thread](#head_link2)
* [Preparing the training data set and user input](#head_link3)
* [Building and Training the Neural Network](#head_link4)
* [Updating the UI](#head_link5)
* [Conclusion](#head_link6)


## DL4JIrisClassifierDemo

![](/images/guide/screen.PNG)

## <a name="head_link1">Setting the Dependencies</a>
Deeplearning4J applications require several dependencies in the build.gradle file. The Deeplearning library in turn depends on the libraries of ND4J and OpenBLAS, thus these must also be added to the dependencies declaration. Starting with Android Studio 3.0, annotationProcessors need to be defined as well, requiring dependencies for -x86 or -arm processors. 
```java
compile (group: 'org.deeplearning4j', name: 'deeplearning4j-nn', version: '{{page.version}}') {
    exclude group: 'org.bytedeco.javacpp-presets', module: 'opencv-platform'
    exclude group: 'org.bytedeco.javacpp-presets', module: 'leptonica-platform'
    exclude group: 'org.bytedeco.javacpp-presets', module: 'hdf5-platform'
}
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}'
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-arm"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-arm64"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-x86"
compile group: 'org.nd4j', name: 'nd4j-native', version: '{{page.version}}', classifier: "android-x86_64"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.2.20-{{presetsversion}}', classifier: "android-arm"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.2.20-{{presetsversion}}', classifier: "android-arm64"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.2.20-{{presetsversion}}', classifier: "android-x86"
compile group: 'org.bytedeco.javacpp-presets', name: 'openblas', version: '0.2.20-{{presetsversion}}', classifier: "android-x86_64"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.1-{{presetsversion}}', classifier: "android-arm"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.1-{{presetsversion}}', classifier: "android-arm64"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.1-{{presetsversion}}', classifier: "android-x86"
compile group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.1-{{presetsversion}}', classifier: "android-x86_64"
```

Compiling these dependencies involves a large number of files, thus it is necessary to set multiDexEnabled to true in defaultConfig.

```java
multiDexEnabled true
```

Finally, a conflict in the junit module versions will likely throw the following error: > Conflict with dependency 'junit:junit' in project ':app'. Resolved versions for app (4.8.2) and test app (4.12) differ.
This can be suppressed by forcing all of the junit modules to use the same version.

```java
configurations.all {
    resolutionStrategy.force 'junit:junit:4.12'
}
```


## <a name="head_link2">Setting up the neural network on a background thread</a>

Training even a simple neural network like in this example requires a significant amount of processor power, which is in limited supply on mobile devices. Thus, it is imperative that a background thread be used for the building and training of the neural network which then returns the output to the main thread for updating the UI. In this example we will be using an AsyncTask which accepts the input measurements from the UI and passes them as type double to the doInBackground() method. First, lets get references to the editTexts in the UI layout that accept the iris measurements inside of our onCreate method. Then an onClickListener will execute our asyncTask, pass it the measurements entered by the user, and show a progress bar until we hide it again in onPostExecute().

```java
public class MainActivity extends AppCompatActivity {
 
 
@Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
 
        //get references to the editTexts that take the measurements
        final EditText PL = (EditText) findViewById(R.id.editText);
        final EditText PW = (EditText) findViewById(R.id.editText2);
        final EditText SL = (EditText) findViewById(R.id.editText3);
        final EditText SW = (EditText) findViewById(R.id.editText4);
 
	  //onclick to capture the input and launch the asyncTask
        Button button = (Button) findViewById(R.id.button);
 
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
 
                final double pl = Double.parseDouble(PL.getText().toString());
                final double pw = Double.parseDouble(PW.getText().toString());
                final double sl = Double.parseDouble(SL.getText().toString());
                final double sw = Double.parseDouble(SW.getText().toString());
 
                AsyncTaskRunner runner = new AsyncTaskRunner();
 
		   //pass the measurement as params to the AsyncTask
                runner.execute(pl,pw,sl,sw);
 
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
        }
```

Now let’s write our AsyncTask<*Params*, *Progress*, *Results*>. The AsyncTask needs to have a *Params* of type Double to receive the decimal value measurements from the UI. The *Result* type is set to INDArray, which is returned from the doInBackground() Method and passed to the onPostExecute() method for updating the UI. NDArrays are provided by the ND4J library and are essentially n-dimensional arrays with a given number of dimensions. For more on NDArrays, see https://nd4j.org/userguide. 

```java
private class AsyncTaskRunner extends AsyncTask<Double, Integer, INDArray> {
 
    // Runs in UI before background thread is called
    @Override
    protected void onPreExecute() {
        super.onPreExecute();
 
        ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
        bar.setVisibility(View.INVISIBLE);
    }
```


## <a name="head_link3">Preparing the training data set and user input</a>

The doInBackground() method will handle the formatting of the training data, the construction of the neural net, the training of the net, and the analysis of the input data by the trained model. The user input has only 4 values, thus we can add those directly to a 1x4 INDArray using the putScalar() method. The training data is much larger and must be converted from CSV lists to matrices through an iterative *for* loop. 
 
The training data is stored in the app as two arrays, one for the Iris measurements named *irisData* which contains a list of 150 iris measurements and another for the labels of iris type named *labelData*. These will be transformed to 150x4 and 150x3 matrices, respectively, so that they can be converted into INDArray objects that the neural network will use for training. 

```java
    // This is our main background thread for the neural net
    @Override
    protected String doInBackground(Double... params) {
    //Get the doubles from params, which is an array so they will be 0,1,2,3
        double pld = params[0];
        double pwd = params[1];
        double sld = params[2];
        double swd = params[3];
     
        //Create input INDArray for the user measurements
        INDArray actualInput = Nd4j.zeros(1,4);
        actualInput.putScalar(new int[]{0,0}, pld);
        actualInput.putScalar(new int[]{0,1}, pwd);
        actualInput.putScalar(new int[]{0,2}, sld);
        actualInput.putScalar(new int[]{0,3}, swd);
     
        //Convert the iris data into 150x4 matrix
        int row=150;
        int col=4;
        double[][] irisMatrix=new double[row][col];
        int i = 0;
        for(int r=0; r<row; r++){
            for( int c=0; c<col; c++){
        irisMatrix[r][c]=com.example.jmerwin.irisclassifier.DataSet.irisData[i++];
            }
        }
     
        //Now do the same for the label data
        int rowLabel=150;
        int colLabel=3;
        double[][] twodimLabel=new double[rowLabel][colLabel];
        int ii = 0;
        for(int r=0; r<rowLabel; r++){
            for( int c=0; c<colLabel; c++){
                twodimLabel[r][c]=com.example.jmerwin.irisclassifier.DataSet.labelData[ii++];
            }
        }
     
        //Converting the data matrices into training INDArrays is straight forward
        INDArray trainingIn = Nd4j.create(irisMatrix);
        INDArray trainingOut = Nd4j.create(twodimLabel);
```
## <a name="head_link4">Building and Training the Neural Network</a>

Now that our data is ready, we can build a simple multi-layer perceptron with a single hidden layer. The *DenseLayer* class is used to create the input layer and the hidden layer of the network while the *OutputLayer* class is used for the Output layer. The number of columns in the input INDArray must equal to the number of neurons in the input layer (nIn). The number of neurons in the hidden layer input must equal the number inputLayer’s output array (nOut). Finally, the outputLayer input should match the hiddenLayer output. The output must equal the number of possible classifications, which is 3.

```java
    //define the layers of the network
    DenseLayer inputLayer = new DenseLayer.Builder()
            .nIn(4)
            .nOut(3)
            .name("Input")
            .build();
 
    DenseLayer hiddenLayer = new DenseLayer.Builder()
            .nIn(3)
            .nOut(3)
            .name("Hidden")
            .build();
 
    OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(3)
            .nOut(3)
            .name("Output")
            .activation(Activation.SOFTMAX)
            .build();
```
The next step is to build the neural network using *nccBuilder*. The parameters selected below for training are standard. To learn more about optimizing network training, see deeplearning4j.org.
```java
    NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
    long seed = 6;
    nncBuilder.seed(seed);
    nncBuilder.activation(Activation.TANH);
    nncBuilder.weightInit(WeightInit.XAVIER);
     
    NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
    listBuilder.layer(0, inputLayer);
    listBuilder.layer(1, hiddenLayer);
    listBuilder.layer(2, outputLayer);
     
    listBuilder.backprop(true);
     
    MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
    myNetwork.init();
     
    //Create a data set from the INDArrays and train the network
    DataSet myData = new DataSet(trainingIn, trainingOut);
    for(int l=0; l<=1000; l++) {
    myNetwork.fit(myData);
    }
     
    //Evaluate the input data against the model
    INDArray actualOutput = myNetwork.output(actualInput);
    Log.d("myNetwork Output ", actualOutput.toString());
     
    //Here we return the INDArray to onPostExecute where it can be 
    //used to update the UI
    return actualOutput;
}
```
## <a name="head_link5">Updating the UI</a>

Once the training of the neural network and the classification of the user measurements are complete, the doInBackground() method will finish and onPostExecute() will have access to the main thread and UI, allowing us to update the UI with the classification results. Note that the decimal places reported on the probabilities can be controlled by setting a DecimalFormat pattern.
```java
//This is where we update the UI with our classification results
    @Override
    protected void onPostExecute(INDArray result) {
        super.onPostExecute(result);
 
    //Hide the progress bar now that we are finished
    ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
    bar.setVisibility(View.INVISIBLE);
 
    //Retrieve the three probabilities
    Double first = result.getDouble(0,0);
    Double second = result.getDouble(0,1);
    Double third = result.getDouble(0,2);
 
    //Update the UI with output
    TextView setosa = (TextView) findViewById(R.id.textView11);
    TextView versicolor = (TextView) findViewById(R.id.textView12);
    TextView virginica = (TextView) findViewById(R.id.textView13);
 
    //Limit the double to values to two decimals using DecimalFormat
    DecimalFormat df2 = new DecimalFormat(".##");
 
    //Set the text of the textViews in UI to show the probabilites
    setosa.setText(String.valueOf(df2.format(first)));
    versicolor.setText(String.valueOf(df2.format(second)));
    virginica.setText(String.valueOf(df2.format(third)));
 
    }
```


## <a name="head_link6">Conclusion</a>

Hopefully this tutorial has illustrated how the compatibility of DL4J with Android makes it easy to build, train, and evaluate neural networks on mobile devices. We used a simple UI to take input values from the measurement and then passed them as the *Params* in an AsyncTask. The processor intensive steps of data preparation, network layer building, model training, and evaluation of the user data were all performed in the doInBackground() method of the background thread, maintaining a stable and responsive device. Once completed, we passed the output INDArray as the AsyncTask *Results* to onPostExecute() where the the UI was updated to demonstrate the classification results. 
The limitations of processing power and battery life of mobile devices make training robust, multi-layer networks somewhat unfeasible. To address this limitation, we will next look at an example Android application that saves the trained model on the device for faster performance after an initial model training.

The complete code for this example is available [here.](https://github.com/deeplearning4j/dl4j-examples/tree/master/android/DL4JIrisClassifierDemo)



