---
title: Using DL4J for Android Image Classification
short_title: Android Image Classifier
description: How to create an Android Image Classification app with Eclipse Deeplearning4j.
category: Mobile
weight: 3
---

## Using Deeplearning4J in Android Applications

Contents

* [Setting the Dependencies](#head_link1)
* [Training and loading the Mnist model in the Android project resources](#head_link2)
* [Accessing the trained model using an AsyncTask](#head_link7)
* [Handling images from user input](#head_link3)
* [Updating the UI](#head_link5)
* [Conclusion](#head_link6)


## DL4JImageRecognitionDemo

This example application uses a neural network trained on the standard MNIST dataset of 28x28 greyscale 0..255 pixel value images of hand drawn numbers 0..9. The application user interace allows the user to draw a number on the device screen which is then tested against the trained network. The output displays the most probable numeric values and the probability score. This tutorial will cover the use of a trained neural network in an Android Application, the handling of user generated images, and the output of the results to the UI from a background thread. For a detailed guide demonstrating how to train and save the neural networks used in this application, please see this DL4J quickstart [tutorial](https://deeplearning4j.org/quickstart). More information on general prerequisites for building DL4J Android Applications can be found [here](https://deeplearning4j.org/android-prerequisites-configuration). 

![](/images/guide/screen2.png)


## <a name="head_link1">Setting the Dependencies</a>

Deeplearning4J applications requires application specific dependencies in the build.gradle file. The Deeplearning library in turn depends on the libraries of ND4J and OpenBLAS, thus these must also be added to the dependencies declaration. Starting with Android Studio 3.0, annotationProcessors need to be defined as well, thus dependencies for either -x86 or -arm processors should be included, depending on your device, if you are working in Android Studio 3.0 or later. Note that both can be include without conflict as is done in the example app.
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

implementation 'com.google.code.gson:gson:2.8.2'
annotationProcessor 'org.projectlombok:lombok:1.16.16'

//This corrects for a junit version conflict.
configurations.all {
    resolutionStrategy.force 'junit:junit:4.12'
}
```

Compiling these dependencies involves a large number of files, thus it is necessary to set multiDexEnabled to true in defaultConfig.
```java
multiDexEnabled true
```
Finally, a conflict in the junit module versions will give the following error: > Conflict with dependency 'junit:junit' in project ':app'. Resolved versions for app (4.8.2) and test app (4.12) differ.
This can be suppressed by forcing  all of the junit modules to use the same version.
```java
configurations.all {
    resolutionStrategy.force 'junit:junit:4.12'
}
```
## <a name="head_link2">Training and loading the Mnist model in the Android project resources</a>

Using a neural  network requires a significant amount of processor power, which is in limited supply on mobile devices. Therefore, a background thread must be used for loading of the trained neural network and the testing of the user drawn image by using AsyncTask. In this application we will run the canvas.draw code on the main thread and use an AsyncTask to load the drawn image from internal memory and test it against the trained model on a background thread. First, lets look at how to save the trained neural network we will be using in the application.

You will need to begin by following the DeepLearning4J quick start [guide](https://deeplearning4j.org/quickstart) to set up, train, and save neural network models on a desktop computer. The DL4J example which trains and saves the Mnist model used in this application is *MnistImagePipelineExampleSave.java* and is included in the quick start guide referenced above. The code for the Mnist demo is also available [here](https://gist.github.com/tomthetrainer/7cb2fbc14a5c631a567a98c3134f7dd6). Running this demo will train the Mnist neural network model and save it as *"trained_mnist_model.zip"* in the *dl4j\target folder* of the *dl4j-examples* directory. You can then copy the file and save it in the raw folder of your Android project.

![](/images/guide/rawFolder.PNG)

## <a name="head_link7">Accessing the trained model using an AsyncTask</a>

Now let’s start by writing our AsyncTask<*Params*, *Progress*, *Results*> to load and use the neural network on a background thread. The AsyncTask will use the parameter types <String, Integer, INDArray>. The *Params* type is set to String, which will pass the Path for the saved image to the asyncTask as it is executed. This path will be used in the doInBackground() method to locate and load the trained Mnist model. The *Results* parameter is of type INDArray which will store the results from the neural network and pass it to the onPostExecute method that has access to the main thread for updating the UI. For more on NDArrays, see https://nd4j.org/userguide. Note that the AsyncTask requires that we override two more methods (the onProgressUpdate and onPostExecute methods) which we will get to later in the demo.
```java
private class AsyncTaskRunner extends AsyncTask<String, Integer, INDArray> {

        // Runs in UI before background thread is called. 
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected INDArray doInBackground(String... params) {
            // Main background thread, this will load the model and test the input image
	    // The dimensions of the images are set here
            int height = 28;
            int width = 28;
            int channels = 1;

            //Now we load the model from the raw folder with a try / catch block
            try {
                // Load the pretrained network.
                InputStream inputStream = getResources().openRawResource(R.raw.trained_mnist_model);
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(inputStream);

                //load the image file to test
                File f=new File(absolutePath, "drawn_image.jpg");

                //Use the nativeImageLoader to convert to numerical matrix
                NativeImageLoader loader = new NativeImageLoader(height, width, channels);

                //put image into INDArray
                INDArray image = loader.asMatrix(f);

                //values need to be scaled
                DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

                //then call that scalar on the image dataset
                scalar.transform(image);

                //pass through neural net and store it in output array
                output = model.output(image);

            } catch (IOException e) {
                e.printStackTrace();
            }
            return output;
        }
```

## <a name="head_link3">Handling images from user input</a>
 
Now lets add the code for the drawing canvas that will run on the main thread and allow the user to draw a number on the screen. This is a generic draw program written as an inner class within the MainActivity. It extends View and overrides a series of methods. The drawing is saved to internal memory and the AsyncTask is executed with the image Path passed to it in the onTouchEvent case statement for case *MotionEvent.ACTION_UP*. This has the streamline action of automatically returning results for an image after the user completes the drawing. 
```java
//code for the drawing input
    public class DrawingView extends View {

        private Path    mPath;
        private Paint   mBitmapPaint;
        private Paint   mPaint;
        private Bitmap  mBitmap;
        private Canvas  mCanvas;

        public DrawingView(Context c) {
            super(c);

            mPath = new Path();
            mBitmapPaint = new Paint(Paint.DITHER_FLAG);
            mPaint = new Paint();
            mPaint.setAntiAlias(true);
            mPaint.setStrokeJoin(Paint.Join.ROUND);
            mPaint.setStrokeCap(Paint.Cap.ROUND);
            mPaint.setStrokeWidth(60);
            mPaint.setDither(true);
            mPaint.setColor(Color.WHITE);
            mPaint.setStyle(Paint.Style.STROKE);
        }

        @Override
        protected void onSizeChanged(int W, int H, int oldW, int oldH) {
            super.onSizeChanged(W, H, oldW, oldH);
            mBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_4444);
            mCanvas = new Canvas(mBitmap);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            canvas.drawBitmap(mBitmap, 0, 0, mBitmapPaint);
            canvas.drawPath(mPath, mPaint);
        }

        private float mX, mY;
        private static final float TOUCH_TOLERANCE = 4;

        private void touch_start(float x, float y) {
            mPath.reset();
            mPath.moveTo(x, y);
            mX = x;
            mY = y;
        }
        private void touch_move(float x, float y) {
            float dx = Math.abs(x - mX);
            float dy = Math.abs(y - mY);
            if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
                mPath.quadTo(mX, mY, (x + mX)/2, (y + mY)/2);
                mX = x;
                mY = y;
            }
        }
        private void touch_up() {
            mPath.lineTo(mX, mY);
            mCanvas.drawPath(mPath, mPaint);
            mPath.reset();
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            float x = event.getX();
            float y = event.getY();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    invalidate();
                    clear();
                    touch_start(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_MOVE:
                    touch_move(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_UP:
                    touch_up();
                    absolutePath = saveDrawing();
                    invalidate();
                    clear();
                    loadImageFromStorage(absolutePath);
                    onProgressBar();
                    //launch the asyncTask now that the image has been saved
                    AsyncTaskRunner runner = new AsyncTaskRunner();
                    runner.execute(absolutePath);
                    break;

            }
            return true;
        }

        public void clear(){
            mBitmap.eraseColor(Color.TRANSPARENT);
            invalidate();
            System.gc();
        }

    }

```
Now we need to build a series of helper methods. First we will write the saveDrawing() method. It uses getDrawingCache() to retrieve the drawing from the drawingView and store it as a bitmap. We then create a file directory and file for the bitmap called "drawn_image.jpg". Finally, FileOutputStream is used in a try / catch block to write the bitmap to the file location. The method returns the absolute Path to the file location which will be used by the loadImageFromStorage() method. 
```java
public String saveDrawing(){
        drawingView.setDrawingCacheEnabled(true);
        Bitmap b = drawingView.getDrawingCache();

        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // set the path to storage
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir and store the file there. Each new drawing will overwrite the previous
        File mypath=new File(directory,"drawn_image.jpg");

        //use a fileOutputStream to write the file to the location in a try / catch block
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            b.compress(Bitmap.CompressFormat.JPEG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }
```

Next we will write the loadImageFromStorage method which will use the absolute path returned from saveDrawing() to load the saved image and display it in the UI as part of the output display. It uses a try / catch block and a FileInputStream to set the image to the ImageView *img* in the UI layout.

```java
    private void loadImageFromStorage(String path)
    {

        //use a fileInputStream to read the file in a try / catch block
        try {
            File f=new File(path, "drawn_image.jpg");
            Bitmap b = BitmapFactory.decodeStream(new FileInputStream(f));
            ImageView img=(ImageView)findViewById(R.id.outputView);
            img.setImageBitmap(b);
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }

    }
```

We also need to write two methods that extract the predicted number from the neural network output and the confidence score, which we will call later when we complete the AsyncTask. 

```java
//helper class to return the largest value in the output array
    public static double arrayMaximum(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for(double cur: arr)
            max = Math.max(max, cur);
        return max;
    }

    // helper class to find the index (and therefore numerical value) of the largest confidence score
    public int getIndexOfLargestValue( double[] array )
    {
        if ( array == null || array.length == 0 ) return -1;
        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {if ( array[i] > array[largest] ) largest = i;            }
        return largest;
    }
```

Finally, we need a few methods we can call to control the visibility of an 'In Progress...' message while the background thread is running. These will be called when the AsyncTask is executed and in the onPostExecute method when the background thread completes.

```java
    public void onProgressBar(){
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.VISIBLE);
    }

    public void offProgressBar(){
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.INVISIBLE);
    }
```

Now let's go to the onCreate method to initialize the draw canvas and set some global variables.

```java
public class MainActivity extends AppCompatActivity {

    MainActivity.DrawingView drawingView;
    String absolutePath;
    public static INDArray output;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        RelativeLayout parent = findViewById(R.id.layout2);
        drawingView = new MainActivity.DrawingView(this);
        parent.addView(drawingView);
    }
```

## <a name="head_link5">Updating the UI</a>

Now we can complete our AsyncTask by overriding the onProgress and onPostExecute methods. Once the doInBackground method of AsyncTask completes, the classification results will be passed to the onPostExecute which has access to the main thread and UI allowing us to update the UI with the results. Since we will not be using the onProgress  method, a call to its superclass will suffice.

```java
@Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }
```

The onPostExecute method will receive an INDArray which contains the neural network results as a 1x10 array of probability values that the input drawing is each possible digit (0..9). From this we need to determine which row of the array contains the largest value and what the size of that value is. These two values will determine which number the neural network has classified the drawing as and how confident the network score is. These values will be referred to in the UI as *Prediction* and the *Confidence*, respectively. In the code below, the individual values for each position of the INDArray are passed to an array of type double using the getDouble() method on the result INDArray. We then get references to the TextViews which will be updated in the UI and call our helper methods on the array to return the array maximum (confidence) and index of the largest value (prediction). Note we also need to limit the number of decimal places reported on the probabilities by setting a DecimalFormat pattern.

```java

        @Override
        protected void onPostExecute(INDArray result) {
            super.onPostExecute(result);

            //used to control the number of decimals places for the output probability
            DecimalFormat df2 = new DecimalFormat(".##");

            //transfer the neural network output to an array
            double[] results = {result.getDouble(0,0),result.getDouble(0,1),result.getDouble(0,2),
                    result.getDouble(0,3),result.getDouble(0,4),result.getDouble(0,5),result.getDouble(0,6),
                    result.getDouble(0,7),result.getDouble(0,8),result.getDouble(0,9),};

            //find the UI tvs to display the prediction and confidence values
            TextView out1 = findViewById(R.id.prediction);
            TextView out2 = findViewById(R.id.confidence);

            //display the values using helper functions defined below
            out2.setText(String.valueOf(df2.format(arrayMaximum(results))));
            out1.setText(String.valueOf(getIndexOfLargestValue(results)));

            //helper function to turn off progress test
            offProgressBar();
        }
```

## <a name="head_link6">Conclusion</a>

This tutorial provides a basic framework for image recognition in an Android Application using a DL4J neural network. It illustrates how to load a pre-trained DL4J model from the raw resources file and how to test user generate input images against the model. The AsyncTask then returns the output to the main thread and updates the UI.

The complete code for this example is available [here.](https://github.com/deeplearning4j/dl4j-examples/tree/master/android/DL4JImageRecognitionDemo)