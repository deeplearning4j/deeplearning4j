/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.image.loader;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.berkeley.Pair;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.EqualizeHistTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2Luv;

/**
 * Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
 * Created by nyghtowl on 12/17/15.
 */
public class CifarLoader extends NativeImageLoader implements Serializable {
    public final static int NUM_TRAIN_IMAGES = 50000;
    public final static int NUM_TEST_IMAGES = 10000;
    public final static int NUM_LABELS = 10; // Note 6000 imgs per class
    public final static int HEIGHT = 32;
    public final static int WIDTH = 32;
    public final static int CHANNELS = 3;
    public final static int BYTEFILELEN = 3073;

    public static String dataBinUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    public static String localDir = "cifar";
    public static String dataBinFile = "cifar-10-batches-bin";
    public static File fullDir = new File(BASE_DIR, FilenameUtils.concat(localDir, dataBinFile));

    //    public String dataUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"; // used for python version - similar structure to datBin structure
//    public String dataFile = "cifar-10-python";
    protected static String labelFileName = "batches.meta.txt";
    protected static InputStream inputStream;
    protected static List<DataSet> inputBatched;
    protected static List<String> labels = new ArrayList<>();

    public static String[] TRAINFILENAMES = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch5.bin"};
    public static String TESTFILENAME = "test_batch.bin";
    protected static String trainFilesSerialized = FilenameUtils.concat(fullDir.toString(), "cifar_train_serialized");
    protected static String testFilesSerialized = FilenameUtils.concat(fullDir.toString(), "cifar_test_serialized.ser");

    protected static boolean train = true;
    public static boolean preProcessCifar = false;
    public static Map<String, String> cifarTrainData = new HashMap<>();

    protected static int height = HEIGHT;
    protected static int width = WIDTH;
    protected static int channels = CHANNELS;
    protected static long seed = System.currentTimeMillis();
    protected static boolean shuffle = true;
    protected int numExamples = 0;
    protected static int numToConvertDS = 10; // TODO put  at 10000
    protected static int numToShowExamples = 10; // TODO put at 50000 not to show
    protected double uMean = 0;
    protected double uStd = 0;
    protected double vMean = 0;
    protected double vStd = 0;
    protected boolean meanStdCalc = false;

    // Using this in spark to reference where to load data from
//    static {
//        load();
//    }

    public CifarLoader() {
        this(height, width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(boolean train) {
        this(height, width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, boolean train, boolean preProcessCifar) {
        this(height, width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(boolean train, File fullPath) {
        this(height, width, channels, null, train, preProcessCifar, fullPath, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar) {
        this(height, width, channels, imgTransform, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar, boolean shuffle) {
        this(height, width, channels, imgTransform, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar, File fullPath, long seed, boolean shuffle) {
        super(height, width, channels, imgTransform);
        this.train = train;
        this.preProcessCifar = preProcessCifar;
        this.fullDir = fullPath;
        this.seed = seed;
        this.shuffle = shuffle;
        load();
    }

    // TODO preload train and test

    @Override
    public INDArray asRowVector(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asRowVector(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    public void generateMaps() {
        cifarTrainData.put("filesFilename", new File(dataBinUrl).getName());
        cifarTrainData.put("filesURL", dataBinUrl);
        cifarTrainData.put("filesFilenameUnzipped", dataBinFile);
    }

    private void defineLabels() {
        try {
            File path = new File(fullDir, labelFileName);
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;

            while ((line = br.readLine()) != null) {
                labels.add(line);
                // TODO resolve duplicate listing
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load() {
        if (!cifarRawFilesExist() && !fullDir.exists()) {
            generateMaps();
            fullDir.mkdir();

            log.info("Downloading {}...", localDir);
            downloadAndUntar(cifarTrainData, fullDir);
        }
        try {
            // Create inputStream
            if (train) {
                Collection<File> subFiles = FileUtils.listFiles(fullDir, new String[]{"bin"}, true);
                Iterator trainIter = subFiles.iterator();
                inputStream = new SequenceInputStream(new FileInputStream((File) trainIter.next()), new FileInputStream((File) trainIter.next()));
                while (trainIter.hasNext()) {
                    File nextFile = (File) trainIter.next();
                    if (!TESTFILENAME.equals(nextFile.getName()))
                        inputStream = new SequenceInputStream(inputStream, new FileInputStream(nextFile));
                }
            } else
                inputStream = new FileInputStream(new File(fullDir, TESTFILENAME));
        } catch (Exception e) {
            e.printStackTrace();
        }

        defineLabels();

        if (preProcessCifar) {
            for (int i = 1; i <= (TRAINFILENAMES.length); i++) {
                DataSet result = convertDataSet(numToConvertDS);
                // TODO need to trigger whether it exists or not
                result.save(new File(trainFilesSerialized + i + ".ser"));
            }
        }
    }

    public boolean cifarRawFilesExist() {
        File f = new File(fullDir, TESTFILENAME);
        if (!f.exists()) return false;

        for (String name : TRAINFILENAMES) {
            f = new File(fullDir, name);
            if (!f.exists()) return false;
        }
        return true;
    }

    private boolean cifarProcessedFilesExists() {
        File f;
        if(train) {
            f = new File(trainFilesSerialized + 1 + ".ser");
            if (!f.exists()) return false;
        } else {
            f = new File(testFilesSerialized);
            if (!f.exists()) return false;
        }
        return true;
    }

    /**
     * Preprocess and store cifar data if it does not exist
     */
    public opencv_core.Mat preProcessCifar(Mat orgImage) {
        numExamples ++;
        Mat resImage = new Mat();
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        ImageTransform yuvTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2Luv);
        ImageTransform histEqualization = new EqualizeHistTransform(new Random(seed), COLOR_BGR2Luv);

        if (converter != null) {
            ImageWritable writable = new ImageWritable(converter.convert(orgImage));
//            if(numExamples % 1000 == 0){
//        ImageTransform showOrig = new ShowImageTransform("Original Image", 50);
//                showOrig.transform(writable);
//            }
            writable = yuvTransform.transform(writable);
            // TODO confirm this works and does sparse contrast ...
            writable = histEqualization.transform(writable);
            if(numExamples % numToShowExamples == 0){
                ImageTransform showTrans = new ShowImageTransform("Transform Image", 50);
                showTrans.transform(writable);
            }
            resImage = converter.convert(writable.getFrame());
        }
        resImage = scalingIfNeed(resImage);

        return resImage;
    }

    public Pair<INDArray, opencv_core.Mat> convertMat(byte[] byteFeature) {
        INDArray label; // first value in the 3073 byte array
        opencv_core.Mat image = new opencv_core.Mat(height, width, CV_8UC(channels)); // feature are 3072
        ByteBuffer imageData = image.createBuffer();

        label = FeatureUtil.toOutcomeVector(byteFeature[0], NUM_LABELS);
        for (int i = 0; i < height * width; i++) {
            imageData.put(3 * i, byteFeature[i + 1 + 2 * height * width]); // blue
            imageData.put(3 * i + 1, byteFeature[i + 1 + height * width]); // green
            imageData.put(3 * i + 2, byteFeature[i + 1]); // red
        }
        if (preProcessCifar) {
            image = preProcessCifar(image);
        }
        return new Pair<>(label, image);
    }


    public DataSet convertDataSet(int num) {
        int batchNumCount = 0;
        List<DataSet> dataSets = new ArrayList<>();
        Pair<INDArray, opencv_core.Mat> matConversion;
        byte[] byteFeature = new byte[BYTEFILELEN];

        try {
            while (inputStream.read(byteFeature) != -1 && batchNumCount != num) {
                matConversion = convertMat(byteFeature);
                try {
                    dataSets.add(new DataSet(asMatrix(matConversion.getSecond()), matConversion.getFirst()));
                    batchNumCount++;
                } catch (Exception e) {
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataSet result = null;
        try {
            result = result.merge(dataSets);
        } catch (IllegalArgumentException e) {
            return result;
        }
        double uTempMean, vTempMean;
        for (DataSet data :result) {
            if (preProcessCifar) {
                INDArray uChannel = data.getFeatures().tensorAlongDimension(1, new int[] {0,2,3});
                INDArray vChannel = data.getFeatures().tensorAlongDimension(2, new int[] {0,2,3});
                uTempMean = uChannel.mean(new int[] {0,2,3}).getDouble(0);
                uStd += varTemp(uChannel, uTempMean);
                uMean += uTempMean;
                vTempMean = vChannel.mean(new int[] {0,2,3}).getDouble(0);
                vStd += varTemp(vChannel, vTempMean);
                vMean += vTempMean;
                // TODO add this to test?
                double totalLen = data.getFeatureMatrix().ravel().length();
                double uLen = uChannel.ravel().length();
                double vLen = vChannel.ravel().length();
            }
            // TODO don't ravel - just bring back 4D - once Alex's changes fully covers cnnsetup
            data.setFeatures(data.getFeatureMatrix().ravel());
        }
        if(shuffle) result.shuffle(seed); // TODO confirm shuffle not same on mult epochs with set seed...
        return result;
    }

    public double varTemp(INDArray x, double mean) {
        // TODO variance result is incorrect based on dimensions passed in - log issue and test example
        INDArray xSubMean = x.sub(mean);
        INDArray squared = xSubMean.muli(xSubMean);
        double accum = Nd4j.getExecutioner().execAndReturn(new Sum(squared)).getFinalResult().doubleValue();
        return accum/x.ravel().length();
    }

    public DataSet next(int batchSize) {
        return next(batchSize, 1, 0);
    }

    public DataSet next(int batchSize, int fileNum, int batchNum) {
        DataSet result =  new DataSet();
        if (cifarProcessedFilesExists()) {
            if (batchNum == 0) {
                result.load(new File(trainFilesSerialized + fileNum + ".ser"));
                if (train) {
                    if(!meanStdCalc) {
                        uMean = Math.abs(uMean/numExamples);
                        uStd = Math.sqrt(uStd);
                        vMean = Math.abs(vMean/numExamples);
                        vStd = Math.sqrt(vStd);
                        // TODO store mean and variance
                        meanStdCalc = true;
                        int t = result.numExamples();
                        for (int i = 0; i < result.numExamples(); i++) {
                            INDArray newFeatures = result.get(i).getFeatureMatrix();
                            // TODO numbers don't appear to be applied right - confirm dim
                            newFeatures.tensorAlongDimension(1, new int[] {0,2,3}).subi(uMean).divi(uStd);
                            newFeatures.tensorAlongDimension(2, new int[] {0,2,3}).subi(vMean).divi(vStd);
                            result.get(i).setFeatures(newFeatures);
                        }
                        result.save(new File(trainFilesSerialized + fileNum + ".ser"));
                    }
                }
                if(shuffle) result.shuffle(seed);
                inputBatched = result.batchBy(batchSize);
            }
            result = inputBatched.get(batchNum);
        } else {
            result = convertDataSet(batchSize);
        }
        return result;
    }

    public InputStream getInputStream() {
        return inputStream;
    }

    public List<String> getLabels(){
        return labels;
    }

    public void reset(){
        numExamples = 0;
        load();
    }


}
