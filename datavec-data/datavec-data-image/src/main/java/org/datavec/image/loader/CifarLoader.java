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
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb;

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
    public static File meanVarPath = new File(fullDir, "meanVarPath.txt");

    protected static String labelFileName = "batches.meta.txt";
    protected static InputStream inputStream;
    protected static InputStream trainInputStream;
    protected static InputStream testInputStream;
    protected static List<DataSet> inputBatched;
    protected static List<String> labels = new ArrayList<>();

    public static String[] TRAINFILENAMES = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch5.bin"};
    public static String TESTFILENAME = "test_batch.bin";
    protected static String trainFilesSerialized = FilenameUtils.concat(fullDir.toString(), "cifar_train_serialized");
    protected static String testFilesSerialized = FilenameUtils.concat(fullDir.toString(), "cifar_test_serialized.ser");
    protected static String cifarStats = FilenameUtils.concat(fullDir.toString(), "cifar_stats.csv");
    protected static boolean train = true;
    public static boolean preProcessCifar = false;
    public static Map<String, String> cifarDataMap = new HashMap<>();

    protected static int height = HEIGHT;
    protected static int width = WIDTH;
    protected static int channels = CHANNELS;
    protected static long seed = System.currentTimeMillis();
    protected static boolean shuffle = true;
    protected int numExamples = 0;
    protected static int numToConvertDS = 1000; // TODO put  at 10000
    protected static int numToShowExamples = 10000; // TODO put at 50000 not to show
    protected double uMean = 0;
    protected double uStd = 0;
    protected double vMean = 0;
    protected double vStd = 0;
    protected boolean meanStdStored = false;
    protected int idx = 0;
    protected DataSet loadDS = new DataSet();

    public CifarLoader() {
        this(height,width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(boolean train) {
        this(height, width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(boolean train, File fullPath) {
        this(height, width, channels, null, train, preProcessCifar, fullPath, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, boolean train, boolean preProcessCifar) {
        this(height, width, channels, null, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar) {
        this(height, width, channels, imgTransform, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar, boolean shuffle) {
        this(height, width, channels, imgTransform, train, preProcessCifar, fullDir, seed, shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar, File fullPath, long seed, boolean shuffle) {
        super(height, width, channels, imgTransform);
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.train = train;
        this.preProcessCifar = preProcessCifar;
        this.fullDir = fullPath;
        this.seed = seed;
        this.shuffle = shuffle;
        load();
    }



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
        cifarDataMap.put("filesFilename", new File(dataBinUrl).getName());
        cifarDataMap.put("filesURL", dataBinUrl);
        cifarDataMap.put("filesFilenameUnzipped", dataBinFile);
    }

    private void defineLabels() {
        try {
            File path = new File(fullDir, labelFileName);
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;

            while ((line = br.readLine()) != null) {
                labels.add(line);
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
            downloadAndUntar(cifarDataMap, new File(BASE_DIR, localDir));
        }
        try {
            Collection<File> subFiles = FileUtils.listFiles(fullDir, new String[]{"bin"}, true);
            Iterator trainIter = subFiles.iterator();
            trainInputStream = new SequenceInputStream(new FileInputStream((File) trainIter.next()), new FileInputStream((File) trainIter.next()));
            while (trainIter.hasNext()) {
                File nextFile = (File) trainIter.next();
                if (!TESTFILENAME.equals(nextFile.getName()))
                    trainInputStream = new SequenceInputStream(trainInputStream, new FileInputStream(nextFile));
            }
            testInputStream = new FileInputStream(new File(fullDir, TESTFILENAME));
        } catch (Exception e) {
            e.printStackTrace();
        }

        defineLabels();

        if (preProcessCifar && train && !cifarProcessedFilesExists()) {
            for (int i = 1; i <= (TRAINFILENAMES.length); i++) {
                inputStream = trainInputStream;
                DataSet result = convertDataSet(numToConvertDS);
                result.save(new File(trainFilesSerialized + i + ".ser"));
            }
            for (int i = 1; i <= (TRAINFILENAMES.length); i++){
                normalizeCifar(new File(trainFilesSerialized + i + ".ser"));
            }
            inputStream = testInputStream;
            DataSet result = convertDataSet(numToConvertDS);
            result.save(new File(testFilesSerialized));
            normalizeCifar(new File(testFilesSerialized));
        }
        setInputStream();
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
        if (train) {
            f = new File(trainFilesSerialized + 1 + ".ser");
            if (!f.exists()) return false;
        } else {
            f = new File(testFilesSerialized);
            if (!f.exists()) return false;
        }
        return true;
    }

    /**
     * Preprocess and store cifar based on successful Torch approach by Sergey Zagoruyko
     * Reference: https://github.com/szagoruyko/cifar.torch
     */
    public opencv_core.Mat convertCifar(Mat orgImage) {
        numExamples++;
        Mat resImage = new Mat();
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
//        ImageTransform yuvTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2Luv);
//        ImageTransform histEqualization = new EqualizeHistTransform(new Random(seed), COLOR_BGR2Luv);
        ImageTransform yuvTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        ImageTransform histEqualization = new EqualizeHistTransform(new Random(seed), COLOR_BGR2YCrCb);

        if (converter != null) {
            ImageWritable writable = new ImageWritable(converter.convert(orgImage));
            // TODO rec to normalize y before transform - currently doing after
            writable = yuvTransform.transform(writable); // Converts to chrome color to help emphasize image objects
            writable = histEqualization.transform(writable); // Normalizes values to further clarify object of interest
            resImage = converter.convert(writable.getFrame());
        }

        return resImage;
    }

    /**
     * Normalize and store cifar based on successful Torch approach by Sergey Zagoruyko
     * Reference: https://github.com/szagoruyko/cifar.torch
     */
    public void normalizeCifar(File fileName) {
        DataSet result =  new DataSet();
        result.load(fileName);
        if(!meanStdStored && train) { // TODO check for file with stats to load
            uMean = Math.abs(uMean/numExamples);
            uStd = Math.sqrt(uStd);
            vMean = Math.abs(vMean/numExamples);
            vStd = Math.sqrt(vStd);
            // TODO find cleaner way to store and load (e.g. json or yaml)
            try {
                FileUtils.write(meanVarPath, uMean + "," + uStd + "," + vMean + "," + vStd);
            } catch (IOException e) {
                e.printStackTrace();
            }
            meanStdStored = true;
        } else if (uMean == 0 && meanStdStored) {
            try {
                String[] values = FileUtils.readFileToString(meanVarPath).split(",");
                uMean = Double.parseDouble(values[0]);
                uStd = Double.parseDouble(values[1]);
                vMean = Double.parseDouble(values[2]);
                vStd = Double.parseDouble(values[3]);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        for (int i = 0; i < result.numExamples(); i++) {
            INDArray newFeatures = result.get(i).getFeatureMatrix();
            newFeatures.tensorAlongDimension(0, new int[] {0,2,3}).divi(255);
            newFeatures.tensorAlongDimension(1, new int[] {0,2,3}).subi(uMean).divi(uStd);
            newFeatures.tensorAlongDimension(2, new int[] {0,2,3}).subi(vMean).divi(vStd);
            result.get(i).setFeatures(newFeatures);
        }
        result.save(fileName);
    }

    public Pair<INDArray, opencv_core.Mat> convertMat(byte[] byteFeature) {
        INDArray label= FeatureUtil.toOutcomeVector(byteFeature[0], NUM_LABELS); ; // first value in the 3073 byte array
        opencv_core.Mat image = new opencv_core.Mat(HEIGHT, WIDTH, CV_8UC(CHANNELS)); // feature are 3072
        ByteBuffer imageData = image.createBuffer();

        for (int i = 0; i < HEIGHT * WIDTH; i++) {
            imageData.put(3 * i, byteFeature[i + 1 + 2 * height * width]); // blue
            imageData.put(3 * i + 1, byteFeature[i + 1 + height * width]); // green
            imageData.put(3 * i + 2, byteFeature[i + 1]); // red
        }
        if (preProcessCifar) {
            image = convertCifar(image);
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

        DataSet result = new DataSet();
        try {
            result = result.merge(dataSets);
        } catch (IllegalArgumentException e) {
            return result;
        }
        double uTempMean, vTempMean;
        for (DataSet data :result) {
            try {
                if (preProcessCifar) {
                    INDArray uChannel = data.getFeatures().tensorAlongDimension(1, new int[] {0,2,3});
                    INDArray vChannel = data.getFeatures().tensorAlongDimension(2, new int[] {0,2,3});
                    uTempMean = uChannel.mean(new int[] {0,2,3}).getDouble(0);
                    // TODO INDArray.var result is incorrect based on dimensions passed in thus manual - log issue and test example
                    uStd += varManual(uChannel, uTempMean);
                    uMean += uTempMean;
                    vTempMean = vChannel.mean(new int[] {0,2,3}).getDouble(0);
                    vStd += varManual(vChannel, vTempMean);
                    vMean += vTempMean;
                }
            } catch (IllegalArgumentException e) {
                throw new IllegalStateException("The number of channels must be 3 to special preProcess Cifar with.");
            }
        }
        if(shuffle && num > 1) result.shuffle(seed); // TODO confirm shuffle not same on mult epochs with set seed...
        return result;
    }

    public double varManual(INDArray x, double mean) {
        INDArray xSubMean = x.sub(mean);
        INDArray squared = xSubMean.muli(xSubMean);
        double accum = Nd4j.getExecutioner().execAndReturn(new Sum(squared)).getFinalResult().doubleValue();
        return accum/x.ravel().length();
    }

    public DataSet next(int batchSize) {
        return next(batchSize, 1, 0);
    }

    public DataSet next(int batchSize, int fileNum, int batchNum) {
        List<DataSet> temp = new ArrayList<>();
        DataSet result;
        if (cifarProcessedFilesExists() && preProcessCifar) {
            if (batchNum == 0) {
                if(train) loadDS.load(new File(trainFilesSerialized + fileNum + ".ser"));
                else loadDS.load(new File(testFilesSerialized));
                // Shuffle all examples in file before batching happens also for each reset
                if(shuffle && batchSize > 1) loadDS.shuffle(seed);
//                inputBatched = loadDS.batchBy(batchSize);
                idx = 0;
            }
//            result = inputBatched.get(batchNum);
            // TODO find better way - loading full dataset using gpu throwing errors
            for (int i = 0; i < batchSize; i++) {
                temp.add(loadDS.get(idx));
                idx ++;
            }
            if (temp.size() > 1 ) result = DataSet.merge(temp);
            else result = temp.get(0);

        } else {
            result = convertDataSet(batchSize);
        }
        return result;
    }

    public InputStream getInputStream() {
        return inputStream;
    }

    public void setInputStream() {
        if (train) inputStream = trainInputStream;
        else inputStream = testInputStream;
    }

    public List<String> getLabels(){
        return labels;
    }

    public void reset(){
        numExamples = 0;
        load();
    }

    public void train() {
        train = true;
        setInputStream();
    }

    public void test() {
        train = false;
        setInputStream();
        shuffle = false;
    }

}
