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
import org.datavec.image.transform.HistEqualizationTransform;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

import static org.bytedeco.javacpp.opencv_core.CV_8UC;

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
    public static File fullDir = new File(BASE_DIR,  FilenameUtils.concat(localDir, dataBinFile));

//    public String dataUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"; // used for python version - similar structure to datBin structure
//    public String dataFile = "cifar-10-python";
    protected static String labelFileName = "batches.meta.txt";
    protected static InputStream inputStream;
    protected static List<String> labels = new ArrayList<>();

    protected static String[] trainFileNames = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch5.bin"};
    protected static String testFileName = "test_batch.bin";
    protected static String  trainFilesSerialized1 = fullDir + "cifar_train_serialized1.ser";
    protected static String  trainFilesSerialized2 = fullDir + "cifar_train_serialized2.ser";
    protected static String  testFilesSerialized = fullDir + "cifar_test_serialized.ser";

    protected static boolean train = true;
    public static boolean preProcessCifar = true;
    public static Map<String, String> cifarTrainData = new HashMap<>();

    protected static int height = HEIGHT;
    protected static int width = WIDTH;
    protected static int channels = CHANNELS;
    protected static  int exampleCount = 0;

    // Using this in spark to reference where to load data from
//    static {
//        load();
//    }

    // TODO pass in height, width and channels
    public CifarLoader(){
        load();
    }

    public CifarLoader(boolean train){
        this.train = train;
    }

    public CifarLoader(int height, int width, int channels){
        super(height, width, channels);
        load();
    }

    public CifarLoader(int height, int width, int channels, boolean train, boolean preProcessCifar){
        this(height, width, channels, null, train, preProcessCifar);
        load();
    }
    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean preProcessCifar){
        super(height, width, channels, imgTransform);
        this.train = train;
        this.preProcessCifar = preProcessCifar;
        load();
    }

    public CifarLoader(boolean train, String localDir){
        this.localDir = localDir;
        this.fullDir = new File(localDir);
        this.train = train;
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
        cifarTrainData.put("filesFilename", new File(dataBinUrl).getName());
        cifarTrainData.put("filesURL", dataBinUrl);
        cifarTrainData.put("filesFilenameUnzipped", dataBinFile);
    }

    private void defineLabels() {
        try {
            File path = new File(fullDir, FilenameUtils.concat(dataBinFile, labelFileName));
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

    public void load()  {
        if (!cifarRawFilesExist() && !fullDir.exists()) {
            generateMaps();
            fullDir.mkdir();

            log.info("Downloading {}...", localDir);
            downloadAndUntar(cifarTrainData, fullDir);
        }
        try {
            // Create inputStream
            if(train) {
                Collection<File> subFiles = FileUtils.listFiles(new File(fullDir, dataBinFile), new String[] {"bin"}, true);
                Iterator trainIter = subFiles.iterator();
                inputStream = new SequenceInputStream(new FileInputStream((File) trainIter.next()), new FileInputStream((File) trainIter.next()));
                while (trainIter.hasNext()) {
                    File nextFile = (File) trainIter.next();
                    if(!testFileName.equals(nextFile.getName()))
                        inputStream = new SequenceInputStream(inputStream, new FileInputStream(nextFile));
                }
            }
            else
                inputStream = new FileInputStream(new File(fullDir, FilenameUtils.concat(dataBinFile, testFileName)));
        } catch(Exception e) {
            e.printStackTrace();
        }
        defineLabels();

        // TODO if preprocessor file use it otherwise preprocess
        if (!cifarProcessedFilesExists() && preProcessCifar) {
            DataSet result = convertDataSet(10000);
            result.save(trainFilesSerialized1);
            // TODO define save file name and save preprocessor
        }
    }

    public boolean cifarRawFilesExist(){
        File f = new File(fullDir, FilenameUtils.concat(dataBinFile, testFileName));
        if (!f.exists()) return false;

        for(String name: trainFileNames) {
            f = new File(fullDir, FilenameUtils.concat(dataBinFile, name));
            if (!f.exists()) return false;
        }
        return true;
    }

    private boolean cifarProcessedFilesExists(){
        File f = new File(trainFilesSerialized1);
        if(!f.exists()) return false;
        f = new File(trainFilesSerialized2);
        if(!f.exists()) return false;
        f = new File(testFilesSerialized);
        if(!f.exists()) return false;
        return true;
    }

    /**
     * Preprocess and store cifar data if it does not exist
     */
    public opencv_core.Mat preProcessCifar(opencv_core.Mat image)  {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        ImageTransform yuvTransform = new ColorConversionTransform();
        ImageTransform histEqualization = new HistEqualizationTransform();

        if (imageTransform != null && converter != null) {
            ImageWritable writable = new ImageWritable(converter.convert(image));
            writable = yuvTransform.transform(writable);
            // TODO confirm this works and does sparse contrast ...
            writable = histEqualization.transform(writable);
            // TODO normalize u and v?
            image = converter.convert(writable.getFrame());
        }
        image = scalingIfNeed(image);

        // TODO optional show images while preprocessing to see how the look

        return image;
    }

    public Pair<INDArray, opencv_core.Mat> convertMat(byte[] byteFeature) {
        INDArray label; // first value in the 3073 byte array
        opencv_core.Mat image = new opencv_core.Mat(height, width, CV_8UC(channels)); // feature are 3072
        ByteBuffer imageData = image.createBuffer();

        label = FeatureUtil.toOutcomeVector(byteFeature[0], NUM_LABELS);
        for (int i = 0; i < height * width; i++) {
            imageData.put(3 * i,     byteFeature[i + 1 + 2 * height * width]); // blue
            imageData.put(3 * i + 1, byteFeature[i + 1 +     height * width]); // green
            imageData.put(3 * i + 2, byteFeature[i + 1                     ]); // red
        }
        if (preProcessCifar) preProcessCifar(image);
        return new Pair<>(label, image);
    }

    public DataSet convertDataSet(int num) {
        int batchNumCount = 0;
        List<DataSet> dataSets = new ArrayList<>();
        Pair<INDArray, opencv_core.Mat> matConversion;
        byte[] byteFeature = new byte[BYTEFILELEN];

        try {
            while((inputStream.read(byteFeature)) != -1 && batchNumCount != num) {
                matConversion = convertMat(byteFeature);
                try {
                    // TODO don't ravel - just bring back 4D
                    dataSets.add(new DataSet(asRowVector(matConversion.getSecond()), matConversion.getFirst()));
                    batchNumCount++;
                } catch(Exception e){
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }
        // TODO shuffle here?
        DataSet result = new DataSet(Nd4j.vstack(inputs), Nd4j.vstack(labels));
        return result;

    }

    public InputStream getInputStream() {
        return inputStream;
    }

    public List<String> getLabels(){
        return labels;
    }


}
