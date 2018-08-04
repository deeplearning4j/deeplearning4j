/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.image.loader;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.nd4j.linalg.primitives.Pair;
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
 * CifarLoader is loader specific for the Cifar10 dataset
 *
 * Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
 *
 * There is a special preProcessor used to normalize the dataset based on Sergey Zagoruyko example
 * https://github.com/szagoruyko/cifar.torch
 */
public class CifarLoader extends NativeImageLoader implements Serializable {
    public static final int NUM_TRAIN_IMAGES = 50000;
    public static final int NUM_TEST_IMAGES = 10000;
    public static final int NUM_LABELS = 10; // Note 6000 imgs per class
    public static final int HEIGHT = 32;
    public static final int WIDTH = 32;
    public static final int CHANNELS = 3;
    public static final boolean DEFAULT_USE_SPECIAL_PREPROC = false;
    public static final boolean DEFAULT_SHUFFLE = true;

    private static final int BYTEFILELEN = 3073;
    private static final String[] TRAINFILENAMES =
                    {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch5.bin"};
    private static final String TESTFILENAME = "test_batch.bin";
    private static final String dataBinUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    private static final String localDir = "cifar";
    private static final String dataBinFile = "cifar-10-batches-bin";
    private static final String labelFileName = "batches.meta.txt";
    private static final int numToConvertDS = 10000; // Each file is 10000 images, limiting for file preprocess load

    protected final File fullDir;
    protected final File meanVarPath;
    protected final String trainFilesSerialized;
    protected final String testFilesSerialized;

    protected InputStream inputStream;
    protected InputStream trainInputStream;
    protected InputStream testInputStream;
    protected List<String> labels = new ArrayList<>();
    public static Map<String, String> cifarDataMap = new HashMap<>();


    protected boolean train;
    protected boolean useSpecialPreProcessCifar;
    protected long seed;
    protected boolean shuffle = true;
    protected int numExamples = 0;
    protected double uMean = 0;
    protected double uStd = 0;
    protected double vMean = 0;
    protected double vStd = 0;
    protected boolean meanStdStored = false;
    protected int loadDSIndex = 0;
    protected DataSet loadDS = new DataSet();
    protected int fileNum = 0;

    private static File getDefaultDirectory() {
        return new File(BASE_DIR, FilenameUtils.concat(localDir, dataBinFile));
    }

    public CifarLoader() {
        this(true);
    }

    public CifarLoader(boolean train) {
        this(train, null);
    }

    public CifarLoader(boolean train, File fullPath) {
        this(HEIGHT, WIDTH, CHANNELS, null, train, DEFAULT_USE_SPECIAL_PREPROC, fullPath, System.currentTimeMillis(),
                        DEFAULT_SHUFFLE);
    }

    public CifarLoader(int height, int width, int channels, boolean train, boolean useSpecialPreProcessCifar) {
        this(height, width, channels, null, train, useSpecialPreProcessCifar);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train,
                    boolean useSpecialPreProcessCifar) {
        this(height, width, channels, imgTransform, train, useSpecialPreProcessCifar, DEFAULT_SHUFFLE);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train,
                    boolean useSpecialPreProcessCifar, boolean shuffle) {
        this(height, width, channels, imgTransform, train, useSpecialPreProcessCifar, null, System.currentTimeMillis(),
                        shuffle);
    }

    public CifarLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train,
                    boolean useSpecialPreProcessCifar, File fullDir, long seed, boolean shuffle) {
        super(height, width, channels, imgTransform);
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.train = train;
        this.useSpecialPreProcessCifar = useSpecialPreProcessCifar;
        this.seed = seed;
        this.shuffle = shuffle;

        if (fullDir == null) {
            this.fullDir = getDefaultDirectory();
        } else {
            this.fullDir = fullDir;
        }
        meanVarPath = new File(this.fullDir, "meanVarPath.txt");
        trainFilesSerialized = FilenameUtils.concat(this.fullDir.toString(), "cifar_train_serialized");
        testFilesSerialized = FilenameUtils.concat(this.fullDir.toString(), "cifar_test_serialized.ser");

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

    protected void generateMaps() {
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

    protected void load() {
        if (!cifarRawFilesExist() && !fullDir.exists()) {
            generateMaps();
            fullDir.mkdir();

            log.info("Downloading CIFAR data set");
            downloadAndUntar(cifarDataMap, new File(BASE_DIR, localDir));
        }
        try {
            Collection<File> subFiles = FileUtils.listFiles(fullDir, new String[] {"bin"}, true);
            Iterator<File> trainIter = subFiles.iterator();
            trainInputStream = new SequenceInputStream(new FileInputStream(trainIter.next()),
                            new FileInputStream(trainIter.next()));
            while (trainIter.hasNext()) {
                File nextFile = trainIter.next();
                if (!TESTFILENAME.equals(nextFile.getName()))
                    trainInputStream = new SequenceInputStream(trainInputStream, new FileInputStream(nextFile));
            }
            testInputStream = new FileInputStream(new File(fullDir, TESTFILENAME));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (labels.isEmpty())
            defineLabels();

        if (useSpecialPreProcessCifar && train && !cifarProcessedFilesExists()) {
            for (int i = fileNum + 1; i <= (TRAINFILENAMES.length); i++) {
                inputStream = trainInputStream;
                DataSet result = convertDataSet(numToConvertDS);
                result.save(new File(trainFilesSerialized + i + ".ser"));
            }
            //            for (int i = 1; i <= (TRAINFILENAMES.length); i++){
            //                normalizeCifar(new File(trainFilesSerialized + i + ".ser"));
            //            }
            inputStream = testInputStream;
            DataSet result = convertDataSet(numToConvertDS);
            result.save(new File(testFilesSerialized));
            //            normalizeCifar(new File(testFilesSerialized));
        }
        setInputStream();
    }

    private boolean cifarRawFilesExist() {
        File f = new File(fullDir, TESTFILENAME);
        if (!f.exists())
            return false;

        for (String name : TRAINFILENAMES) {
            f = new File(fullDir, name);
            if (!f.exists())
                return false;
        }
        return true;
    }

    private boolean cifarProcessedFilesExists() {
        File f;
        if (train) {
            f = new File(trainFilesSerialized + 1 + ".ser");
            if (!f.exists())
                return false;
        } else {
            f = new File(testFilesSerialized);
            if (!f.exists())
                return false;
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
            // TODO determine if need to normalize y before transform - opencv docs rec but currently doing after
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
        DataSet result = new DataSet();
        result.load(fileName);
        if (!meanStdStored && train) {
            uMean = Math.abs(uMean / numExamples);
            uStd = Math.sqrt(uStd);
            vMean = Math.abs(vMean / numExamples);
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
            INDArray newFeatures = result.get(i).getFeatures();
            newFeatures.tensorAlongDimension(0, new int[] {0, 2, 3}).divi(255);
            newFeatures.tensorAlongDimension(1, new int[] {0, 2, 3}).subi(uMean).divi(uStd);
            newFeatures.tensorAlongDimension(2, new int[] {0, 2, 3}).subi(vMean).divi(vStd);
            result.get(i).setFeatures(newFeatures);
        }
        result.save(fileName);
    }

    public Pair<INDArray, opencv_core.Mat> convertMat(byte[] byteFeature) {
        INDArray label = FeatureUtil.toOutcomeVector(byteFeature[0], NUM_LABELS);; // first value in the 3073 byte array
        opencv_core.Mat image = new opencv_core.Mat(HEIGHT, WIDTH, CV_8UC(CHANNELS)); // feature are 3072
        ByteBuffer imageData = image.createBuffer();

        for (int i = 0; i < HEIGHT * WIDTH; i++) {
            imageData.put(3 * i, byteFeature[i + 1 + 2 * HEIGHT * WIDTH]); // blue
            imageData.put(3 * i + 1, byteFeature[i + 1 + HEIGHT * WIDTH]); // green
            imageData.put(3 * i + 2, byteFeature[i + 1]); // red
        }
        //        if (useSpecialPreProcessCifar) {
        //            image = convertCifar(image);
        //        }

        return new Pair<>(label, image);
    }

    public DataSet convertDataSet(int num) {
        int batchNumCount = 0;
        List<DataSet> dataSets = new ArrayList<>();
        Pair<INDArray, opencv_core.Mat> matConversion;
        byte[] byteFeature = new byte[BYTEFILELEN];

        try {
//            while (inputStream.read(byteFeature) != -1 && batchNumCount != num) {
            while (batchNumCount != num && inputStream.read(byteFeature) != -1 ) {
                matConversion = convertMat(byteFeature);
                try {
                    dataSets.add(new DataSet(asMatrix(matConversion.getSecond()), matConversion.getFirst()));
                    batchNumCount++;
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if(dataSets.size() == 0){
            return new DataSet();
        }

        DataSet result = DataSet.merge(dataSets);

        double uTempMean, vTempMean;
        for (DataSet data : result) {
            try {
                if (useSpecialPreProcessCifar) {
                    INDArray uChannel = data.getFeatures().tensorAlongDimension(1, new int[] {0, 2, 3});
                    INDArray vChannel = data.getFeatures().tensorAlongDimension(2, new int[] {0, 2, 3});
                    uTempMean = uChannel.meanNumber().doubleValue();
                    // TODO INDArray.var result is incorrect based on dimensions passed in thus using manual
                    uStd += varManual(uChannel, uTempMean);
                    uMean += uTempMean;
                    vTempMean = vChannel.meanNumber().doubleValue();
                    vStd += varManual(vChannel, vTempMean);
                    vMean += vTempMean;
                    data.setFeatures(data.getFeatures().div(255));
                } else {
                    // normalize if just input stream and not special preprocess
                    data.setFeatures(data.getFeatures().div(255));
                }
            } catch (IllegalArgumentException e) {
                throw new IllegalStateException("The number of channels must be 3 to special preProcess Cifar with.");
            }
        }
        if (shuffle && num > 1)
            result.shuffle(seed);
        return result;
    }

    public double varManual(INDArray x, double mean) {
        INDArray xSubMean = x.sub(mean);
        INDArray squared = xSubMean.muli(xSubMean);
        double accum = Nd4j.getExecutioner().execAndReturn(new Sum(squared)).getFinalResult().doubleValue();
        return accum / x.ravel().length();
    }

    public DataSet next(int batchSize) {
        return next(batchSize, 0);
    }

    public DataSet next(int batchSize, int exampleNum) {
        List<DataSet> temp = new ArrayList<>();
        DataSet result;
        if (cifarProcessedFilesExists() && useSpecialPreProcessCifar) {
            if (exampleNum == 0 || ((exampleNum / fileNum) == numToConvertDS && train)) {
                fileNum++;
                if (train)
                    loadDS.load(new File(trainFilesSerialized + fileNum + ".ser"));
                loadDS.load(new File(testFilesSerialized));
                // Shuffle all examples in file before batching happens also for each reset
                if (shuffle && batchSize > 1)
                    loadDS.shuffle(seed);
                loadDSIndex = 0;
                //          inputBatched = loadDS.batchBy(batchSize);
            }
            // TODO loading full train dataset when using cuda causes memory error - find way to load into list off gpu
            //            result = inputBatched.get(batchNum);
            for (int i = 0; i < batchSize; i++) {
                if (loadDS.get(loadDSIndex) != null)
                    temp.add(loadDS.get(loadDSIndex));
                else
                    break;
                loadDSIndex++;
            }
            if (temp.size() > 1)
                result = DataSet.merge(temp);
            else
                result = temp.get(0);
        } else {
            result = convertDataSet(batchSize);
        }
        return result;
    }

    public InputStream getInputStream() {
        return inputStream;
    }

    public void setInputStream() {
        if (train)
            inputStream = trainInputStream;
        else
            inputStream = testInputStream;
    }

    public List<String> getLabels() {
        return labels;
    }

    public void reset() {
        numExamples = 0;
        fileNum = 0;
        load();
    }

}
