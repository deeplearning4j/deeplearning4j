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


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PatternPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.data.Image;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Loads LFW faces data transform.
 * Customize the size of images by passing in preferred dimensions.
 *
 * DataSet
 *      5749 different individuals
 *      1680 people have two or more images in the database
 *      4069 people have just a single image in the database
 *      available as 250 by 250 pixel JPEG images
 *      most images are in color, although a few are grayscale
 *
 */
public class LFWLoader extends BaseImageLoader implements Serializable {

    public final static int NUM_IMAGES = 13233;
    public final static int NUM_LABELS = 5749;
    public final static int SUB_NUM_IMAGES = 1054;
    public final static int SUB_NUM_LABELS = 432;
    public final static int HEIGHT = 250;
    public final static int WIDTH = 250;
    public final static int CHANNELS = 3;
    public final static String DATA_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz";
    public final static String LABEL_URL = "http://vis-www.cs.umass.edu/lfw/lfw-names.txt";
    public final static String SUBSET_URL = "http://vis-www.cs.umass.edu/lfw/lfw-a.tgz";
    protected final static String REGEX_PATTERN = ".[0-9]+";
    public final static PathLabelGenerator LABEL_PATTERN = new PatternPathLabelGenerator(REGEX_PATTERN);

    public String dataFile = "lfw";
    public String labelFile = "lfw-names.txt";
    public String subsetFile = "lfw-a";

    public String localDir = "lfw";
    public String localSubDir = "lfw-a/lfw";
    protected File fullDir;

    protected boolean useSubset = false;
    InputSplit[] inputSplit;

    public static Map<String, String> lfwData = new HashMap<>();
    public static Map<String, String> lfwLabel = new HashMap<>();
    public static Map<String, String> lfwSubsetData = new HashMap<>();

    public LFWLoader() {
        this(false);
    }

    public LFWLoader(boolean useSubset) {
        this(new long[] {HEIGHT, WIDTH, CHANNELS,}, null, useSubset);
    }

    public LFWLoader(int[] imgDim, boolean useSubset) {
        this(imgDim, null, useSubset);
    }

    public LFWLoader(long[] imgDim, boolean useSubset) {
        this(imgDim, null, useSubset);
    }

    public LFWLoader(int[] imgDim, ImageTransform imgTransform, boolean useSubset) {
        this.height = imgDim[0];
        this.width = imgDim[1];
        this.channels = imgDim[2];
        this.imageTransform = imgTransform;
        this.useSubset = useSubset;
        this.localDir = useSubset ? localSubDir : localDir;
        this.fullDir = new File(BASE_DIR, localDir);
        generateLfwMaps();
    }

    public LFWLoader(long[] imgDim, ImageTransform imgTransform, boolean useSubset) {
        this.height = imgDim[0];
        this.width = imgDim[1];
        this.channels = imgDim[2];
        this.imageTransform = imgTransform;
        this.useSubset = useSubset;
        this.localDir = useSubset ? localSubDir : localDir;
        this.fullDir = new File(BASE_DIR, localDir);
        generateLfwMaps();
    }

    public void generateLfwMaps() {
        if (useSubset) {
            // Subset of just faces with a name starting with A
            lfwSubsetData.put("filesFilename", new File(SUBSET_URL).getName());
            lfwSubsetData.put("filesURL", SUBSET_URL);
            lfwSubsetData.put("filesFilenameUnzipped", subsetFile);

        } else {
            lfwData.put("filesFilename", new File(DATA_URL).getName());
            lfwData.put("filesURL", DATA_URL);
            lfwData.put("filesFilenameUnzipped", dataFile);

            lfwLabel.put("filesFilename", labelFile);
            lfwLabel.put("filesURL", LABEL_URL);
            lfwLabel.put("filesFilenameUnzipped", labelFile);
        }

    }

    public void load() {
        load(NUM_IMAGES, NUM_IMAGES, NUM_LABELS, LABEL_PATTERN, 1, rng);
    }

    public void load(long batchSize, long numExamples, long numLabels, PathLabelGenerator labelGenerator,
                    double splitTrainTest, Random rng) {
        if (!imageFilesExist()) {
            if (!fullDir.exists() || fullDir.listFiles() == null || fullDir.listFiles().length == 0) {
                fullDir.mkdir();

                if (useSubset) {
                    log.info("Downloading {} subset...", localDir);
                    downloadAndUntar(lfwSubsetData, fullDir);
                } else {
                    log.info("Downloading {}...", localDir);
                    downloadAndUntar(lfwData, fullDir);
                    downloadAndUntar(lfwLabel, fullDir);
                }
            }
        }
        FileSplit fileSplit = new FileSplit(fullDir, ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, ALLOWED_FORMATS, labelGenerator, numExamples,
                        numLabels, 0, batchSize, null);
        inputSplit = fileSplit.sample(pathFilter, numExamples * splitTrainTest, numExamples * (1 - splitTrainTest));
    }

    public boolean imageFilesExist() {
        if (useSubset) {
            File f = new File(BASE_DIR, lfwSubsetData.get("filesFilenameUnzipped"));
            if (!f.exists())
                return false;
        } else {
            File f = new File(BASE_DIR, lfwData.get("filesFilenameUnzipped"));
            if (!f.exists())
                return false;
            f = new File(BASE_DIR, lfwLabel.get("filesFilenameUnzipped"));
            if (!f.exists())
                return false;
        }
        return true;
    }


    public RecordReader getRecordReader(long numExamples) {
        return getRecordReader(numExamples, numExamples, new long[] {height, width, channels},
                        useSubset ? SUB_NUM_LABELS : NUM_LABELS, LABEL_PATTERN, true, 1,
                        new Random(System.currentTimeMillis()));
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, long numLabels, Random rng) {
        return getRecordReader(numExamples, batchSize, new long[] {height, width, channels}, numLabels, LABEL_PATTERN,
                        true, 1, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, boolean train, double splitTrainTest) {
        return getRecordReader(numExamples, batchSize, new long[] {height, width, channels},
                        useSubset ? SUB_NUM_LABELS : NUM_LABELS, LABEL_PATTERN, train, splitTrainTest,
                        new Random(System.currentTimeMillis()));
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, int[] imgDim, boolean train,
                    double splitTrainTest, Random rng) {
        return getRecordReader(numExamples, batchSize, imgDim, useSubset ? SUB_NUM_LABELS : NUM_LABELS, LABEL_PATTERN,
                        train, splitTrainTest, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, long[] imgDim, boolean train,
                    double splitTrainTest, Random rng) {
        return getRecordReader(numExamples, batchSize, imgDim, useSubset ? SUB_NUM_LABELS : NUM_LABELS, LABEL_PATTERN,
                        train, splitTrainTest, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, PathLabelGenerator labelGenerator,
                    boolean train, double splitTrainTest, Random rng) {
        return getRecordReader(numExamples, batchSize, new long[] {height, width, channels},
                        useSubset ? SUB_NUM_LABELS : NUM_LABELS, labelGenerator, train, splitTrainTest, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, int[] imgDim, PathLabelGenerator labelGenerator,
                    boolean train, double splitTrainTest, Random rng) {
        return getRecordReader(numExamples, batchSize, imgDim, useSubset ? SUB_NUM_LABELS : NUM_LABELS, labelGenerator,
                        train, splitTrainTest, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, long[] imgDim, PathLabelGenerator labelGenerator,
                    boolean train, double splitTrainTest, Random rng) {
        return getRecordReader(numExamples, batchSize, imgDim, useSubset ? SUB_NUM_LABELS : NUM_LABELS, labelGenerator,
                        train, splitTrainTest, rng);
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, int[] imgDim, long numLabels,
                    PathLabelGenerator labelGenerator, boolean train, double splitTrainTest, Random rng) {
        load(batchSize, numExamples, numLabels, labelGenerator, splitTrainTest, rng);
        RecordReader recordReader =
                        new ImageRecordReader(imgDim[0], imgDim[1], imgDim[2], labelGenerator, imageTransform);

        try {
            InputSplit data = train ? inputSplit[0] : inputSplit[1];
            recordReader.initialize(data);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
    }

    public RecordReader getRecordReader(long batchSize, long numExamples, long[] imgDim, long numLabels,
                    PathLabelGenerator labelGenerator, boolean train, double splitTrainTest, Random rng) {
        load(batchSize, numExamples, numLabels, labelGenerator, splitTrainTest, rng);
        RecordReader recordReader =
                        new ImageRecordReader(imgDim[0], imgDim[1], imgDim[2], labelGenerator, imageTransform);

        try {
            InputSplit data = train ? inputSplit[0] : inputSplit[1];
            recordReader.initialize(data);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
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

    @Override
    public Image asImageMatrix(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Image asImageMatrix(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

}
