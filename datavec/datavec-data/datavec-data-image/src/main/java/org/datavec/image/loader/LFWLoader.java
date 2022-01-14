/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.loader;


import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PatternPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.data.Image;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.eclipse.deeplearning4j.resources.DataSetResource;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@Slf4j
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


    private static DataSetResource lfwFull = ResourceDataSets.lfwFullData();
    private static DataSetResource lfwSub = ResourceDataSets.lfwSubData();
    private static DataSetResource lfwLabels = ResourceDataSets.lfwFullData();


    protected boolean useSubset = false;
    protected InputSplit[] inputSplit;



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
    }

    public LFWLoader(long[] imgDim, ImageTransform imgTransform, boolean useSubset) {
        this.height = imgDim[0];
        this.width = imgDim[1];
        this.channels = imgDim[2];
        this.imageTransform = imgTransform;
        this.useSubset = useSubset;

    }


    public void load() {
        load(NUM_IMAGES, NUM_IMAGES, NUM_LABELS, LABEL_PATTERN, 1, rng);
    }

    public void load(long batchSize, long numExamples, long numLabels, PathLabelGenerator labelGenerator,
                     double splitTrainTest, Random rng) {
        if (!imageFilesExist()) {
            if (useSubset) {
                lfwSub.download(true,3,20000,20000);
                lfwLabels.download(true,3,30000,3000);
            } else {
                lfwFull.download(true,3,20000,20000);
                lfwLabels.download(true,3,30000,3000);

            }

        }

        File inputDir = useSubset ? lfwSub.localCacheDirectory() : lfwFull.localCacheDirectory();

        FileSplit fileSplit = new FileSplit(inputDir, ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, ALLOWED_FORMATS, labelGenerator, numExamples,
                numLabels, 0, batchSize, null);
        inputSplit = fileSplit.sample(pathFilter, numExamples * splitTrainTest, numExamples * (1 - splitTrainTest));
    }

    public boolean imageFilesExist() {
        if (useSubset) {
            if (!lfwSub.existsLocally())
                return lfwSub.existsLocally();
        } else {
            return lfwFull.existsLocally();
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
            log.error("",e);
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
            log.error("",e);
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
    public INDArray asMatrix(File f, boolean nchw) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(InputStream inputStream, boolean nchw) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Image asImageMatrix(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Image asImageMatrix(File f, boolean nchw) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Image asImageMatrix(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Image asImageMatrix(InputStream inputStream, boolean nchw) throws IOException {
        throw new UnsupportedOperationException();
    }

}
