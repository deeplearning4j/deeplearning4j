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

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.io.*;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 *
 */
public class LoaderTests {

    @Test
    public void testLfwReader() throws Exception {
        String subDir = "lfw-a/lfw";
        File path = new File(FilenameUtils.concat(System.getProperty("user.home"), subDir));
        FileSplit fileSplit = new FileSplit(path, LFWLoader.ALLOWED_FORMATS, new Random(42));
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(42), LFWLoader.LABEL_PATTERN, 1, 1, 1);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 1);
        RecordReader rr = new ImageRecordReader(250, 250, 3, LFWLoader.LABEL_PATTERN);
        rr.initialize(inputSplit[0]);
        List<String> exptedLabel = rr.getLabels();

        RecordReader rr2 = new LFWLoader(new int[]{250, 250, 3}, true).getRecordReader(1, 1, 1, new Random(42));

        assertEquals(exptedLabel.get(0), rr2.getLabels().get(0));
    }

    @Test
    public void testCifarLoader() {
        File dir = new File(FilenameUtils.concat(System.getProperty("user.home"), "cifar/cifar-10-batches-bin"));
        CifarLoader cifar = new CifarLoader(false, dir);
        assertTrue(dir.exists());
        assertTrue(cifar.getLabels() != null);
    }

    @Test
    public void testCifarInputStream() throws Exception {
        // check train
        String subDir = "cifar/cifar-10-batches-bin/data_batch_1.bin";
        String path = FilenameUtils.concat(System.getProperty("user.home"), subDir);
        byte[] fullDataExpected = new byte[3073];
        FileInputStream inExpected = new FileInputStream(new File(path));
        inExpected.read(fullDataExpected);

        byte[] fullDataActual = new byte[3073];
        CifarLoader cifarLoad = new CifarLoader(true);
        InputStream inActual = cifarLoad.getInputStream();
        inActual.read(fullDataActual);
        assertEquals(fullDataExpected[0], fullDataActual[0]);

        // check test
        subDir = "cifar/cifar-10-batches-bin/test_batch.bin";
        path = FilenameUtils.concat(System.getProperty("user.home"), subDir);
        fullDataExpected = new byte[3073];
        inExpected = new FileInputStream(new File(path));
        inExpected.read(fullDataExpected);

        fullDataActual = new byte[3073];
        cifarLoad = new CifarLoader(false);
        inActual = cifarLoad.getInputStream();
        inActual.read(fullDataActual);
        assertEquals(fullDataExpected[0], fullDataActual[0]);

    }

    @Test
    public void testCifarLoaderNext() throws Exception {
        boolean train = true;
        boolean preProcessCifar = false;
        int numExamples = 10;
        int row = 28;
        int col = 28;
        int channels = 1;
        CifarLoader loader = new CifarLoader(row, col, channels, train, preProcessCifar);
        DataSet data = loader.next(numExamples);
        assertEquals(numExamples, data.getLabels().size(0));
        assertEquals(channels, data.getFeatureMatrix().size(1));

        train = true;
        preProcessCifar = true;
        row = 28;
        col = 28;
        channels = 3;
        loader = new CifarLoader(row, col, channels, train, preProcessCifar);
        data = loader.next(1);
        assertEquals(1, data.getFeatureMatrix().size(0));
        assertEquals(channels * row * col, data.getFeatureMatrix().ravel().length());

        train = false;
        preProcessCifar = false;
        loader = new CifarLoader(row, col, channels, train, preProcessCifar);
        data = loader.next(numExamples);
        assertEquals(row, data.getFeatureMatrix().size(2));

        train = false;
        preProcessCifar = true;
        loader = new CifarLoader(row, col, channels, train, preProcessCifar);
        data = loader.next(numExamples);
        assertEquals(numExamples, data.getLabels().size(0));
        assertEquals(col, data.getFeatureMatrix().size(3));

    }

    @Test
    public void testCifarLoaderReset() throws Exception {
        int numExamples = 50;
        int row = 28;
        int col = 28;
        int channels = 3;
        CifarLoader loader = new CifarLoader(row,col,channels, null, false, false, false);
        DataSet data;
        for (int i =0; i < loader.NUM_TEST_IMAGES/numExamples; i++) {
            loader.next(numExamples);
        }
        data = loader.next(numExamples);
        assertEquals(new DataSet(), data);
        loader.reset();
        data = loader.next(numExamples);
        assertEquals(numExamples, data.getLabels().size(0));
    }


    //@Ignore // Use when confirming data is getting stored
    @Test
    public void testProcessCifar() {
        int row = 32;
        int col = 32;
        int channels = 3;
        File dir1 = new File(CifarLoader.trainFilesSerialized + "1.ser");
//        File dir5 = new File(CifarLoader.trainFilesSerialized + "5.ser");
        CifarLoader cifar = new CifarLoader(row,col,channels, null, true, true, false);
        assertTrue(dir1.exists());
        DataSet result = cifar.next(1);
        assertEquals(result.getFeatureMatrix().length(), 32*32*3, 0.0);

    }

}
