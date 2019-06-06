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

package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.tools.VariableMultiTimeseriesGenerator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.MultiDataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AsyncMultiDataSetIteratorTest extends BaseDL4JTest {

    /**
     * THIS TEST SHOULD BE ALWAYS RUN WITH DOUBLE PRECISION, WITHOUT ANY EXCLUSIONS
     *
     * @throws Exception
     */
    @Test
    public void testVariableTimeSeries1() throws Exception {
        val iterator = new VariableMultiTimeseriesGenerator(1192, 1000, 32, 128, 10, 500, 10);
        iterator.reset();
        iterator.hasNext();
        val amdsi = new AsyncMultiDataSetIterator(iterator, 2, true);

        for (int e = 0; e < 10; e++) {
            int cnt = 0;
            while (amdsi.hasNext()) {
                MultiDataSet mds = amdsi.next();


                //log.info("Features ptr: {}", AtomicAllocator.getInstance().getPointer(mds.getFeatures()[0].data()).address());
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt,
                                mds.getFeatures()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.25,
                                mds.getLabels()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.5,
                                mds.getFeaturesMaskArrays()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.75,
                                mds.getLabelsMaskArrays()[0].meanNumber().doubleValue(), 1e-10);

                cnt++;
            }

            amdsi.reset();
            log.info("Epoch {} finished...", e);
        }
    }


    @Test
    public void testVariableTimeSeries2() throws Exception {
        val iterator = new VariableMultiTimeseriesGenerator(1192, 1000, 32, 128, 10, 500, 10);

        for (int e = 0; e < 10; e++) {
            iterator.reset();
            iterator.hasNext();
            val amdsi = new AsyncMultiDataSetIterator(iterator, 2, true);

            int cnt = 0;
            while (amdsi.hasNext()) {
                MultiDataSet mds = amdsi.next();


                //log.info("Features ptr: {}", AtomicAllocator.getInstance().getPointer(mds.getFeatures()[0].data()).address());
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt,
                        mds.getFeatures()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.25,
                        mds.getLabels()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.5,
                        mds.getFeaturesMaskArrays()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.75,
                        mds.getLabelsMaskArrays()[0].meanNumber().doubleValue(), 1e-10);

                cnt++;
            }
        }
    }
/*
    @Test
    public void testResetBug() throws Exception {
        // /home/raver119/develop/dl4j-examples/src/main/resources/uci/train/features

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit("/home/raver119/develop/dl4j-examples/src/main/resources/uci/train/features" + "/%d.csv", 0, 449));
        RecordReader trainLabels = new CSVRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit("/home/raver119/develop/dl4j-examples/src/main/resources/uci/train/labels" + "/%d.csv", 0, 449));

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
                .addSequenceReader("features", trainFeatures)
                .addReader("labels", trainLabels)
                .addInput("features")
                .addOutputOneHot("labels", 0, numLabelClasses)
                .build();

        //Normalize the training data
        MultiDataNormalization normalizer = new MultiNormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();


        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit("/home/raver119/develop/dl4j-examples/src/main/resources/uci/test/features" + "/%d.csv", 0, 149));
        RecordReader testLabels = new CSVRecordReader();
        testLabels.initialize(new NumberedFileInputSplit("/home/raver119/develop/dl4j-examples/src/main/resources/uci/test/labels" + "/%d.csv", 0, 149));

        MultiDataSetIterator testData = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
                .addSequenceReader("features", testFeatures)
                .addReader("labels", testLabels)
                .addInput("features")
                .addOutputOneHot("labels", 0, numLabelClasses)
                .build();

        System.out.println("-------------- HASH 1----------------");
        testData.reset();
        while(testData.hasNext()){
            System.out.println(Arrays.hashCode(testData.next().getFeatures(0).data().asFloat()));
        }

        System.out.println("-------------- HASH 2 ----------------");
        testData.reset();
        testData.hasNext();     //***** Remove this (or move to after async creation), and we get expected results *****
        val adsi = new AsyncMultiDataSetIterator(testData, 4, true);    //OR remove this (keeping hasNext) and we get expected results
        //val adsi = new AsyncShieldMultiDataSetIterator(testData);
        while(adsi.hasNext()){
            System.out.println(Arrays.hashCode(adsi.next().getFeatures(0).data().asFloat()));
        }
    }
    */
}
