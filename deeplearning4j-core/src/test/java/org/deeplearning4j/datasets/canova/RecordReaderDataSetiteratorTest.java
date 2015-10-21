/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.datasets.canova;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 3/6/15.
 */
public class RecordReaderDataSetiteratorTest {

    @Test
    public void testRecordReader() throws Exception {
        RecordReader recordReader = new CSVRecordReader();
        FileSplit csv = new FileSplit(new ClassPathResource("csv-example.csv").getFile());
        recordReader.initialize(csv);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 34);
        DataSet next = iter.next();
        assertEquals(34, next.numExamples());
    }

    @Test
    public void testSequenceRecordReader() throws Exception {
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabels_0.txt");
        String labelsPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false);

        List<DataSet> dsList = new ArrayList<>();
        while (iter.hasNext()) {
            dsList.add(iter.next());
        }

        assertEquals(3, dsList.size());  //3 files
        for (int i = 0; i < 3; i++) {
            DataSet ds = dsList.get(i);
            INDArray features = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();
            assertEquals(1, features.size(0));   //1 example in mini-batch
            assertEquals(1, labels.size(0));
            assertEquals(3, features.size(1));   //3 values per line/time step
            assertEquals(4, labels.size(1));     //1 value per line, but 4 possible values -> one-hot vector
            assertEquals(4, features.size(2));   //sequence length = 4
            assertEquals(4, labels.size(2));
        }

        //Check features vs. expected:
        INDArray expF0 = Nd4j.create(1, 3, 4);
        expF0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 2}));
        expF0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{10, 11, 12}));
        expF0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{20, 21, 22}));
        expF0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{30, 31, 32}));
        assertEquals(dsList.get(0).getFeatureMatrix(), expF0);

        INDArray expF1 = Nd4j.create(1, 3, 4);
        expF1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{100, 101, 102}));
        expF1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{110, 111, 112}));
        expF1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{120, 121, 122}));
        expF1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{130, 131, 132}));
        assertEquals(dsList.get(1).getFeatureMatrix(), expF1);

        INDArray expF2 = Nd4j.create(1, 3, 4);
        expF2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{200, 201, 202}));
        expF2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{210, 211, 212}));
        expF2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{220, 221, 222}));
        expF2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{230, 231, 232}));
        assertEquals(dsList.get(2).getFeatureMatrix(), expF2);

        //Check labels vs. expected:
        INDArray expL0 = Nd4j.create(1, 4, 4);
        expL0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1, 0, 0, 0}));
        expL0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        expL0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 0, 1, 0}));
        expL0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 0, 0, 1}));
        assertEquals(dsList.get(0).getLabels(), expL0);

        INDArray expL1 = Nd4j.create(1, 4, 4);
        expL1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 0, 0, 1}));
        expL1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 0, 1, 0}));
        expL1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        expL1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{1, 0, 0, 0}));
        assertEquals(dsList.get(1).getLabels(), expL1);

        INDArray expL2 = Nd4j.create(1, 4, 4);
        expL2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        expL2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{1, 0, 0, 0}));
        expL2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 0, 0, 1}));
        expL2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 0, 1, 0}));
        assertEquals(dsList.get(2).getLabels(), expL2);

    }
}
