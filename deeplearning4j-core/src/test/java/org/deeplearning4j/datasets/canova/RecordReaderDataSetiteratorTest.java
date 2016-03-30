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

import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

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
    public void testRecordReaderMaxBatchLimit() throws Exception {
        RecordReader recordReader = new CSVRecordReader();
        FileSplit csv = new FileSplit(new ClassPathResource("csv-example.csv").getFile());
        recordReader.initialize(csv);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 10, -1, -1, 2);
        iter.next();
        iter.next();
        assertEquals(false, iter.hasNext());
    }

    @Test
    public void testRecordReaderMultiRegression() throws Exception {

        RecordReader csv = new CSVRecordReader();
        csv.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        int batchSize = 3;
        int labelIdxFrom = 3;
        int labelIdxTo = 4;

        DataSetIterator iter = new RecordReaderDataSetIterator(csv,batchSize,labelIdxFrom,labelIdxTo,true);
        DataSet ds = iter.next();

        INDArray f = ds.getFeatureMatrix();
        INDArray l = ds.getLabels();
        assertArrayEquals(new int[]{3,3}, f.shape());
        assertArrayEquals(new int[]{3,2}, l.shape());

        //Check values:
        double[][] fExpD = new double[][]{
                {5.1,3.5,1.4},
                {4.9,3.0,1.4},
                {4.7,3.2,1.3}};

        double[][] lExpD = new double[][]{
                {0.2,0},
                {0.2,0},
                {0.2,0}};

        INDArray fExp = Nd4j.create(fExpD);
        INDArray lExp = Nd4j.create(lExpD);

        assertEquals(fExp, f);
        assertEquals(lExp, l);

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

        assertEquals(3, iter.inputColumns());
        assertEquals(4, iter.totalOutcomes());

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

    @Test
    public void testSequenceRecordReaderRegression() throws Exception{
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequence_0.txt");
        String labelsPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 0, true);

        assertEquals(3, iter.inputColumns());
        assertEquals(3, iter.totalOutcomes());

        List<DataSet> dsList = new ArrayList<>();
        while (iter.hasNext()) {
            dsList.add(iter.next());
        }

        assertEquals(3, dsList.size());  //3 files
        for (int i = 0; i < 3; i++) {
            DataSet ds = dsList.get(i);
            INDArray features = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();
            assertArrayEquals(new int[]{1,3,4},features.shape());   //1 examples, 3 values, 4 time steps
            assertArrayEquals(new int[]{1,3,4},labels.shape());

            assertEquals(features,labels);
        }

        //Also test regression + reset from a single reader:
        featureReader.reset();
        iter = new SequenceRecordReaderDataSetIterator(featureReader, 1, 0, 2, true);
        int count = 0;
        while(iter.hasNext()){
            DataSet ds = iter.next();
            assertEquals(2,ds.getFeatureMatrix().size(1));
            assertEquals(1,ds.getLabels().size(1));
            count++;
        }
        assertEquals(3,count);


        iter.reset();
        count = 0;
        while(iter.hasNext()){
            iter.next();
            count++;
        }
        assertEquals(3,count);
    }

    @Test
    public void testSequenceRecordReaderReset() throws Exception {
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

        assertEquals(3, iter.inputColumns());
        assertEquals(4, iter.totalOutcomes());

        int nResets = 5;
        for( int i=0; i<nResets; i++ ) {
            iter.reset();
            int count = 0;
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                INDArray features = ds.getFeatureMatrix();
                INDArray labels = ds.getLabels();
                assertArrayEquals(new int[]{1,3,4},features.shape());
                assertArrayEquals(new int[]{1,4,4},labels.shape());
                count++;
            }
            assertEquals(3,count);
        }
    }



    @Test
    public void testCSVLoadingRegression() throws Exception {
        int nLines = 30;
        int nFeatures = 5;
        int miniBatchSize = 10;
        int labelIdx = 0;

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"rr_csv_test_rand.csv");
        double[][] data = makeRandomCSV(path,nLines,nFeatures);
        RecordReader testReader = new CSVRecordReader();
        testReader.initialize(new FileSplit(new File(path)));

        DataSetIterator iter = new RecordReaderDataSetIterator(testReader,null,miniBatchSize,labelIdx,1,true);
        int miniBatch = 0;
        while(iter.hasNext()){
            DataSet test = iter.next();
            INDArray features = test.getFeatureMatrix();
            INDArray labels = test.getLabels();
            assertArrayEquals(new int[]{miniBatchSize,nFeatures},features.shape());
            assertArrayEquals(new int[]{miniBatchSize, 1}, labels.shape());

            int startRow = miniBatch * miniBatchSize;
            for( int i=0; i<miniBatchSize; i++ ){
                double labelExp = data[startRow+i][labelIdx];
                double labelAct = labels.getDouble(i);
                assertEquals(labelExp,labelAct,1e-5f);

                int featureCount = 0;
                for( int j=0; j<nFeatures+1; j++ ){
                    if(j == labelIdx) continue;
                    double featureExp = data[startRow+i][j];
                    double featureAct = features.getDouble(i,featureCount++);
                    assertEquals(featureExp,featureAct,1e-5f);
                }
            }

            miniBatch++;
        }
        assertEquals(nLines/miniBatchSize,miniBatch);
    }


    public static double[][] makeRandomCSV(String tempFile, int nLines, int nFeatures) {
        File temp = new File(tempFile);
        temp.deleteOnExit();
        Random rand = new Random(12345);

        double[][] dArr = new double[nLines][nFeatures+1];

        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(temp)))) {
            for (int i = 0; i < nLines; i++) {
                dArr[i][0] = rand.nextDouble(); //First column: label
                out.print(dArr[i][0]);
                for (int j = 0; j < nFeatures; j++) {
                    dArr[i][j+1] = rand.nextDouble();
                    out.print("," + dArr[i][j+1]);
                }
                out.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dArr;
    }

    @Test
    public void testVariableLengthSequence() throws Exception{
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabelsShort_0.txt");
        String labelsPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iterAlignStart =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false,
                        SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);

        SequenceRecordReaderDataSetIterator iterAlignEnd =
                new SequenceRecordReaderDataSetIterator(featureReader2, labelReader2, 1, 4, false,
                        SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        assertEquals(3, iterAlignStart.inputColumns());
        assertEquals(4, iterAlignStart.totalOutcomes());

        assertEquals(3, iterAlignEnd.inputColumns());
        assertEquals(4, iterAlignEnd.totalOutcomes());

        List<DataSet> dsListAlignStart = new ArrayList<>();
        while (iterAlignStart.hasNext()) {
            dsListAlignStart.add(iterAlignStart.next());
        }

        List<DataSet> dsListAlignEnd = new ArrayList<>();
        while (iterAlignEnd.hasNext()) {
            dsListAlignEnd.add(iterAlignEnd.next());
        }

        assertEquals(3, dsListAlignStart.size());  //3 files
        assertEquals(3, dsListAlignEnd.size());  //3 files

        for (int i = 0; i < 3; i++) {
            DataSet ds = dsListAlignStart.get(i);
            INDArray features = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();
            assertEquals(1, features.size(0));   //1 example in mini-batch
            assertEquals(1, labels.size(0));
            assertEquals(3, features.size(1));   //3 values per line/time step
            assertEquals(4, labels.size(1));     //1 value per line, but 4 possible values -> one-hot vector
            assertEquals(4, features.size(2));   //sequence length = 4
            assertEquals(4, labels.size(2));

            DataSet ds2 = dsListAlignEnd.get(i);
            features = ds2.getFeatureMatrix();
            labels = ds2.getLabels();
            assertEquals(1, features.size(0));   //1 example in mini-batch
            assertEquals(1, labels.size(0));
            assertEquals(3, features.size(1));   //3 values per line/time step
            assertEquals(4, labels.size(1));     //1 value per line, but 4 possible values -> one-hot vector
            assertEquals(4, features.size(2));   //sequence length = 4
            assertEquals(4, labels.size(2));
        }

        //Check features vs. expected:
        //Here: labels always longer than features -> same features for align start and align end
        INDArray expF0 = Nd4j.create(1, 3, 4);
        expF0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 2}));
        expF0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{10, 11, 12}));
        expF0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{20, 21, 22}));
        expF0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{30, 31, 32}));
        assertEquals(dsListAlignStart.get(0).getFeatureMatrix(), expF0);
        assertEquals(dsListAlignEnd.get(0).getFeatureMatrix(), expF0);

        INDArray expF1 = Nd4j.create(1, 3, 4);
        expF1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{100, 101, 102}));
        expF1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{110, 111, 112}));
        expF1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{120, 121, 122}));
        expF1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{130, 131, 132}));
        assertEquals(dsListAlignStart.get(1).getFeatureMatrix(), expF1);
        assertEquals(dsListAlignEnd.get(1).getFeatureMatrix(), expF1);

        INDArray expF2 = Nd4j.create(1, 3, 4);
        expF2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{200, 201, 202}));
        expF2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{210, 211, 212}));
        expF2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{220, 221, 222}));
        expF2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{230, 231, 232}));
        assertEquals(dsListAlignStart.get(2).getFeatureMatrix(), expF2);
        assertEquals(dsListAlignEnd.get(2).getFeatureMatrix(), expF2);

        //Check features mask array:
        INDArray featuresMaskExpected = Nd4j.ones(1,4); //1 example, 4 values: same for both start/end align here
        for( int i=0; i<3; i++ ){
            INDArray featuresMaskStart = dsListAlignStart.get(i).getFeaturesMaskArray();
            INDArray featuresMaskEnd = dsListAlignEnd.get(i).getFeaturesMaskArray();
            assertEquals(featuresMaskExpected, featuresMaskStart);
            assertEquals(featuresMaskExpected, featuresMaskEnd);
        }


        //Check labels vs. expected:
        //First: aligning start
        INDArray expL0 = Nd4j.create(1, 4, 4);
        expL0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1, 0, 0, 0}));
        expL0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL0, dsListAlignStart.get(0).getLabels());

        INDArray expL1 = Nd4j.create(1, 4, 4);
        expL1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL1, dsListAlignStart.get(1).getLabels());

        INDArray expL2 = Nd4j.create(1, 4, 4);
        expL2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 0, 0, 1}));
        expL2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 0, 1, 0}));
        expL2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL2, dsListAlignStart.get(2).getLabels());

        //Second: align end
        INDArray expL0end = Nd4j.create(1, 4, 4);
        expL0end.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{1, 0, 0, 0}));
        expL0end.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL0end, dsListAlignEnd.get(0).getLabels());

        INDArray expL1end = Nd4j.create(1, 4, 4);
        expL1end.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL1end, dsListAlignEnd.get(1).getLabels());

        INDArray expL2end = Nd4j.create(1, 4, 4);
        expL2end.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 0, 0, 1}));
        expL2end.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 0, 1, 0}));
        expL2end.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 1, 0, 0}));
        assertEquals(expL2end, dsListAlignEnd.get(2).getLabels());

        //Check labels mask array
        INDArray[] labelsMaskExpectedStart = new INDArray[]{
                Nd4j.create(new float[]{1,1,0,0},new int[]{1,4}),
                Nd4j.create(new float[]{1,0,0,0},new int[]{1,4}),
                Nd4j.create(new float[]{1,1,1,0},new int[]{1,4})};
        INDArray[] labelsMaskExpectedEnd = new INDArray[]{
                Nd4j.create(new float[]{0,0,1,1},new int[]{1,4}),
                Nd4j.create(new float[]{0,0,0,1},new int[]{1,4}),
                Nd4j.create(new float[]{0,1,1,1},new int[]{1,4})};

        for( int i=0; i<3; i++ ){
            INDArray labelsMaskStart = dsListAlignStart.get(i).getLabelsMaskArray();
            INDArray labelsMaskEnd = dsListAlignEnd.get(i).getLabelsMaskArray();
            assertEquals(labelsMaskExpectedStart[i], labelsMaskStart);
            assertEquals(labelsMaskExpectedEnd[i], labelsMaskEnd);
        }
    }

    @Test
    public void testSequenceRecordReaderSingleReader() throws Exception{
        ClassPathResource resource = new ClassPathResource("csvsequenceSingle_0.txt");
        String path = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader reader = new CSVSequenceRecordReader(1, ",");
        reader.initialize(new NumberedFileInputSplit(path, 0, 2));
        SequenceRecordReaderDataSetIterator iteratorClassification = new SequenceRecordReaderDataSetIterator(reader, 1, 3, 0, false);

        SequenceRecordReader reader2 = new CSVSequenceRecordReader(1, ",");
        reader2.initialize(new NumberedFileInputSplit(path, 0, 2));
        SequenceRecordReaderDataSetIterator iteratorRegression = new SequenceRecordReaderDataSetIterator(reader2, 1, 3, 0, true);

        INDArray expF0 = Nd4j.create(1, 2, 4);
        expF0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1, 2}));
        expF0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{11, 12}));
        expF0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{21, 22}));
        expF0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{31, 32}));

        INDArray expF1 = Nd4j.create(1, 2, 4);
        expF1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{101, 102}));
        expF1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{111, 112}));
        expF1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{121, 122}));
        expF1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{131, 132}));

        INDArray expF2 = Nd4j.create(1, 2, 4);
        expF2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{201, 202}));
        expF2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{211, 212}));
        expF2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{221, 222}));
        expF2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{231, 232}));

        INDArray[] expF = new INDArray[]{expF0,expF1,expF2};

        //Expected out for classification:
        INDArray expOut0 = Nd4j.create(1, 3, 4);
        expOut0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1, 0, 0}));
        expOut0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 1, 0}));
        expOut0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 0, 1}));
        expOut0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{1, 0, 0}));

        INDArray expOut1 = Nd4j.create(1, 3, 4);
        expOut1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 0}));
        expOut1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0, 0, 1}));
        expOut1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{1, 0, 0}));
        expOut1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 0, 1}));

        INDArray expOut2 = Nd4j.create(1, 3, 4);
        expOut2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0, 1, 0}));
        expOut2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{1, 0, 0}));
        expOut2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0, 1, 0}));
        expOut2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0, 0, 1}));
        INDArray[] expOutClassification = new INDArray[]{expOut0,expOut1,expOut2};

        //Expected out for regression:
        INDArray expOutR0 = Nd4j.create(1, 1, 4);
        expOutR0.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{0}));
        expOutR0.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{1}));
        expOutR0.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{2}));
        expOutR0.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{0}));

        INDArray expOutR1 = Nd4j.create(1, 1, 4);
        expOutR1.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1}));
        expOutR1.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{2}));
        expOutR1.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{0}));
        expOutR1.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{2}));

        INDArray expOutR2 = Nd4j.create(1, 1, 4);
        expOutR2.tensorAlongDimension(0, 1).assign(Nd4j.create(new double[]{1}));
        expOutR2.tensorAlongDimension(1, 1).assign(Nd4j.create(new double[]{0}));
        expOutR2.tensorAlongDimension(2, 1).assign(Nd4j.create(new double[]{1}));
        expOutR2.tensorAlongDimension(3, 1).assign(Nd4j.create(new double[]{2}));
        INDArray[] expOutRegression = new INDArray[]{expOutR0,expOutR1,expOutR2};

        int countC = 0;
        while(iteratorClassification.hasNext()){
            DataSet ds = iteratorClassification.next();
            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            assertNull(ds.getFeaturesMaskArray());
            assertNull(ds.getLabelsMaskArray());

            assertArrayEquals(new int[]{1, 2, 4}, f.shape());
            assertArrayEquals(new int[]{1, 3, 4}, l.shape()); //One-hot representation

            assertEquals(expF[countC], f);
            assertEquals(expOutClassification[countC++],l);
        }
        assertEquals(3,countC);
        assertEquals(3,iteratorClassification.totalOutcomes());

        int countF = 0;
        while(iteratorRegression.hasNext()){
            DataSet ds = iteratorRegression.next();
            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            assertNull(ds.getFeaturesMaskArray());
            assertNull(ds.getLabelsMaskArray());

            assertArrayEquals(new int[]{1, 2, 4}, f.shape());
            assertArrayEquals(new int[]{1, 1, 4}, l.shape());   //Regression (single output)

            assertEquals(expF[countF], f);
            assertEquals(expOutRegression[countF++], l);
        }
        assertEquals(3,countF);
        assertEquals(1,iteratorRegression.totalOutcomes());
    }
}
