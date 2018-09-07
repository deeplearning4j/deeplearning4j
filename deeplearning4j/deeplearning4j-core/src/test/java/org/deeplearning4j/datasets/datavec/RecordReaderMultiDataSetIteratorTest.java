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

package org.deeplearning4j.datasets.datavec;


import com.google.common.io.Files;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.net.URI;
import java.util.*;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class RecordReaderMultiDataSetIteratorTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void testsBasic() throws Exception {
        //Load details from CSV files; single input/output -> compare to RecordReaderDataSetIterator
        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr, 10, 4, 3);

        RecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        MultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10).addReader("reader", rr2)
                        .addInput("reader", 0, 3).addOutputOneHot("reader", 4, 3).build();

        while (rrdsi.hasNext()) {
            DataSet ds = rrdsi.next();
            INDArray fds = ds.getFeatures();
            INDArray lds = ds.getLabels();

            MultiDataSet mds = rrmdsi.next();
            assertEquals(1, mds.getFeatures().length);
            assertEquals(1, mds.getLabels().length);
            assertNull(mds.getFeaturesMaskArrays());
            assertNull(mds.getLabelsMaskArrays());
            INDArray fmds = mds.getFeatures(0);
            INDArray lmds = mds.getLabels(0);

            assertNotNull(fmds);
            assertNotNull(lmds);

            assertEquals(fds, fmds);
            assertEquals(lds, lmds);
        }
        assertFalse(rrmdsi.hasNext());

        //need to manually extract
        File rootDir = temporaryFolder.newFolder();
        for (int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabels_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt", i)).getTempFileFromArchive(rootDir);
        }

        //Load time series from CSV sequence files; compare to SequenceRecordReaderDataSetIterator
        String featuresPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequence_%d.txt");
        String labelsPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequencelabels_%d.txt");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                        new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false);

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        MultiDataSetIterator srrmdsi = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("in", featureReader2).addSequenceReader("out", labelReader2).addInput("in")
                        .addOutputOneHot("out", 0, 4).build();

        while (iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray fds = ds.getFeatures();
            INDArray lds = ds.getLabels();

            MultiDataSet mds = srrmdsi.next();
            assertEquals(1, mds.getFeatures().length);
            assertEquals(1, mds.getLabels().length);
            assertNull(mds.getFeaturesMaskArrays());
            assertNull(mds.getLabelsMaskArrays());
            INDArray fmds = mds.getFeatures(0);
            INDArray lmds = mds.getLabels(0);

            assertNotNull(fmds);
            assertNotNull(lmds);

            assertEquals(fds, fmds);
            assertEquals(lds, lmds);
        }
        assertFalse(srrmdsi.hasNext());
    }

    @Test
    public void testsBasicMeta() throws Exception {
        //As per testBasic - but also loading metadata
        RecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        RecordReaderMultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                        .addReader("reader", rr2).addInput("reader", 0, 3).addOutputOneHot("reader", 4, 3).build();

        rrmdsi.setCollectMetaData(true);

        int count = 0;
        while (rrmdsi.hasNext()) {
            MultiDataSet mds = rrmdsi.next();
            MultiDataSet fromMeta = rrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(150 / 10, count);
    }

    @Test
    public void testSplittingCSV() throws Exception {
        //Here's the idea: take Iris, and split it up into 2 inputs and 2 output arrays
        //Inputs: columns 0 and 1-2
        //Outputs: columns 3, and 4->OneHot
        //need to manually extract
        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr, 10, 4, 3);

        RecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        MultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10).addReader("reader", rr2)
                        .addInput("reader", 0, 0).addInput("reader", 1, 2).addOutput("reader", 3, 3)
                        .addOutputOneHot("reader", 4, 3).build();

        while (rrdsi.hasNext()) {
            DataSet ds = rrdsi.next();
            INDArray fds = ds.getFeatures();
            INDArray lds = ds.getLabels();

            MultiDataSet mds = rrmdsi.next();
            assertEquals(2, mds.getFeatures().length);
            assertEquals(2, mds.getLabels().length);
            assertNull(mds.getFeaturesMaskArrays());
            assertNull(mds.getLabelsMaskArrays());
            INDArray[] fmds = mds.getFeatures();
            INDArray[] lmds = mds.getLabels();

            assertNotNull(fmds);
            assertNotNull(lmds);
            for (int i = 0; i < fmds.length; i++)
                assertNotNull(fmds[i]);
            for (int i = 0; i < lmds.length; i++)
                assertNotNull(lmds[i]);

            //Get the subsets of the original iris data
            INDArray expIn1 = fds.get(all(), point(0));
            INDArray expIn2 = fds.get(all(), NDArrayIndex.interval(1, 2, true));
            INDArray expOut1 = fds.get(all(), point(3));
            INDArray expOut2 = lds;

            assertEquals(expIn1, fmds[0]);
            assertEquals(expIn2, fmds[1]);
            assertEquals(expOut1, lmds[0]);
            assertEquals(expOut2, lmds[1]);
        }
        assertFalse(rrmdsi.hasNext());
    }

    @Test
    public void testSplittingCSVMeta() throws Exception {
        //Here's the idea: take Iris, and split it up into 2 inputs and 2 output arrays
        //Inputs: columns 0 and 1-2
        //Outputs: columns 3, and 4->OneHot
        RecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        RecordReaderMultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                        .addReader("reader", rr2).addInput("reader", 0, 0).addInput("reader", 1, 2)
                        .addOutput("reader", 3, 3).addOutputOneHot("reader", 4, 3).build();
        rrmdsi.setCollectMetaData(true);

        int count = 0;
        while (rrmdsi.hasNext()) {
            MultiDataSet mds = rrmdsi.next();
            MultiDataSet fromMeta = rrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(150 / 10, count);
    }

    @Test
    public void testSplittingCSVSequence() throws Exception {
        //Idea: take CSV sequences, and split "csvsequence_i.txt" into two separate inputs; keep "csvSequencelables_i.txt"
        // as standard one-hot output
        //need to manually extract
        File rootDir = temporaryFolder.newFolder();
        for (int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabels_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt", i)).getTempFileFromArchive(rootDir);
        }

        String featuresPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequence_%d.txt");
        String labelsPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequencelabels_%d.txt");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                        new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false);

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        MultiDataSetIterator srrmdsi = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("seq1", featureReader2).addSequenceReader("seq2", labelReader2)
                        .addInput("seq1", 0, 1).addInput("seq1", 2, 2).addOutputOneHot("seq2", 0, 4).build();

        while (iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray fds = ds.getFeatures();
            INDArray lds = ds.getLabels();

            MultiDataSet mds = srrmdsi.next();
            assertEquals(2, mds.getFeatures().length);
            assertEquals(1, mds.getLabels().length);
            assertNull(mds.getFeaturesMaskArrays());
            assertNull(mds.getLabelsMaskArrays());
            INDArray[] fmds = mds.getFeatures();
            INDArray[] lmds = mds.getLabels();

            assertNotNull(fmds);
            assertNotNull(lmds);
            for (int i = 0; i < fmds.length; i++)
                assertNotNull(fmds[i]);
            for (int i = 0; i < lmds.length; i++)
                assertNotNull(lmds[i]);

            INDArray expIn1 = fds.get(all(), NDArrayIndex.interval(0, 1, true), all());
            INDArray expIn2 = fds.get(all(), NDArrayIndex.interval(2, 2, true), all());

            assertEquals(expIn1, fmds[0]);
            assertEquals(expIn2, fmds[1]);
            assertEquals(lds, lmds[0]);
        }
        assertFalse(srrmdsi.hasNext());
    }

    @Test
    public void testSplittingCSVSequenceMeta() throws Exception {
        //Idea: take CSV sequences, and split "csvsequence_i.txt" into two separate inputs; keep "csvSequencelables_i.txt"
        // as standard one-hot output
        //need to manually extract
        File rootDir = temporaryFolder.newFolder();
        for (int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabels_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt", i)).getTempFileFromArchive(rootDir);
        }

        String featuresPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequence_%d.txt");
        String labelsPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequencelabels_%d.txt");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        RecordReaderMultiDataSetIterator srrmdsi = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("seq1", featureReader2).addSequenceReader("seq2", labelReader2)
                        .addInput("seq1", 0, 1).addInput("seq1", 2, 2).addOutputOneHot("seq2", 0, 4).build();

        srrmdsi.setCollectMetaData(true);

        int count = 0;
        while (srrmdsi.hasNext()) {
            MultiDataSet mds = srrmdsi.next();
            MultiDataSet fromMeta = srrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(3, count);
    }


    @Test
    public void testInputValidation() {

        //Test: no readers
        try {
            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1).addInput("something")
                            .addOutput("something").build();
            fail("Should have thrown exception");
        } catch (Exception e) {
        }

        //Test: reference to reader that doesn't exist
        try {
            RecordReader rr = new CSVRecordReader(0, ',');
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1).addReader("iris", rr)
                            .addInput("thisDoesntExist", 0, 3).addOutputOneHot("iris", 4, 3).build();
            fail("Should have thrown exception");
        } catch (Exception e) {
        }

        //Test: no inputs or outputs
        try {
            RecordReader rr = new CSVRecordReader(0, ',');
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1).addReader("iris", rr).build();
            fail("Should have thrown exception");
        } catch (Exception e) {
        }
    }

    @Test
    public void testVariableLengthTS() throws Exception {
        //need to manually extract
        File rootDir = temporaryFolder.newFolder();
        for (int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabels_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt", i)).getTempFileFromArchive(rootDir);
        }

        String featuresPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequence_%d.txt");
        String labelsPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequencelabelsShort_%d.txt");

        //Set up SequenceRecordReaderDataSetIterators for comparison

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iterAlignStart = new SequenceRecordReaderDataSetIterator(featureReader,
                        labelReader, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);

        SequenceRecordReaderDataSetIterator iterAlignEnd = new SequenceRecordReaderDataSetIterator(featureReader2,
                        labelReader2, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        //Set up
        SequenceRecordReader featureReader3 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader3 = new CSVSequenceRecordReader(1, ",");
        featureReader3.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader3.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader4 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader4 = new CSVSequenceRecordReader(1, ",");
        featureReader4.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader4.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        RecordReaderMultiDataSetIterator rrmdsiStart = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("in", featureReader3).addSequenceReader("out", labelReader3).addInput("in")
                        .addOutputOneHot("out", 0, 4)
                        .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START).build();

        RecordReaderMultiDataSetIterator rrmdsiEnd = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("in", featureReader4).addSequenceReader("out", labelReader4).addInput("in")
                        .addOutputOneHot("out", 0, 4)
                        .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END).build();


        while (iterAlignStart.hasNext()) {
            DataSet dsStart = iterAlignStart.next();
            DataSet dsEnd = iterAlignEnd.next();

            MultiDataSet mdsStart = rrmdsiStart.next();
            MultiDataSet mdsEnd = rrmdsiEnd.next();

            assertEquals(1, mdsStart.getFeatures().length);
            assertEquals(1, mdsStart.getLabels().length);
            //assertEquals(1, mdsStart.getFeaturesMaskArrays().length); //Features data is always longer -> don't need mask arrays for it
            assertEquals(1, mdsStart.getLabelsMaskArrays().length);

            assertEquals(1, mdsEnd.getFeatures().length);
            assertEquals(1, mdsEnd.getLabels().length);
            //assertEquals(1, mdsEnd.getFeaturesMaskArrays().length);
            assertEquals(1, mdsEnd.getLabelsMaskArrays().length);


            assertEquals(dsStart.getFeatures(), mdsStart.getFeatures(0));
            assertEquals(dsStart.getLabels(), mdsStart.getLabels(0));
            assertEquals(dsStart.getLabelsMaskArray(), mdsStart.getLabelsMaskArray(0));

            assertEquals(dsEnd.getFeatures(), mdsEnd.getFeatures(0));
            assertEquals(dsEnd.getLabels(), mdsEnd.getLabels(0));
            assertEquals(dsEnd.getLabelsMaskArray(), mdsEnd.getLabelsMaskArray(0));
        }
        assertFalse(rrmdsiStart.hasNext());
        assertFalse(rrmdsiEnd.hasNext());
    }


    @Test
    public void testVariableLengthTSMeta() throws Exception {
        //need to manually extract
        File rootDir = temporaryFolder.newFolder();
        for (int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabels_%d.txt", i)).getTempFileFromArchive(rootDir);
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt", i)).getTempFileFromArchive(rootDir);
        }
        //Set up SequenceRecordReaderDataSetIterators for comparison

        String featuresPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequence_%d.txt");
        String labelsPath = FilenameUtils.concat(rootDir.getAbsolutePath(), "csvsequencelabelsShort_%d.txt");

        //Set up
        SequenceRecordReader featureReader3 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader3 = new CSVSequenceRecordReader(1, ",");
        featureReader3.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader3.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader4 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader4 = new CSVSequenceRecordReader(1, ",");
        featureReader4.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader4.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        RecordReaderMultiDataSetIterator rrmdsiStart = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("in", featureReader3).addSequenceReader("out", labelReader3).addInput("in")
                        .addOutputOneHot("out", 0, 4)
                        .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START).build();

        RecordReaderMultiDataSetIterator rrmdsiEnd = new RecordReaderMultiDataSetIterator.Builder(1)
                        .addSequenceReader("in", featureReader4).addSequenceReader("out", labelReader4).addInput("in")
                        .addOutputOneHot("out", 0, 4)
                        .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END).build();

        rrmdsiStart.setCollectMetaData(true);
        rrmdsiEnd.setCollectMetaData(true);

        int count = 0;
        while (rrmdsiStart.hasNext()) {
            MultiDataSet mdsStart = rrmdsiStart.next();
            MultiDataSet mdsEnd = rrmdsiEnd.next();

            MultiDataSet mdsStartFromMeta =
                            rrmdsiStart.loadFromMetaData(mdsStart.getExampleMetaData(RecordMetaData.class));
            MultiDataSet mdsEndFromMeta = rrmdsiEnd.loadFromMetaData(mdsEnd.getExampleMetaData(RecordMetaData.class));

            assertEquals(mdsStart, mdsStartFromMeta);
            assertEquals(mdsEnd, mdsEndFromMeta);

            count++;
        }
        assertFalse(rrmdsiStart.hasNext());
        assertFalse(rrmdsiEnd.hasNext());
        assertEquals(3, count);
    }

    @Test
    public void testImagesRRDMSI() throws Exception {
        File parentDir = temporaryFolder.newFolder();
        parentDir.deleteOnExit();
        String str1 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Zico/");
        String str2 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Ziwang_Xu/");

        File f1 = new File(str1);
        File f2 = new File(str2);
        f1.mkdirs();
        f2.mkdirs();

        TestUtils.writeStreamToFile(new File(FilenameUtils.concat(f1.getPath(), "Zico_0001.jpg")),
                        new ClassPathResource("lfwtest/Zico/Zico_0001.jpg").getInputStream());
        TestUtils.writeStreamToFile(new File(FilenameUtils.concat(f2.getPath(), "Ziwang_Xu_0001.jpg")),
                        new ClassPathResource("lfwtest/Ziwang_Xu/Ziwang_Xu_0001.jpg").getInputStream());


        int outputNum = 2;
        Random r = new Random(12345);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader rr1 = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s = new ImageRecordReader(5, 5, 1, labelMaker);

        rr1.initialize(new FileSplit(parentDir));
        rr1s.initialize(new FileSplit(parentDir));


        MultiDataSetIterator trainDataIterator = new RecordReaderMultiDataSetIterator.Builder(1).addReader("rr1", rr1)
                        .addReader("rr1s", rr1s).addInput("rr1", 0, 0).addInput("rr1s", 0, 0)
                        .addOutputOneHot("rr1s", 1, outputNum).build();

        //Now, do the same thing with ImageRecordReader, and check we get the same results:
        ImageRecordReader rr1_b = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s_b = new ImageRecordReader(5, 5, 1, labelMaker);
        rr1_b.initialize(new FileSplit(parentDir));
        rr1s_b.initialize(new FileSplit(parentDir));

        DataSetIterator dsi1 = new RecordReaderDataSetIterator(rr1_b, 1, 1, 2);
        DataSetIterator dsi2 = new RecordReaderDataSetIterator(rr1s_b, 1, 1, 2);

        for (int i = 0; i < 2; i++) {
            MultiDataSet mds = trainDataIterator.next();

            DataSet d1 = dsi1.next();
            DataSet d2 = dsi2.next();

            assertEquals(d1.getFeatures(), mds.getFeatures(0));
            assertEquals(d2.getFeatures(), mds.getFeatures(1));
            assertEquals(d1.getLabels(), mds.getLabels(0));
        }
    }

    @Test
    public void testImagesRRDMSI_Batched() throws Exception {
        File parentDir = temporaryFolder.newFolder();
        parentDir.deleteOnExit();
        String str1 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Zico/");
        String str2 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Ziwang_Xu/");

        File f1 = new File(str1);
        File f2 = new File(str2);
        f1.mkdirs();
        f2.mkdirs();

        TestUtils.writeStreamToFile(new File(FilenameUtils.concat(f1.getPath(), "Zico_0001.jpg")),
                        new ClassPathResource("lfwtest/Zico/Zico_0001.jpg").getInputStream());
        TestUtils.writeStreamToFile(new File(FilenameUtils.concat(f2.getPath(), "Ziwang_Xu_0001.jpg")),
                        new ClassPathResource("lfwtest/Ziwang_Xu/Ziwang_Xu_0001.jpg").getInputStream());

        int outputNum = 2;
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader rr1 = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s = new ImageRecordReader(5, 5, 1, labelMaker);

        URI[] uris = new FileSplit(parentDir).locations();

        rr1.initialize(new CollectionInputSplit(uris));
        rr1s.initialize(new CollectionInputSplit(uris));

        MultiDataSetIterator trainDataIterator = new RecordReaderMultiDataSetIterator.Builder(2).addReader("rr1", rr1)
                        .addReader("rr1s", rr1s).addInput("rr1", 0, 0).addInput("rr1s", 0, 0)
                        .addOutputOneHot("rr1s", 1, outputNum).build();

        //Now, do the same thing with ImageRecordReader, and check we get the same results:
        ImageRecordReader rr1_b = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s_b = new ImageRecordReader(5, 5, 1, labelMaker);
        rr1_b.initialize(new FileSplit(parentDir));
        rr1s_b.initialize(new FileSplit(parentDir));

        DataSetIterator dsi1 = new RecordReaderDataSetIterator(rr1_b, 2, 1, 2);
        DataSetIterator dsi2 = new RecordReaderDataSetIterator(rr1s_b, 2, 1, 2);

        MultiDataSet mds = trainDataIterator.next();

        DataSet d1 = dsi1.next();
        DataSet d2 = dsi2.next();

        assertEquals(d1.getFeatures(), mds.getFeatures(0));
        assertEquals(d2.getFeatures(), mds.getFeatures(1));
        assertEquals(d1.getLabels(), mds.getLabels(0));

        //Check label assignment:

        File currentFile = rr1_b.getCurrentFile();
        INDArray expLabels;
        if(currentFile.getAbsolutePath().contains("Zico")){
            expLabels = Nd4j.create(new double[][] {{0, 1}, {1, 0}});
        } else {
            expLabels = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        }

        assertEquals(expLabels, d1.getLabels());
        assertEquals(expLabels, d2.getLabels());
    }
    



    @Test
    public void testTimeSeriesRandomOffset() {
        //2 in, 2 out, 3 total sequences of length [1,3,5]

        List<List<Writable>> seq1 =
                        Arrays.asList(Arrays.<Writable>asList(new DoubleWritable(1.0), new DoubleWritable(2.0)));
        List<List<Writable>> seq2 =
                        Arrays.asList(Arrays.<Writable>asList(new DoubleWritable(10.0), new DoubleWritable(11.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(20.0), new DoubleWritable(21.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(30.0), new DoubleWritable(31.0)));
        List<List<Writable>> seq3 =
                        Arrays.asList(Arrays.<Writable>asList(new DoubleWritable(100.0), new DoubleWritable(101.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(200.0), new DoubleWritable(201.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(300.0), new DoubleWritable(301.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(400.0), new DoubleWritable(401.0)),
                                        Arrays.<Writable>asList(new DoubleWritable(500.0), new DoubleWritable(501.0)));

        Collection<List<List<Writable>>> seqs = Arrays.asList(seq1, seq2, seq3);

        SequenceRecordReader rr = new CollectionSequenceRecordReader(seqs);

        RecordReaderMultiDataSetIterator rrmdsi =
                        new RecordReaderMultiDataSetIterator.Builder(3).addSequenceReader("rr", rr).addInput("rr", 0, 0)
                                        .addOutput("rr", 1, 1).timeSeriesRandomOffset(true, 1234L).build();


        Random r = new Random(1234); //Provides seed for each minibatch
        long seed = r.nextLong();
        Random r2 = new Random(seed); //Use same RNG seed in new RNG for each minibatch
        int expOffsetSeq1 = r2.nextInt(5 - 1 + 1); //0 to 4 inclusive
        int expOffsetSeq2 = r2.nextInt(5 - 3 + 1);
        int expOffsetSeq3 = 0; //Longest TS, always 0
        //With current seed: 3, 1, 0
        //        System.out.println(expOffsetSeq1 + "\t" + expOffsetSeq2 + "\t" + expOffsetSeq3);

        MultiDataSet mds = rrmdsi.next();

        INDArray expMask = Nd4j.create(new double[][] {{0, 0, 0, 1, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}});

        assertEquals(expMask, mds.getFeaturesMaskArray(0));
        assertEquals(expMask, mds.getLabelsMaskArray(0));

        INDArray f = mds.getFeatures(0);
        INDArray l = mds.getLabels(0);

        INDArray expF1 = Nd4j.create(new double[] {1.0});
        INDArray expL1 = Nd4j.create(new double[] {2.0});

        INDArray expF2 = Nd4j.create(new double[] {10, 20, 30});
        INDArray expL2 = Nd4j.create(new double[] {11, 21, 31});

        INDArray expF3 = Nd4j.create(new double[] {100, 200, 300, 400, 500});
        INDArray expL3 = Nd4j.create(new double[] {101, 201, 301, 401, 501});

        assertEquals(expF1, f.get(point(0), all(),
                        NDArrayIndex.interval(expOffsetSeq1, expOffsetSeq1 + 1)));
        assertEquals(expL1, l.get(point(0), all(),
                        NDArrayIndex.interval(expOffsetSeq1, expOffsetSeq1 + 1)));

        assertEquals(expF2, f.get(point(1), all(),
                        NDArrayIndex.interval(expOffsetSeq2, expOffsetSeq2 + 3)));
        assertEquals(expL2, l.get(point(1), all(),
                        NDArrayIndex.interval(expOffsetSeq2, expOffsetSeq2 + 3)));

        assertEquals(expF3, f.get(point(2), all(),
                        NDArrayIndex.interval(expOffsetSeq3, expOffsetSeq3 + 5)));
        assertEquals(expL3, l.get(point(2), all(),
                        NDArrayIndex.interval(expOffsetSeq3, expOffsetSeq3 + 5)));
    }


    @Test
    public void testSeqRRDSIMasking(){
        //This also tests RecordReaderMultiDataSetIterator, by virtue of
        List<List<List<Writable>>> features = new ArrayList<>();
        List<List<List<Writable>>> labels = new ArrayList<>();

        features.add(Arrays.asList(l(new DoubleWritable(1)), l(new DoubleWritable(2)), l(new DoubleWritable(3))));
        features.add(Arrays.asList(l(new DoubleWritable(4)), l(new DoubleWritable(5))));

        labels.add(Arrays.asList(l(new IntWritable(0))));
        labels.add(Arrays.asList(l(new IntWritable(1))));

        CollectionSequenceRecordReader fR = new CollectionSequenceRecordReader(features);
        CollectionSequenceRecordReader lR = new CollectionSequenceRecordReader(labels);

        SequenceRecordReaderDataSetIterator seqRRDSI = new SequenceRecordReaderDataSetIterator(
                fR, lR, 2, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        DataSet ds = seqRRDSI.next();

        INDArray fMask = Nd4j.create(new double[][]{
                {1,1,1},
                {1,1,0}});

        INDArray lMask = Nd4j.create(new double[][]{
                {0,0,1},
                {0,1,0}});

        assertEquals(fMask, ds.getFeaturesMaskArray());
        assertEquals(lMask, ds.getLabelsMaskArray());

        INDArray f = Nd4j.create(new double[][]{
                {1,2,3},
                {4,5,0}});

        INDArray l = Nd4j.create(2,2,3);
        l.putScalar(0,0,2, 1.0);
        l.putScalar(1,1,1, 1.0);

        assertEquals(f, ds.getFeatures().get(all(), point(0), all()));
        assertEquals(l, ds.getLabels());
    }

    private static List<Writable> l(Writable... in){
        return Arrays.asList(in);
    }



    @Test
    public void testExcludeStringColCSV() throws Exception {
        File csvFile = temporaryFolder.newFile();

        StringBuilder sb = new StringBuilder();
        for(int i=1; i<=10; i++ ){
            if(i > 1){
                sb.append("\n");
            }
            sb.append("skip_").append(i).append(",").append(i).append(",").append(i + 0.5);
        }
        FileUtils.writeStringToFile(csvFile, sb.toString());

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(csvFile));

        RecordReaderMultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("rr", rr)
                .addInput("rr", 1, 1)
                .addOutput("rr", 2, 2)
                .build();

        INDArray expFeatures = Nd4j.linspace(1,10,10).transpose();
        INDArray expLabels = Nd4j.linspace(1,10,10).addi(0.5).transpose();

        MultiDataSet mds = rrmdsi.next();
        assertFalse(rrmdsi.hasNext());

        assertEquals(expFeatures, mds.getFeatures(0));
        assertEquals(expLabels, mds.getLabels(0));
    }


    private static final int nX = 32;
    private static final int nY = 32;
    private static final int nZ = 28;


    @Test
    public void testRRMDSI5D() {
        int batchSize = 5;

        CustomRecordReader recordReader = new CustomRecordReader();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize,
                1, /* Index of label in records */
                2 /* number of different labels */);

        int count = 0;
        while(dataIter.hasNext()){
            DataSet ds = dataIter.next();

            int offset = 5*count;
            for( int i=0; i<5; i++ ){
                INDArray act = ds.getFeatures().get(interval(i,i,true), all(), all(), all(), all());
                INDArray exp = Nd4j.valueArrayOf(new int[]{1, 1, nZ, nX, nY}, i + offset );
                assertEquals(exp, act);
            }
            count++;
        }

        assertEquals(2, count);
    }


    static class CustomRecordReader extends BaseRecordReader {

        int n = 0;

        CustomRecordReader() { }

        @Override
        public boolean batchesSupported() {
            return false;
        }

        @Override
        public List<List<Writable>> next(int num) {
            throw new RuntimeException("Not implemented");
        }

        @Override
        public List<Writable> next() {
            INDArray nd = Nd4j.create(new float[nZ*nY*nX], new int[] {1, 1, nZ, nY, nX }, 'C').assign(n);
            final List<Writable>res = RecordConverter.toRecord(nd);
            res.add(new IntWritable(0));
            n++;
            return res;
        }

        @Override
        public boolean hasNext() {
            return n<10;
        }

        final static ArrayList<String> labels = new ArrayList<>(2);
        static {
            labels.add("lbl0");
            labels.add("lbl1");
        }
        @Override
        public List<String> getLabels() {
            return labels;
        }

        @Override
        public void reset() {
            n = 0;
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public List<Writable> record(URI uri, DataInputStream dataInputStream) {
            return next();
        }

        @Override
        public Record nextRecord() {
            List<Writable> r = next();
            return new org.datavec.api.records.impl.Record(r, null);
        }

        @Override
        public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
            throw new RuntimeException("Not implemented");
        }

        @Override
        public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) {
            throw new RuntimeException("Not implemented");
        }

        @Override
        public void close() {
        }

        @Override
        public void setConf(Configuration conf) {
        }

        @Override
        public Configuration getConf() {
            return null;
        }

        @Override
        public void initialize(InputSplit split) {
            n = 0;
        }
        @Override
        public void initialize(Configuration conf, InputSplit split) {
            n = 0;
        }
    }
}
