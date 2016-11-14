package org.deeplearning4j.datasets.datavec;


import com.google.common.io.Files;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.util.Random;

import static org.junit.Assert.*;

public class RecordReaderMultiDataSetIteratorTest {

    @Test
    public void testsBasic() throws Exception {
        //Load details from CSV files; single input/output -> compare to RecordReaderDataSetIterator
        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr,10,4,3);

        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        MultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("reader",rr2)
                .addInput("reader",0,3)
                .addOutputOneHot("reader",4,3)
                .build();

        while(rrdsi.hasNext()){
            DataSet ds = rrdsi.next();
            INDArray fds = ds.getFeatureMatrix();
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
            assertEquals(lds,lmds);
        }
        assertFalse(rrmdsi.hasNext());

        //need to manually extract
        for(int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabels_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt",i)).getTempFileFromArchive();
        }

        //Load time series from CSV sequence files; compare to SequenceRecordReaderDataSetIterator
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabels_0.txt");
        String labelsPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");

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
                .addSequenceReader("in",featureReader2)
                .addSequenceReader("out",labelReader2)
                .addInput("in")
                .addOutputOneHot("out",0,4)
                .build();

        while(iter.hasNext()){
            DataSet ds = iter.next();
            INDArray fds = ds.getFeatureMatrix();
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
            assertEquals(lds,lmds);
        }
        assertFalse(srrmdsi.hasNext());
    }

    @Test
    public void testsBasicMeta() throws Exception {
        //As per testBasic - but also loading metadata
        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        RecordReaderMultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("reader",rr2)
                .addInput("reader",0,3)
                .addOutputOneHot("reader",4,3)
                .build();

        rrmdsi.setCollectMetaData(true);

        int count = 0;
        while(rrmdsi.hasNext()){
            MultiDataSet mds = rrmdsi.next();
            MultiDataSet fromMeta = rrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(150/10, count);
    }

    @Test
    public void testSplittingCSV() throws Exception{
        //Here's the idea: take Iris, and split it up into 2 inputs and 2 output arrays
        //Inputs: columns 0 and 1-2
        //Outputs: columns 3, and 4->OneHot
        //need to manually extract
        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr,10,4,3);

        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        MultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("reader",rr2)
                .addInput("reader",0,0)
                .addInput("reader",1,2)
                .addOutput("reader",3,3)
                .addOutputOneHot("reader", 4, 3)
                .build();

        while(rrdsi.hasNext()){
            DataSet ds = rrdsi.next();
            INDArray fds = ds.getFeatureMatrix();
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
            for( int i=0; i<fmds.length; i++ ) assertNotNull(fmds[i]);
            for( int i=0; i<lmds.length; i++ ) assertNotNull(lmds[i]);

            //Get the subsets of the original iris data
            INDArray expIn1 = fds.get(NDArrayIndex.all(), NDArrayIndex.point(0));
            INDArray expIn2 = fds.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, true));
            INDArray expOut1 = fds.get(NDArrayIndex.all(), NDArrayIndex.point(3));
            INDArray expOut2 = lds;

            assertEquals(expIn1,fmds[0]);
            assertEquals(expIn2,fmds[1]);
            assertEquals(expOut1,lmds[0]);
            assertEquals(expOut2,lmds[1]);
        }
        assertFalse(rrmdsi.hasNext());
    }

    @Test
    public void testSplittingCSVMeta() throws Exception{
        //Here's the idea: take Iris, and split it up into 2 inputs and 2 output arrays
        //Inputs: columns 0 and 1-2
        //Outputs: columns 3, and 4->OneHot
        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        RecordReaderMultiDataSetIterator rrmdsi = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("reader",rr2)
                .addInput("reader",0,0)
                .addInput("reader",1,2)
                .addOutput("reader",3,3)
                .addOutputOneHot("reader", 4, 3)
                .build();
        rrmdsi.setCollectMetaData(true);

        int count = 0;
        while(rrmdsi.hasNext()){
            MultiDataSet mds = rrmdsi.next();
            MultiDataSet fromMeta = rrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(150/10, count);
    }

    @Test
    public void testSplittingCSVSequence() throws Exception {
        //Idea: take CSV sequences, and split "csvsequence_i.txt" into two separate inputs; keep "csvSequencelables_i.txt"
        // as standard one-hot output
        //need to manually extract
        for(int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabels_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt",i)).getTempFileFromArchive();
        }

        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabels_0.txt");
        String labelsPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");

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
                .addSequenceReader("seq1", featureReader2)
                .addSequenceReader("seq2", labelReader2)
                .addInput("seq1",0,1)
                .addInput("seq1", 2, 2)
                .addOutputOneHot("seq2",0,4)
                .build();

        while(iter.hasNext()){
            DataSet ds = iter.next();
            INDArray fds = ds.getFeatureMatrix();
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
            for( int i=0; i<fmds.length; i++ ) assertNotNull(fmds[i]);
            for( int i=0; i<lmds.length; i++ ) assertNotNull(lmds[i]);

            INDArray expIn1 = fds.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 1, true), NDArrayIndex.all());
            INDArray expIn2 = fds.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 2, true), NDArrayIndex.all());

            assertEquals(expIn1,fmds[0]);
            assertEquals(expIn2,fmds[1]);
            assertEquals(lds,lmds[0]);
        }
        assertFalse(srrmdsi.hasNext());
    }

    @Test
    public void testSplittingCSVSequenceMeta() throws Exception {
        //Idea: take CSV sequences, and split "csvsequence_i.txt" into two separate inputs; keep "csvSequencelables_i.txt"
        // as standard one-hot output
        //need to manually extract
        for(int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabels_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt",i)).getTempFileFromArchive();
        }

        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabels_0.txt");
        String labelsPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(1, ",");
        featureReader2.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader2.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        RecordReaderMultiDataSetIterator srrmdsi = new RecordReaderMultiDataSetIterator.Builder(1)
                .addSequenceReader("seq1", featureReader2)
                .addSequenceReader("seq2", labelReader2)
                .addInput("seq1",0,1)
                .addInput("seq1", 2, 2)
                .addOutputOneHot("seq2",0,4)
                .build();

        srrmdsi.setCollectMetaData(true);

        int count = 0;
        while(srrmdsi.hasNext()){
            MultiDataSet mds = srrmdsi.next();
            MultiDataSet fromMeta = srrmdsi.loadFromMetaData(mds.getExampleMetaData(RecordMetaData.class));
            assertEquals(mds, fromMeta);
            count++;
        }
        assertEquals(3, count);
    }


    @Test
    public void testInputValidation(){

        //Test: no readers
        try{
            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1)
                    .addInput("something")
                    .addOutput("something")
                    .build();
            fail("Should have thrown exception");
        }catch(Exception e){ }

        //Test: reference to reader that doesn't exist
        try{
            RecordReader rr = new CSVRecordReader(0,",");
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1)
                    .addReader("iris",rr)
                    .addInput("thisDoesntExist", 0, 3)
                    .addOutputOneHot("iris", 4, 3)
                    .build();
            fail("Should have thrown exception");
        }catch(Exception e){ }

        //Test: no inputs or outputs
        try{
            RecordReader rr = new CSVRecordReader(0,",");
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1)
                    .addReader("iris", rr)
                    .build();
            fail("Should have thrown exception");
        }catch(Exception e){ }
    }

    @Test
    public void testVariableLengthTS() throws Exception {
        //need to manually extract
        for(int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabels_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt",i)).getTempFileFromArchive();
        }
        //Set up SequenceRecordReaderDataSetIterators for comparison
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabelsShort_0.txt");
        String labelsPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");

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
                .addSequenceReader("in",featureReader3)
                .addSequenceReader("out",labelReader3)
                .addInput("in")
                .addOutputOneHot("out",0,4)
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START)
                .build();

        RecordReaderMultiDataSetIterator rrmdsiEnd = new RecordReaderMultiDataSetIterator.Builder(1)
                .addSequenceReader("in",featureReader4)
                .addSequenceReader("out",labelReader4)
                .addInput("in")
                .addOutputOneHot("out",0,4)
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
                .build();


        while(iterAlignStart.hasNext()) {
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


            assertEquals(dsStart.getFeatureMatrix(), mdsStart.getFeatures(0));
            assertEquals(dsStart.getLabels(), mdsStart.getLabels(0));
            assertEquals(dsStart.getLabelsMaskArray(), mdsStart.getLabelsMaskArray(0));

            assertEquals(dsEnd.getFeatureMatrix(), mdsEnd.getFeatures(0));
            assertEquals(dsEnd.getLabels(), mdsEnd.getLabels(0));
            assertEquals(dsEnd.getLabelsMaskArray(), mdsEnd.getLabelsMaskArray(0));
        }
        assertFalse(rrmdsiStart.hasNext());
        assertFalse(rrmdsiEnd.hasNext());
    }


    @Test
    public void testVariableLengthTSMeta() throws Exception {
        //need to manually extract
        for(int i = 0; i < 3; i++) {
            new ClassPathResource(String.format("csvsequence_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabels_%d.txt",i)).getTempFileFromArchive();
            new ClassPathResource(String.format("csvsequencelabelsShort_%d.txt",i)).getTempFileFromArchive();
        }
        //Set up SequenceRecordReaderDataSetIterators for comparison
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("csvsequencelabelsShort_0.txt");
        String labelsPath = resource.getTempFileFromArchive().getAbsolutePath().replaceAll("0", "%d");

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
                .addSequenceReader("in",featureReader3)
                .addSequenceReader("out",labelReader3)
                .addInput("in")
                .addOutputOneHot("out",0,4)
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START)
                .build();

        RecordReaderMultiDataSetIterator rrmdsiEnd = new RecordReaderMultiDataSetIterator.Builder(1)
                .addSequenceReader("in",featureReader4)
                .addSequenceReader("out",labelReader4)
                .addInput("in")
                .addOutputOneHot("out",0,4)
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
                .build();

        rrmdsiStart.setCollectMetaData(true);
        rrmdsiEnd.setCollectMetaData(true);

        int count = 0;
        while(rrmdsiStart.hasNext()) {
            MultiDataSet mdsStart = rrmdsiStart.next();
            MultiDataSet mdsEnd = rrmdsiEnd.next();

            MultiDataSet mdsStartFromMeta = rrmdsiStart.loadFromMetaData(mdsStart.getExampleMetaData(RecordMetaData.class));
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
        File parentDir = Files.createTempDir();
        parentDir.deleteOnExit();
        String str1 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Zico/");
        String str2 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Ziwang_Xu/");

        File f1 = new File(str1);
        File f2 = new File(str2);
        f1.mkdirs();
        f2.mkdirs();

        writeStreamToFile(new File(FilenameUtils.concat(f1.getPath(),"Zico_0001.jpg")), new ClassPathResource("lfwtest/Zico/Zico_0001.jpg").getInputStream());
        writeStreamToFile(new File(FilenameUtils.concat(f2.getPath(),"Ziwang_Xu_0001.jpg")), new ClassPathResource("lfwtest/Ziwang_Xu/Ziwang_Xu_0001.jpg").getInputStream());


        int outputNum = 2;
        Random r = new Random(12345);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader rr1 = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s = new ImageRecordReader(5, 5, 1, labelMaker);

        rr1.initialize(new FileSplit(parentDir));
        rr1s.initialize(new FileSplit(parentDir));


        MultiDataSetIterator trainDataIterator = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("rr1", rr1)
                .addReader("rr1s", rr1s)
                .addInput("rr1", 0, 0)
                .addInput("rr1s", 0, 0)
                .addOutputOneHot("rr1s", 1, outputNum)
                .build();

        //Now, do the same thing with ImageRecordReader, and check we get the same results:
        ImageRecordReader rr1_b = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s_b = new ImageRecordReader(5, 5, 1, labelMaker);
        rr1_b.initialize(new FileSplit(parentDir));
        rr1s_b.initialize(new FileSplit(parentDir));

        DataSetIterator dsi1 = new RecordReaderDataSetIterator(rr1_b, 1, 1, 2);
        DataSetIterator dsi2 = new RecordReaderDataSetIterator(rr1s_b, 1, 1, 2);

        for(int i=0; i<2; i++ ) {
            MultiDataSet mds = trainDataIterator.next();

            DataSet d1 = dsi1.next();
            DataSet d2 = dsi2.next();

            assertEquals(d1.getFeatureMatrix(), mds.getFeatures(0));
            assertEquals(d2.getFeatureMatrix(), mds.getFeatures(1));
            assertEquals(d1.getLabels(), mds.getLabels(0));
        }
    }

    @Test
    public void testImagesRRDMSI_Batched() throws Exception {
        File parentDir = Files.createTempDir();
        parentDir.deleteOnExit();
        String str1 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Zico/");
        String str2 = FilenameUtils.concat(parentDir.getAbsolutePath(), "Ziwang_Xu/");

        File f1 = new File(str1);
        File f2 = new File(str2);
        f1.mkdirs();
        f2.mkdirs();

        writeStreamToFile(new File(FilenameUtils.concat(f1.getPath(),"Zico_0001.jpg")), new ClassPathResource("lfwtest/Zico/Zico_0001.jpg").getInputStream());
        writeStreamToFile(new File(FilenameUtils.concat(f2.getPath(),"Ziwang_Xu_0001.jpg")), new ClassPathResource("lfwtest/Ziwang_Xu/Ziwang_Xu_0001.jpg").getInputStream());

        int outputNum = 2;
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader rr1 = new ImageRecordReader(10, 10, 1, labelMaker);
        ImageRecordReader rr1s = new ImageRecordReader(5, 5, 1, labelMaker);

        rr1.initialize(new FileSplit(parentDir));
        rr1s.initialize(new FileSplit(parentDir));

        MultiDataSetIterator trainDataIterator = new RecordReaderMultiDataSetIterator.Builder(2)
                .addReader("rr1", rr1)
                .addReader("rr1s", rr1s)
                .addInput("rr1", 0, 0)
                .addInput("rr1s", 0, 0)
                .addOutputOneHot("rr1s", 1, outputNum)
                .build();

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

        assertEquals(d1.getFeatureMatrix(), mds.getFeatures(0));
        assertEquals(d2.getFeatureMatrix(), mds.getFeatures(1));
        assertEquals(d1.getLabels(), mds.getLabels(0));
    }


    private static void writeStreamToFile(File out, InputStream is) throws IOException {
        byte[] b = IOUtils.toByteArray(is);
        try(OutputStream os = new BufferedOutputStream(new FileOutputStream(out))){
            os.write(b);
        }
    }
}
