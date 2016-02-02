package org.deeplearning4j.datasets.canova;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.*;

public class RecordReaderMultiDataSetIteratorTest {

    @Test
    public void testsBasic() throws Exception {

        //Load details from CSV files; single input/output -> compare to RecordReaderDataSetIterator
        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr,10,4,3);

        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

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


        //Load time series from CSV sequence files; compare to SequenceRecordReaderDataSetIterator
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
    public void testSplittingCSV() throws Exception{
        //Here's the idea: take Iris, and split it up into 2 inputs and 2 output arrays
        //Inputs: columns 0 and 1-2
        //Outputs: columns 3, and 4->OneHot

        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        RecordReaderDataSetIterator rrdsi = new RecordReaderDataSetIterator(rr,10,4,3);

        RecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

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
    public void testSplittingCSVSequence() throws Exception {
        //Idea: take CSV sequences, and split "csvsequence_i.txt" into two separate inputs; keep "csvSequencelables_i.txt"
        // as standard one-hot output

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
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

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
            rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

            MultiDataSetIterator r = new RecordReaderMultiDataSetIterator.Builder(1)
                    .addReader("iris", rr)
                    .build();
            fail("Should have thrown exception");
        }catch(Exception e){ }
    }

    @Test
    public void testVariableLengthTS() throws Exception {

        //Set up SequenceRecordReaderDataSetIterators for comparison
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

}
