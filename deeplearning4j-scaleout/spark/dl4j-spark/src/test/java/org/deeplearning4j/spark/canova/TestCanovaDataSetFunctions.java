package org.deeplearning4j.spark.canova;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.functions.SequenceRecordReaderFunction;
import org.canova.spark.functions.pairdata.*;
import org.canova.spark.util.CanovaSparkUtil;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

public class TestCanovaDataSetFunctions extends BaseSparkTest {

    @Test
    public void testCanovaDataSetFunction() throws Exception {
        JavaSparkContext sc = getContext();
        //Test Spark record reader functionality vs. local

        ClassPathResource cpr = new ClassPathResource("/imagetest/0/a.bmp");
        List<String> labelsList = Arrays.asList("0", "1");   //Need this for Spark: can't infer without init call

        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 7);
        path = folder + "*";

        JavaPairRDD<String,PortableDataStream> origData = sc.binaryFiles(path);
        assertEquals(4,origData.count());    //4 images

        RecordReader rr = new ImageRecordReader(28,28,1,true,labelsList);
        org.canova.spark.functions.RecordReaderFunction rrf = new org.canova.spark.functions.RecordReaderFunction(rr);
        JavaRDD<Collection<Writable>> rdd = origData.map(rrf);
        JavaRDD<DataSet> data = rdd.map(new CanovaDataSetFunction(28*28,2,false));
        List<DataSet> collected = data.collect();

        //Load normally (i.e., not via Spark), and check that we get the same results (order not withstanding)
        InputSplit is = new FileSplit(new File(folder),new String[]{"bmp"}, true);
        ImageRecordReader irr = new ImageRecordReader(28,28,1,true);
        irr.initialize(is);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(irr,1,1,2);
        List<DataSet> listLocal = new ArrayList<>(4);
        while(iter.hasNext()){
            listLocal.add(iter.next());
        }


        //Compare:
        assertEquals(4,collected.size());
        assertEquals(4,listLocal.size());

        //Check that results are the same (order not withstanding)
        boolean[] found = new boolean[4];
        for (int i = 0; i < 4; i++) {
            int foundIndex = -1;
            DataSet ds = collected.get(i);
            for (int j = 0; j < 4; j++) {
                if (ds.equals(listLocal.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(4, count);  //Expect all 4 and exactly 4 pairwise matches between spark and local versions
    }

    @Test
    public void testCanovaSequenceDataSetFunction() throws Exception {
        JavaSparkContext sc = getContext();
        //Test Spark record reader functionality vs. local

        ClassPathResource cpr = new ClassPathResource("/csvsequence/csvsequence_0.txt");
        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 17);
        path = folder + "*";

        JavaPairRDD<String,PortableDataStream> origData = sc.binaryFiles(path);
        assertEquals(3, origData.count());    //3 CSV sequences


        SequenceRecordReader seqRR = new CSVSequenceRecordReader(1,",");
        SequenceRecordReaderFunction rrf = new SequenceRecordReaderFunction(seqRR);
        JavaRDD<Collection<Collection<Writable>>> rdd = origData.map(rrf);
        JavaRDD<DataSet> data = rdd.map(new CanovaSequenceDataSetFunction(2, -1, true, null, null));
        List<DataSet> collected = data.collect();

        //Load normally (i.e., not via Spark), and check that we get the same results (order not withstanding)
        InputSplit is = new FileSplit(new File(folder),new String[]{"txt"}, true);
        SequenceRecordReader seqRR2 = new CSVSequenceRecordReader(1,",");
        seqRR2.initialize(is);

        SequenceRecordReaderDataSetIterator iter = new SequenceRecordReaderDataSetIterator(seqRR2,1,-1,2,true);
        List<DataSet> listLocal = new ArrayList<>(3);
        while(iter.hasNext()){
            listLocal.add(iter.next());
        }


        //Compare:
        assertEquals(3,collected.size());
        assertEquals(3,listLocal.size());

        //Check that results are the same (order not withstanding)
        boolean[] found = new boolean[3];
        for (int i = 0; i < 3; i++) {
            int foundIndex = -1;
            DataSet ds = collected.get(i);
            for (int j = 0; j < 3; j++) {
                if (ds.equals(listLocal.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(3, count);  //Expect all 3 and exactly 3 pairwise matches between spark and local versions
    }

    @Test
    public void testCanovaSequencePairDataSetFunction() throws Exception {
        JavaSparkContext sc = getContext();

        //Convert data to a SequenceFile:
        ClassPathResource cpr = new ClassPathResource("/csvsequence/csvsequence_0.txt");
        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 17);
        path = folder + "*";

        PathToKeyConverter pathConverter = new PathToKeyConverterFilename();
        JavaPairRDD<Text,BytesPairWritable> toWrite = CanovaSparkUtil.combineFilesForSequenceFile(sc, path, path, pathConverter);

        Path p = Files.createTempDirectory("dl4j_testSeqPairFn");
        p.toFile().deleteOnExit();
        String outPath = p.toString() + "/out";
        new File(outPath).deleteOnExit();
        toWrite.saveAsNewAPIHadoopFile(outPath, Text.class, BytesPairWritable.class, SequenceFileOutputFormat.class);

        //Load from sequence file:
        JavaPairRDD<Text,BytesPairWritable> fromSeq = sc.sequenceFile(outPath, Text.class, BytesPairWritable.class);

        SequenceRecordReader srr1 = new CSVSequenceRecordReader(1,",");
        SequenceRecordReader srr2 = new CSVSequenceRecordReader(1,",");
        PairSequenceRecordReaderBytesFunction psrbf = new PairSequenceRecordReaderBytesFunction(srr1,srr2);
        JavaRDD<Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>>> writables = fromSeq.map(psrbf);

            //Map to DataSet:
        CanovaSequencePairDataSetFunction pairFn = new CanovaSequencePairDataSetFunction();
        JavaRDD<DataSet> data = writables.map(pairFn);
        List<DataSet> sparkData = data.collect();


        //Now: do the same thing locally (SequenceRecordReaderDataSetIterator) and compare
        ClassPathResource resource = new ClassPathResource("csvsequence_0.txt");
        String featuresPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, -1, true);

        List<DataSet> localData = new ArrayList<>(3);
        while(iter.hasNext()) localData.add(iter.next());

        assertEquals(3,sparkData.size());
        assertEquals(3,localData.size());

        for( int i=0; i<3; i++ ){
            //Check shapes etc. data sets order may differ for spark vs. local
            DataSet dsSpark = sparkData.get(i);
            DataSet dsLocal = localData.get(i);

            assertNull(dsSpark.getFeaturesMaskArray());
            assertNull(dsSpark.getLabelsMaskArray());

            INDArray fSpark = dsSpark.getFeatureMatrix();
            INDArray fLocal = dsLocal.getFeatureMatrix();
            INDArray lSpark = dsSpark.getLabels();
            INDArray lLocal = dsLocal.getLabels();

            int[] s = new int[]{1,3,4}; //1 example, 3 values, 3 time steps
            assertArrayEquals(s,fSpark.shape());
            assertArrayEquals(s,fLocal.shape());
            assertArrayEquals(s,lSpark.shape());
            assertArrayEquals(s,lLocal.shape());
        }


        //Check that results are the same (order not withstanding)
        boolean[] found = new boolean[3];
        for (int i = 0; i < 3; i++) {
            int foundIndex = -1;
            DataSet ds = sparkData.get(i);
            for (int j = 0; j < 3; j++) {
                if (ds.equals(localData.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(3, count);  //Expect all 3 and exactly 3 pairwise matches between spark and local versions
    }

    @Test
    public void testCanovaSequencePairDataSetFunctionVariableLength() throws Exception {
        //Same sort of test as testCanovaSequencePairDataSetFunction() but with variable length time series (labels shorter, align end)

        //Convert data to a SequenceFile:
        ClassPathResource cpr = new ClassPathResource("/csvsequence/csvsequence_0.txt");
        String pathFeatures = cpr.getFile().getAbsolutePath();
        String folderFeatures = pathFeatures.substring(0, pathFeatures.length() - 17);
        pathFeatures = folderFeatures + "*";

        cpr = new ClassPathResource("/csvsequencelabels/csvsequencelabelsShort_0.txt");
        String pathLabels = cpr.getFile().getAbsolutePath();
        String folderLabels = pathLabels.substring(0, pathLabels.length() - 28);
        pathLabels = folderLabels + "*";


        PathToKeyConverter pathConverter = new PathToKeyConverterNumber();  //Extract a number from the file name
        JavaPairRDD<Text,BytesPairWritable> toWrite = CanovaSparkUtil.combineFilesForSequenceFile(sc, pathFeatures, pathLabels, pathConverter);

        Path p = Files.createTempDirectory("dl4j_testSeqPairFnVarLength");
        p.toFile().deleteOnExit();
        String outPath = p.toString() + "/out";
        new File(outPath).deleteOnExit();
        toWrite.saveAsNewAPIHadoopFile(outPath, Text.class, BytesPairWritable.class, SequenceFileOutputFormat.class);

        //Load from sequence file:
        JavaPairRDD<Text,BytesPairWritable> fromSeq = sc.sequenceFile(outPath, Text.class, BytesPairWritable.class);

        SequenceRecordReader srr1 = new CSVSequenceRecordReader(1,",");
        SequenceRecordReader srr2 = new CSVSequenceRecordReader(1,",");
        PairSequenceRecordReaderBytesFunction psrbf = new PairSequenceRecordReaderBytesFunction(srr1,srr2);
        JavaRDD<Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>>> writables = fromSeq.map(psrbf);

        //Map to DataSet:
        CanovaSequencePairDataSetFunction pairFn = new CanovaSequencePairDataSetFunction(4,false, CanovaSequencePairDataSetFunction.AlignmentMode.ALIGN_END);
        JavaRDD<DataSet> data = writables.map(pairFn);
        List<DataSet> sparkData = data.collect();


        //Now: do the same thing locally (SequenceRecordReaderDataSetIterator) and compare
        ClassPathResource resource = new ClassPathResource("/csvsequence/csvsequence_0.txt");
        String featuresPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");
        resource = new ClassPathResource("/csvsequencelabels/csvsequencelabelsShort_0.txt");
        String labelsPath = resource.getFile().getAbsolutePath().replaceAll("0", "%d");

        SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));

        SequenceRecordReaderDataSetIterator iter =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        List<DataSet> localData = new ArrayList<>(3);
        while(iter.hasNext()) localData.add(iter.next());

        assertEquals(3,sparkData.size());
        assertEquals(3,localData.size());

        int[] fShapeExp = new int[]{1,3,4}; //1 example, 3 values, 4 time steps
        int[] lShapeExp = new int[]{1,4,4}; //1 example, 4 values/classes, 4 time steps (after padding)
        for( int i=0; i<3; i++ ){
            //Check shapes etc. data sets order may differ for spark vs. local
            DataSet dsSpark = sparkData.get(i);
            DataSet dsLocal = localData.get(i);

            assertNull(dsSpark.getFeaturesMaskArray());     //In this particular test: featuresLength > labelLength therefore no mast array needed for features
            assertNotNull(dsSpark.getLabelsMaskArray());    //Expect mask array for labels

            INDArray fSpark = dsSpark.getFeatureMatrix();
            INDArray fLocal = dsLocal.getFeatureMatrix();
            INDArray lSpark = dsSpark.getLabels();
            INDArray lLocal = dsLocal.getLabels();


            assertArrayEquals(fShapeExp,fSpark.shape());
            assertArrayEquals(fShapeExp,fLocal.shape());
            assertArrayEquals(lShapeExp,lSpark.shape());
            assertArrayEquals(lShapeExp,lLocal.shape());
        }


        //Check that results are the same (order not withstanding)
        boolean[] found = new boolean[3];
        for (int i = 0; i < 3; i++) {
            int foundIndex = -1;
            DataSet ds = sparkData.get(i);
            for (int j = 0; j < 3; j++) {
                if (ds.equals(localData.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(3, count);  //Expect all 3 and exactly 3 pairwise matches between spark and local versions


        //-------------------------------------------------
        //NOW: test same thing, but for align start...
        CanovaSequencePairDataSetFunction pairFnAlignStart = new CanovaSequencePairDataSetFunction(4,false, CanovaSequencePairDataSetFunction.AlignmentMode.ALIGN_START);
        JavaRDD<DataSet> rddDataAlignStart = writables.map(pairFnAlignStart);
        List<DataSet> sparkDataAlignStart = rddDataAlignStart.collect();

        featureReader.initialize(new NumberedFileInputSplit(featuresPath, 0, 2));   //re-initialize to reset
        labelReader.initialize(new NumberedFileInputSplit(labelsPath, 0, 2));
        SequenceRecordReaderDataSetIterator iterAlignStart =
                new SequenceRecordReaderDataSetIterator(featureReader, labelReader, 1, 4, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);

        List<DataSet> localDataAlignStart = new ArrayList<>(3);
        while(iterAlignStart.hasNext()) localDataAlignStart.add(iterAlignStart.next());

        assertEquals(3,sparkDataAlignStart.size());
        assertEquals(3,localDataAlignStart.size());

        for( int i=0; i<3; i++ ){
            //Check shapes etc. data sets order may differ for spark vs. local
            DataSet dsSpark = sparkDataAlignStart.get(i);
            DataSet dsLocal = localDataAlignStart.get(i);

            assertNull(dsSpark.getFeaturesMaskArray());     //In this particular test: featuresLength > labelLength therefore no mast array needed for features
            assertNotNull(dsSpark.getLabelsMaskArray());    //Expect mask array for labels

            INDArray fSpark = dsSpark.getFeatureMatrix();
            INDArray fLocal = dsLocal.getFeatureMatrix();
            INDArray lSpark = dsSpark.getLabels();
            INDArray lLocal = dsLocal.getLabels();


            assertArrayEquals(fShapeExp,fSpark.shape());
            assertArrayEquals(fShapeExp,fLocal.shape());
            assertArrayEquals(lShapeExp,lSpark.shape());
            assertArrayEquals(lShapeExp,lLocal.shape());
        }


        //Check that results are the same (order not withstanding)
        found = new boolean[3];
        for (int i = 0; i < 3; i++) {
            int foundIndex = -1;
            DataSet ds = sparkData.get(i);
            for (int j = 0; j < 3; j++) {
                if (ds.equals(localData.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(3, count);  //Expect all 3 and exactly 3 pairwise matches between spark and local versions
    }


}
