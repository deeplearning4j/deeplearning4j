package org.deeplearning4j.spark.canova;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.functions.SequenceRecordReaderFunction;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

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

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(irr,1,28*28,2);
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


}
