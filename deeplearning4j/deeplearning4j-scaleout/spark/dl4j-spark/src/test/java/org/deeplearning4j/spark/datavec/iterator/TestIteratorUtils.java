package org.deeplearning4j.spark.datavec.iterator;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.*;

import static org.junit.Assert.assertEquals;

public class TestIteratorUtils extends BaseSparkTest {


    @Test
    public void testIrisRRMDSI() throws Exception {

        ClassPathResource cpr = new ClassPathResource("iris.txt");
        File f = cpr.getFile();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(f));

        RecordReaderMultiDataSetIterator rrmdsi1 = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("reader", rr)
                .addInput("reader", 0, 3)
                .addOutputOneHot("reader", 4, 3)
                .build();

        RecordReaderMultiDataSetIterator rrmdsi2 = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("reader", new SparkSourceDummyReader(0))
                .addInput("reader", 0, 3)
                .addOutputOneHot("reader", 4, 3)
                .build();

        List<MultiDataSet> expected = new ArrayList<>(150);
        while(rrmdsi1.hasNext()){
            expected.add(rrmdsi1.next());
        }

        JavaRDD<List<Writable>> rdd = sc.textFile(f.getPath()).coalesce(1)
                .map(new StringToWritablesFunction(new CSVRecordReader()));

        JavaRDD<MultiDataSet> mdsRdd = IteratorUtils.mapRRMDSI(rdd, rrmdsi2);

        List<MultiDataSet> act = mdsRdd.collect();

        assertEquals(expected, act);
    }

    @Test
    public void testRRMDSIJoin() throws Exception {

        ClassPathResource cpr1 = new ClassPathResource("spark/rrmdsi/file1.txt");
        ClassPathResource cpr2 = new ClassPathResource("spark/rrmdsi/file2.txt");

        RecordReader rr1 = new CSVRecordReader();
        rr1.initialize(new FileSplit(cpr1.getFile()));
        RecordReader rr2 = new CSVRecordReader();
        rr2.initialize(new FileSplit(cpr2.getFile()));

        RecordReaderMultiDataSetIterator rrmdsi1 = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("r1", rr1)
                .addReader("r2", rr2)
                .addInput("r1", 1, 2)
                .addOutput("r2",1,2)
                .build();

        RecordReaderMultiDataSetIterator rrmdsi2 = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("r1", new SparkSourceDummyReader(0))
                .addReader("r2", new SparkSourceDummyReader(1))
                .addInput("r1", 1, 2)
                .addOutput("r2",1,2)
                .build();

        List<MultiDataSet> expected = new ArrayList<>(3);
        while(rrmdsi1.hasNext()){
            expected.add(rrmdsi1.next());
        }

        JavaRDD<List<Writable>> rdd1 = sc.textFile(cpr1.getFile().getPath()).coalesce(1)
                .map(new StringToWritablesFunction(new CSVRecordReader()));
        JavaRDD<List<Writable>> rdd2 = sc.textFile(cpr2.getFile().getPath()).coalesce(1)
                .map(new StringToWritablesFunction(new CSVRecordReader()));

        List<JavaRDD<List<Writable>>> list = Arrays.asList(rdd1, rdd2);
        JavaRDD<MultiDataSet> mdsRdd = IteratorUtils.mapRRMDSI(list, null, new int[]{0,0}, null, false, rrmdsi2);

        List<MultiDataSet> act = mdsRdd.collect();


        expected = new ArrayList<>(expected);
        act = new ArrayList<>(act);
        Comparator<MultiDataSet> comp = new Comparator<MultiDataSet>() {
            @Override
            public int compare(MultiDataSet d1, MultiDataSet d2) {
                return Double.compare(d1.getFeatures(0).getDouble(0), d2.getFeatures(0).getDouble(0));
            }
        };

        Collections.sort(expected, comp);
        Collections.sort(act, comp);

        assertEquals(expected, act);
    }

}
