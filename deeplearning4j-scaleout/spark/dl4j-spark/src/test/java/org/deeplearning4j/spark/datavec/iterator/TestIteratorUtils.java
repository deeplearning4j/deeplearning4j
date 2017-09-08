package org.deeplearning4j.spark.datavec.iterator;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

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

}
