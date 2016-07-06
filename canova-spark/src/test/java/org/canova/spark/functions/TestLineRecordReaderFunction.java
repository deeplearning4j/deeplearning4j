package org.canova.spark.functions;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.spark.BaseSparkTest;
import org.junit.Test;

import java.io.File;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 21/05/2016.
 */
public class TestLineRecordReaderFunction extends BaseSparkTest {

    @Test
    public void testLineRecordReader() throws Exception {

        File dataFile = new ClassPathResource("iris.dat").getFile();
        List<String> lines = FileUtils.readLines(dataFile);

        JavaSparkContext sc = getContext();
        JavaRDD<String> linesRdd = sc.parallelize(lines);

        CSVRecordReader rr = new CSVRecordReader(0,",");

        JavaRDD<Collection<Writable>> out = linesRdd.map(new LineRecordReaderFunction(rr));
        List<Collection<Writable>> outList = out.collect();


        CSVRecordReader rr2 = new CSVRecordReader(0,",");
        rr2.initialize(new FileSplit(dataFile));
        Set<Collection<Writable>> expectedSet = new HashSet<>();
        int totalCount = 0;
        while(rr2.hasNext()){
            expectedSet.add(rr2.next());
            totalCount++;
        }

        assertEquals(totalCount, outList.size());

        for(Collection<Writable> line : outList){
            assertTrue(expectedSet.contains(line));
        }
    }
}
