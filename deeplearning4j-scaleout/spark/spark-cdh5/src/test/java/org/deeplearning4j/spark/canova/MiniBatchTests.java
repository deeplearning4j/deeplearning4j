package org.deeplearning4j.spark.canova;

import static org.junit.Assert.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

/**
 * Handle dividing things up by mini batch
 */
public class MiniBatchTests {
    private static Logger log = LoggerFactory.getLogger(MiniBatchTests.class);

    @Test
    public void testMiniBatches() throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local")
                .setAppName("SparkDebugExample");

        System.out.println("Setting up Spark Context...");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile(new ClassPathResource("data/svmLight/iris_svmLight_0.txt").getFile().toURI().toString()).cache();
        long count = lines.count();
        assertEquals(150,count);
        // gotta map this to a Matrix/INDArray
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(new SVMLightRecordReader(), 0, 3)).cache();
        count = points.count();
        assertEquals(150,count);

        JavaRDD<DataSet> miniBatches = new RDDMiniBatches(10,points).miniBatchesJava();
        count = miniBatches.count();
        assertEquals(15,count);

    }

}
