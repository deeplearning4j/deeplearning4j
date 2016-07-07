package org.datavec.spark.transform.analysis;

import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.columns.DoubleAnalysis;
import org.datavec.api.transform.analysis.columns.IntegerAnalysis;
import org.datavec.api.transform.analysis.columns.TimeAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.BaseSparkTest;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.io.data.DoubleWritable;
import org.datavec.api.io.data.IntWritable;
import org.datavec.api.io.data.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 23/06/2016.
 */
public class TestAnalysis extends BaseSparkTest {

    @Test
    public void TestAnalysisBasic(){

        Schema schema = new Schema.Builder()
                .addColumnInteger("intCol")
                .addColumnDouble("doubleCol")
                .addColumnTime("timeCol", DateTimeZone.UTC)
                .build();

        List<List<Writable>> data = new ArrayList<>();
        data.add(Arrays.asList((Writable)new IntWritable(0), new DoubleWritable(1.0), new LongWritable(1000)));
        data.add(Arrays.asList((Writable)new IntWritable(5), new DoubleWritable(0.0), new LongWritable(2000)));
        data.add(Arrays.asList((Writable)new IntWritable(3), new DoubleWritable(10.0), new LongWritable(3000)));
        data.add(Arrays.asList((Writable)new IntWritable(-1), new DoubleWritable(-1.0), new LongWritable(20000)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        DataAnalysis da = AnalyzeSpark.analyze(schema, rdd);

        List<ColumnAnalysis> ca = da.getColumnAnalysis();
        assertEquals(3, ca.size());

        assertTrue(ca.get(0) instanceof IntegerAnalysis);
        assertTrue(ca.get(1) instanceof DoubleAnalysis);
        assertTrue(ca.get(2) instanceof TimeAnalysis);

        IntegerAnalysis ia = (IntegerAnalysis)ca.get(0);
        assertEquals(-1, ia.getMin());
        assertEquals(5, ia.getMax());
        assertEquals(4, ia.getCountTotal());

        DoubleAnalysis dba = (DoubleAnalysis) ca.get(1);
        assertEquals(-1.0, dba.getMin(), 0.0);
        assertEquals(10.0, dba.getMax(), 0.0);
        assertEquals(4, dba.getCountTotal());

        TimeAnalysis ta = (TimeAnalysis)ca.get(2);
        assertEquals(1000, ta.getMin());
        assertEquals(20000, ta.getMax());
        assertEquals(4, ta.getCountTotal());

        assertNotNull(ia.getHistogramBuckets());
        assertNotNull(ia.getHistogramBucketCounts());

        assertNotNull(dba.getHistogramBuckets());
        assertNotNull(dba.getHistogramBucketCounts());

        assertNotNull(ta.getHistogramBuckets());
        assertNotNull(ta.getHistogramBucketCounts());


        double[] bucketsD = dba.getHistogramBuckets();
        long[] countD = dba.getHistogramBucketCounts();

        assertEquals(-1.0, bucketsD[0], 0.0);
        assertEquals(10.0, bucketsD[bucketsD.length-1], 0.0);
        assertEquals(1, countD[0]);
        assertEquals(1, countD[countD.length-1]);
    }

}
