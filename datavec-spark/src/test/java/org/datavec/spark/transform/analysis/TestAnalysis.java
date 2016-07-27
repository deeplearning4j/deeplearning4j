/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.transform.analysis;

import org.apache.spark.util.StatCounter;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.BaseSparkTest;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

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


    @Test
    public void testAnalysisStdev(){
        //Test stdev calculations compared to Spark's stats calculation


        Random r = new Random(12345);
        List<Double> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();
        List<Long> l3 = new ArrayList<>();

        int n = 10000;
        for( int i=0; i<n; i++ ){
            l1.add(10*r.nextDouble());
            l2.add(-1000 + r.nextInt(2000));
            l3.add(-1000L + r.nextInt(2000));
        }

        List<Double> l2d = new ArrayList<>();
        for(Integer i : l2) l2d.add(i.doubleValue());
        List<Double> l3d = new ArrayList<>();
        for(Long l : l3) l3d.add(l.doubleValue());


        StatCounter sc1 = sc.parallelizeDoubles(l1).stats();
        StatCounter sc2 = sc.parallelizeDoubles(l2d).stats();
        StatCounter sc3 = sc.parallelizeDoubles(l3d).stats();

        List<List<Writable>> data = new ArrayList<>();
        for( int i=0; i<n; i++ ){
            List<Writable> l = new ArrayList<>();
            l.add(new DoubleWritable(l1.get(i)));
            l.add(new IntWritable(l2.get(i)));
            l.add(new LongWritable(l3.get(i)));
            data.add(l);
        }

        Schema schema = new Schema.Builder()
                .addColumnDouble("d")
                .addColumnInteger("i")
                .addColumnLong("l")
                .build();

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        DataAnalysis da = AnalyzeSpark.analyze(schema, rdd);

        double stdev1 = sc1.sampleStdev();
        double stdev1a = ((DoubleAnalysis)da.getColumnAnalysis("d")).getSampleStdev();
        double re1 = Math.abs(stdev1 - stdev1a) / (Math.abs(stdev1) + Math.abs(stdev1a));
        assertTrue(re1 < 1e-6);

        double stdev2 = sc2.sampleStdev();
        double stdev2a = ((IntegerAnalysis)da.getColumnAnalysis("i")).getSampleStdev();
        double re2 = Math.abs(stdev2 - stdev2a) / (Math.abs(stdev2) + Math.abs(stdev2a));
        assertTrue(re2 < 1e-6);

        double stdev3 = sc3.sampleStdev();
        double stdev3a = ((LongAnalysis)da.getColumnAnalysis("l")).getSampleStdev();
        double re3 = Math.abs(stdev3 - stdev3a) / (Math.abs(stdev3) + Math.abs(stdev3a));
        assertTrue(re3 < 1e-6);
    }

}
