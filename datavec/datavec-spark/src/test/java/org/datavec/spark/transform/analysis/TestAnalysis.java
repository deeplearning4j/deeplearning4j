/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform.analysis;

import com.tdunning.math.stats.TDigest;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.util.StatCounter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.*;
import org.datavec.local.transforms.AnalyzeLocal;
import org.datavec.spark.BaseSparkTest;
import org.datavec.spark.transform.AnalyzeSpark;
import org.joda.time.DateTimeZone;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by Alex on 23/06/2016.
 */
public class TestAnalysis extends BaseSparkTest {

    @Test
    public void testAnalysis() throws Exception {

        Schema schema = new Schema.Builder().addColumnInteger("intCol").addColumnDouble("doubleCol")
                        .addColumnTime("timeCol", DateTimeZone.UTC).addColumnCategorical("catCol", "A", "B")
                        .addColumnNDArray("ndarray", new long[] {1, 10}).build();

        List<List<Writable>> data = new ArrayList<>();
        data.add(Arrays.asList((Writable) new IntWritable(0), new DoubleWritable(1.0), new LongWritable(1000),
                        new Text("A"), new NDArrayWritable(Nd4j.valueArrayOf(10, 100.0))));
        data.add(Arrays.asList((Writable) new IntWritable(5), new DoubleWritable(0.0), new LongWritable(2000),
                        new Text("A"), new NDArrayWritable(Nd4j.valueArrayOf(10, 200.0))));
        data.add(Arrays.asList((Writable) new IntWritable(3), new DoubleWritable(10.0), new LongWritable(3000),
                        new Text("A"), new NDArrayWritable(Nd4j.valueArrayOf(10, 300.0))));
        data.add(Arrays.asList((Writable) new IntWritable(-1), new DoubleWritable(-1.0), new LongWritable(20000),
                        new Text("B"), new NDArrayWritable(Nd4j.valueArrayOf(10, 400.0))));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        DataAnalysis da = AnalyzeSpark.analyze(schema, rdd);
        String daString = da.toString();

        System.out.println(da);

        List<ColumnAnalysis> ca = da.getColumnAnalysis();
        assertEquals(5, ca.size());

        assertTrue(ca.get(0) instanceof IntegerAnalysis);
        assertTrue(ca.get(1) instanceof DoubleAnalysis);
        assertTrue(ca.get(2) instanceof TimeAnalysis);
        assertTrue(ca.get(3) instanceof CategoricalAnalysis);
        assertTrue(ca.get(4) instanceof NDArrayAnalysis);

        IntegerAnalysis ia = (IntegerAnalysis) ca.get(0);
        assertEquals(-1, ia.getMin());
        assertEquals(5, ia.getMax());
        assertEquals(4, ia.getCountTotal());
        TDigest itd = ia.getDigest();
        assertEquals(-0.5, itd.quantile(0.25), 1e-9); // right-biased linear approximations w/ few points
        assertEquals(1.5, itd.quantile(0.5), 1e-9);
        assertEquals(4.0, itd.quantile(0.75), 1e-9);
        assertEquals(5.0, itd.quantile(1), 1e-9);

        DoubleAnalysis dba = (DoubleAnalysis) ca.get(1);
        assertEquals(-1.0, dba.getMin(), 0.0);
        assertEquals(10.0, dba.getMax(), 0.0);
        assertEquals(4, dba.getCountTotal());
        TDigest dtd = dba.getDigest();
        assertEquals(-0.5, dtd.quantile(0.25), 1e-9); // right-biased linear approximations w/ few points
        assertEquals(0.5, dtd.quantile(0.5), 1e-9);
        assertEquals(5.5, dtd.quantile(0.75), 1e-9);
        assertEquals(10.0, dtd.quantile(1), 1e-9);


        TimeAnalysis ta = (TimeAnalysis) ca.get(2);
        assertEquals(1000, ta.getMin());
        assertEquals(20000, ta.getMax());
        assertEquals(4, ta.getCountTotal());
        TDigest ttd = ta.getDigest();
        assertEquals(1500.0, ttd.quantile(0.25), 1e-9); // right-biased linear approximations w/ few points
        assertEquals(2500.0, ttd.quantile(0.5), 1e-9);
        assertEquals(11500.0, ttd.quantile(0.75), 1e-9);
        assertEquals(20000.0, ttd.quantile(1), 1e-9);

        CategoricalAnalysis cata = (CategoricalAnalysis) ca.get(3);
        Map<String, Long> map = cata.getMapOfCounts();
        assertEquals(2, map.keySet().size());
        assertEquals(3L, (long) map.get("A"));
        assertEquals(1L, (long) map.get("B"));

        NDArrayAnalysis na = (NDArrayAnalysis) ca.get(4);
        assertEquals(4, na.getCountTotal());
        assertEquals(0, na.getCountNull());
        assertEquals(10, na.getMinLength());
        assertEquals(10, na.getMaxLength());
        assertEquals(4 * 10, na.getTotalNDArrayValues());
        assertEquals(Collections.singletonMap(2, 4L), na.getCountsByRank());
        assertEquals(100.0, na.getMinValue(), 0.0);
        assertEquals(400.0, na.getMaxValue(), 0.0);

        assertNotNull(ia.getHistogramBuckets());
        assertNotNull(ia.getHistogramBucketCounts());

        assertNotNull(dba.getHistogramBuckets());
        assertNotNull(dba.getHistogramBucketCounts());

        assertNotNull(ta.getHistogramBuckets());
        assertNotNull(ta.getHistogramBucketCounts());

        assertNotNull(na.getHistogramBuckets());
        assertNotNull(na.getHistogramBucketCounts());

        double[] bucketsD = dba.getHistogramBuckets();
        long[] countD = dba.getHistogramBucketCounts();

        assertEquals(-1.0, bucketsD[0], 0.0);
        assertEquals(10.0, bucketsD[bucketsD.length - 1], 0.0);
        assertEquals(1, countD[0]);
        assertEquals(1, countD[countD.length - 1]);

        File f = Files.createTempFile("datavec_spark_analysis_UITest", ".html").toFile();
        System.out.println(f.getAbsolutePath());
        f.deleteOnExit();
        HtmlAnalysis.createHtmlAnalysisFile(da, f);
    }


    @Test
    public void testAnalysisStdev() {
        //Test stdev calculations compared to Spark's stats calculation


        Random r = new Random(12345);
        List<Double> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();
        List<Long> l3 = new ArrayList<>();

        int n = 10000;
        for (int i = 0; i < n; i++) {
            l1.add(10 * r.nextDouble());
            l2.add(-1000 + r.nextInt(2000));
            l3.add(-1000L + r.nextInt(2000));
        }

        List<Double> l2d = new ArrayList<>();
        for (Integer i : l2)
            l2d.add(i.doubleValue());
        List<Double> l3d = new ArrayList<>();
        for (Long l : l3)
            l3d.add(l.doubleValue());


        StatCounter sc1 = sc.parallelizeDoubles(l1).stats();
        StatCounter sc2 = sc.parallelizeDoubles(l2d).stats();
        StatCounter sc3 = sc.parallelizeDoubles(l3d).stats();

        org.datavec.api.transform.analysis.counter.StatCounter sc1new = new org.datavec.api.transform.analysis.counter.StatCounter();
        for(double d : l1){
            sc1new.add(d);
        }

        assertEquals(sc1.sampleStdev(), sc1new.getStddev(false), 1e-6);


        List<StatCounter> sparkCounters = new ArrayList<>();
        List<org.datavec.api.transform.analysis.counter.StatCounter> counters = new ArrayList<>();
        for( int i=0; i<10; i++ ){
            counters.add(new org.datavec.api.transform.analysis.counter.StatCounter());
            sparkCounters.add(new StatCounter());
        }
        for( int i=0; i<l1.size(); i++){
            int idx = i % 10;
            counters.get(idx).add(l1.get(i));
            sparkCounters.get(idx).merge(l1.get(i));
        }
        org.datavec.api.transform.analysis.counter.StatCounter counter = counters.get(0);
        StatCounter sparkCounter = sparkCounters.get(0);
        for( int i=1; i<10; i++ ){
            counter.merge(counters.get(i));
            sparkCounter.merge(sparkCounters.get(i));
            System.out.println();
        }
        assertEquals(sc1.sampleStdev(), counter.getStddev(false), 1e-6);
        assertEquals(sparkCounter.sampleStdev(), counter.getStddev(false), 1e-6);

        List<List<Writable>> data = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Writable> l = new ArrayList<>();
            l.add(new DoubleWritable(l1.get(i)));
            l.add(new IntWritable(l2.get(i)));
            l.add(new LongWritable(l3.get(i)));
            data.add(l);
        }

        Schema schema = new Schema.Builder().addColumnDouble("d").addColumnInteger("i").addColumnLong("l").build();

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        DataAnalysis da = AnalyzeSpark.analyze(schema, rdd);

        double stdev1 = sc1.sampleStdev();
        double stdev1a = ((DoubleAnalysis) da.getColumnAnalysis("d")).getSampleStdev();
        double re1 = Math.abs(stdev1 - stdev1a) / (Math.abs(stdev1) + Math.abs(stdev1a));
        assertTrue(re1 < 1e-6);

        double stdev2 = sc2.sampleStdev();
        double stdev2a = ((IntegerAnalysis) da.getColumnAnalysis("i")).getSampleStdev();
        double re2 = Math.abs(stdev2 - stdev2a) / (Math.abs(stdev2) + Math.abs(stdev2a));
        assertTrue(re2 < 1e-6);

        double stdev3 = sc3.sampleStdev();
        double stdev3a = ((LongAnalysis) da.getColumnAnalysis("l")).getSampleStdev();
        double re3 = Math.abs(stdev3 - stdev3a) / (Math.abs(stdev3) + Math.abs(stdev3a));
        assertTrue(re3 < 1e-6);
    }


    @Test
    public void testSampleMostFrequent() {

        List<List<Writable>> toParallelize = new ArrayList<>();
        toParallelize.add(Arrays.<Writable>asList(new Text("a"), new Text("MostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("b"), new Text("SecondMostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("c"), new Text("SecondMostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("d"), new Text("0")));
        toParallelize.add(Arrays.<Writable>asList(new Text("e"), new Text("MostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("f"), new Text("ThirdMostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("c"), new Text("MostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("h"), new Text("1")));
        toParallelize.add(Arrays.<Writable>asList(new Text("i"), new Text("SecondMostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("j"), new Text("2")));
        toParallelize.add(Arrays.<Writable>asList(new Text("k"), new Text("ThirdMostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("l"), new Text("MostCommon")));
        toParallelize.add(Arrays.<Writable>asList(new Text("m"), new Text("3")));
        toParallelize.add(Arrays.<Writable>asList(new Text("n"), new Text("4")));
        toParallelize.add(Arrays.<Writable>asList(new Text("o"), new Text("5")));


        JavaRDD<List<Writable>> rdd = sc.parallelize(toParallelize);

        Schema schema = new Schema.Builder().addColumnsString("irrelevant", "column").build();

        Map<Writable, Long> map = AnalyzeSpark.sampleMostFrequentFromColumn(3, "column", schema, rdd);

        //        System.out.println(map);

        assertEquals(3, map.size());
        assertEquals(4L, (long) map.get(new Text("MostCommon")));
        assertEquals(3L, (long) map.get(new Text("SecondMostCommon")));
        assertEquals(2L, (long) map.get(new Text("ThirdMostCommon")));
    }


    @Test
    public void testAnalysisVsLocal() throws Exception {

        Schema s = new Schema.Builder()
                .addColumnsDouble("%d", 0, 3)
                .addColumnInteger("label")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        List<List<Writable>> toParallelize = new ArrayList<>();
        while(rr.hasNext()){
            toParallelize.add(rr.next());
        }

        JavaRDD<List<Writable>> rdd = sc.parallelize(toParallelize).coalesce(1);


        rr.reset();
        DataAnalysis local = AnalyzeLocal.analyze(s, rr);
        DataAnalysis spark = AnalyzeSpark.analyze(s, rdd);

//        assertEquals(local.toJson(), spark.toJson());
        assertEquals(local, spark);


        //Also quality analysis:
        rr.reset();
        DataQualityAnalysis localQ = AnalyzeLocal.analyzeQuality(s, rr);
        DataQualityAnalysis sparkQ = AnalyzeSpark.analyzeQuality(s, rdd);

        assertEquals(localQ, sparkQ);


        //And, check unique etc:
        rr.reset();
        Map<String,Set<Writable>> mapLocal = AnalyzeLocal.getUnique(s.getColumnNames(), s, rr);
        Map<String,List<Writable>> mapSpark = AnalyzeSpark.getUnique(s.getColumnNames(), s, rdd);

        assertEquals(mapLocal.keySet(), mapSpark.keySet());
        for( String k : mapLocal.keySet()){
            assertEquals(mapLocal.get(k), new HashSet<Writable>(mapSpark.get(k)));
        }
    }

}
