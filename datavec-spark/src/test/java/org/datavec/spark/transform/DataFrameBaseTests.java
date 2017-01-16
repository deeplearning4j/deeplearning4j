package org.datavec.spark.transform;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class DataFrameBaseTests extends BaseSparkTest {

    @Test
    public void testNormalize() {
        List<List<Writable>> data = new ArrayList<>();

        data.add(Arrays.<Writable>asList(new DoubleWritable(1), new DoubleWritable(10)));
        data.add(Arrays.<Writable>asList(new DoubleWritable(2), new DoubleWritable(20)));
        data.add(Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(30)));


        List<List<Writable>> expMinMax = new ArrayList<>();
        expMinMax.add(Arrays.<Writable>asList(new DoubleWritable(0.0), new DoubleWritable(0.0)));
        expMinMax.add(Arrays.<Writable>asList(new DoubleWritable(0.5), new DoubleWritable(0.5)));
        expMinMax.add(Arrays.<Writable>asList(new DoubleWritable(1.0), new DoubleWritable(1.0)));

        double m1 = (1 + 2 + 3)/ 3.0;
        double s1 = new StandardDeviation().evaluate(new double[]{1,2,3});
        double m2 = (10 + 20 + 30) / 3.0;
        double s2 = new StandardDeviation().evaluate(new double[]{10,20,30});

        List<List<Writable>> expStandardize = new ArrayList<>();
        expStandardize.add(Arrays.<Writable>asList(new DoubleWritable((1 - m1) / s1), new DoubleWritable((10 - m2)/ s2)));
        expStandardize.add(Arrays.<Writable>asList(new DoubleWritable((2 - m1)/s1), new DoubleWritable((20 - m2)/ s2)));
        expStandardize.add(Arrays.<Writable>asList(new DoubleWritable((3 - m1)/s1), new DoubleWritable((30 - m2) / s2)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        Schema schema = new Schema.Builder()
                .addColumnInteger("c0")
                .addColumnDouble("c1")
                .build();


        JavaRDD<List<Writable>> normalized = Normalization.normalize(schema, rdd);
        JavaRDD<List<Writable>> standardize = Normalization.zeromeanUnitVariance(schema, rdd);

        Comparator<List<Writable>> comparator = new ComparatorFirstCol();

        List<List<Writable>> c = new ArrayList<>(normalized.collect());
        List<List<Writable>> c2 = new ArrayList<>(standardize.collect());
        Collections.sort(c, comparator);
        Collections.sort(c2, comparator);

//        System.out.println("Normalized:");
//        System.out.println(c);
//        System.out.println("Standardized:");
//        System.out.println(c2);

        for( int i = 0; i < expMinMax.size(); i++) {
            List<Writable> exp = expMinMax.get(i);
            List<Writable> act = c.get(i);

            for( int j = 0; j < exp.size(); j++) {
                assertEquals(exp.get(j).toDouble(), act.get(j).toDouble(), 1e-6);
            }
        }

        for( int i = 0; i < expStandardize.size(); i++) {
            List<Writable> exp = expStandardize.get(i);
            List<Writable> act = c2.get(i);

            for( int j = 0; j < exp.size(); j++ ){
                assertEquals(exp.get(j).toDouble(), act.get(j).toDouble(), 1e-6);
            }
        }
    }


    @Test
    public void testDataFrameSequenceNormalization() {
        List<List<List<Writable>>> sequences = new ArrayList<>();

        List<List<Writable>> seq1 = new ArrayList<>();
        seq1.add(Arrays.<Writable>asList(new DoubleWritable(1), new DoubleWritable(10), new DoubleWritable(100)));
        seq1.add(Arrays.<Writable>asList(new DoubleWritable(2), new DoubleWritable(20), new DoubleWritable(200)));
        seq1.add(Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(30), new DoubleWritable(300)));

        List<List<Writable>> seq2 = new ArrayList<>();
        seq2.add(Arrays.<Writable>asList(new DoubleWritable(4), new DoubleWritable(40), new DoubleWritable(400)));
        seq2.add(Arrays.<Writable>asList(new DoubleWritable(5), new DoubleWritable(50), new DoubleWritable(500)));

        sequences.add(seq1);
        sequences.add(seq2);

        Schema schema = new Schema.Builder()
                .addColumnInteger("c0")
                .addColumnDouble("c1")
                .addColumnLong("c2")
                .build();

        JavaRDD<List<List<Writable>>> rdd = sc.parallelize(sequences);

        JavaRDD<List<List<Writable>>> normalized = Normalization.normalizeSequence(schema, rdd, 0, 1);
        JavaRDD<List<List<Writable>>> standardized = Normalization.zeroMeanUnitVarianceSequence(schema, rdd);


        //Min/max normalization:
        List<List<Writable>> expSeq1MinMax = new ArrayList<>();
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((1 - 1.0) / (5.0-1.0)), new DoubleWritable((10- 10.0)/(50.0 - 10.0)), new DoubleWritable((100-100.0) / (500.0-100.0))));
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((2 - 1.0) / (5.0 - 1.0)), new DoubleWritable((20 - 10.0)/(50.0 - 10.0)), new DoubleWritable((200 - 100.0) / (500.0-100.0))));
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((3 - 1.0) / (5.0 - 1.0)), new DoubleWritable((30 - 10.0)/(50.0 - 10.0)), new DoubleWritable((300 - 100.0) / (500.0-100.0))));

        List<List<Writable>> expSeq2MinMax = new ArrayList<>();
        expSeq2MinMax.add(Arrays.<Writable>asList(new DoubleWritable((4 - 1.0)/(5.0 - 1.0)), new DoubleWritable((40 - 10.0)/(50.0 - 10.0)), new DoubleWritable((400 - 100.0) / (500.0 - 100.0))));
        expSeq2MinMax.add(Arrays.<Writable>asList(new DoubleWritable((5 - 1.0)/(5.0 - 1.0)), new DoubleWritable((50 - 10.0)/(50.0 - 10.0)), new DoubleWritable((500 - 100.0) / (500.0 - 100.0))));


        List<List<List<Writable>>> norm = new ArrayList<>(normalized.collect());
        Collections.sort(norm, new ComparatorSeqLength());
        assertEquals(2, norm.size());

//        System.out.println(norm);

        for(int i = 0; i < 2; i++) {
            List<List<Writable>> seqExp = (i == 0 ? expSeq1MinMax : expSeq2MinMax);
            for( int j = 0; j<seqExp.size(); j++ ){
                List<Writable> stepExp = seqExp.get(j);
                List<Writable> stepAct = norm.get(i).get(j);
                for( int k = 0; k<stepExp.size(); k++ ){
                    assertEquals(stepExp.get(k).toDouble(), stepAct.get(k).toDouble(), 1e-6);
                }
            }
        }





        //Standardize:
        double m1 = (1 + 2 + 3 + 4 +5) / 5.0;
        double s1 = new StandardDeviation().evaluate(new double[]{1,2,3,4,5});
        double m2 = (10 + 20 + 30 + 40 + 50) / 5.0;
        double s2 = new StandardDeviation().evaluate(new double[]{10,20,30,40,50});
        double m3 = (100 + 200 + 300 + 400 + 500) / 5.0;
        double s3 = new StandardDeviation().evaluate(new double[]{100,200,300,400,500});

        List<List<Writable>> expSeq1Std = new ArrayList<>();
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((1 - m1)/s1), new DoubleWritable((10 - m2) / s2), new DoubleWritable((100 - m3)/s3)));
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((2 - m1)/s1), new DoubleWritable((20 - m2) / s2), new DoubleWritable((200 - m3)/s3)));
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((3 - m1)/s1), new DoubleWritable((30 - m2) / s2), new DoubleWritable((300 - m3)/s3)));

        List<List<Writable>> expSeq2Std = new ArrayList<>();
        expSeq2Std.add(Arrays.<Writable>asList(new DoubleWritable((4 - m1) / s1), new DoubleWritable((40 - m2) / s2), new DoubleWritable((400 - m3) / s3)));
        expSeq2Std.add(Arrays.<Writable>asList(new DoubleWritable((5 - m1) / s1), new DoubleWritable((50 - m2) / s2), new DoubleWritable((500 - m3) / s3)));


        List<List<List<Writable>>> std = new ArrayList<>(standardized.collect());
        Collections.sort(std, new ComparatorSeqLength());
        assertEquals(2, std.size());

//        System.out.println(std);

        for( int i = 0; i < 2; i++) {
            List<List<Writable>> seqExp = (i == 0 ? expSeq1Std : expSeq2Std);
            for( int j = 0; j < seqExp.size(); j++) {
                List<Writable> stepExp = seqExp.get(j);
                List<Writable> stepAct = std.get(i).get(j);
                for( int k = 0; k<stepExp.size(); k++ ){
                    assertEquals(stepExp.get(k).toDouble(), stepAct.get(k).toDouble(), 1e-6);
                }
            }
        }
    }


    private static class ComparatorFirstCol implements Comparator<List<Writable>>{
        @Override
        public int compare(List<Writable> o1, List<Writable> o2) {
            return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
        }
    }

    private static class ComparatorSeqLength implements Comparator<List<List<Writable>>>{
        @Override
        public int compare(List<List<Writable>> o1, List<List<Writable>> o2) {
            return -Integer.compare(o1.size(), o2.size());
        }
    }
}
