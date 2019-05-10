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

package org.datavec.spark.transform;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class DataFramesTests extends BaseSparkTest {

    @Test
    public void testMinMax() {
        INDArray arr = Nd4j.linspace(1, 10, 10).broadcast(10, 10);
        for (int i = 0; i < arr.rows(); i++)
            arr.getRow(i).addi(i);

        List<List<Writable>> records = RecordConverter.toRecords(arr);
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 10;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));
        Schema schema = builder.build();
        DataRowsFacade dataFrame = DataFrames.toDataFrame(schema, sc.parallelize(records));
        dataFrame.get().show();
        dataFrame.get().describe(DataFrames.toArray(schema.getColumnNames())).show();
        //        System.out.println(Normalization.minMaxColumns(dataFrame,schema.getColumnNames()));
        //        System.out.println(Normalization.stdDevMeanColumns(dataFrame,schema.getColumnNames()));

    }


    @Test
    public void testDataFrameConversions() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for (int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for (int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema, DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(), DataFrames.toRecords(DataFrames.toDataFrame(schema, rdd)).getSecond().collect());

        DataRowsFacade dataFrame = DataFrames.toDataFrame(schema, rdd);
        dataFrame.get().show();
        Column mean = DataFrames.mean(dataFrame, "0");
        Column std = DataFrames.std(dataFrame, "0");
        dataFrame.get().withColumn("0", dataFrame.get().col("0").minus(mean)).show();
        dataFrame.get().withColumn("0", dataFrame.get().col("0").divide(std)).show();

        /*   DataFrame desc = dataFrame.describe(dataFrame.columns());
        dataFrame.show();
        System.out.println(dataFrame.agg(avg("0"), dataFrame.col("0")));
        dataFrame.withColumn("0",dataFrame.col("0").minus(avg(dataFrame.col("0"))));
        dataFrame.show();
        
        
        for(String column : dataFrame.columns()) {
            System.out.println(DataFrames.mean(desc,column));
            System.out.println(DataFrames.min(desc,column));
            System.out.println(DataFrames.max(desc,column));
            System.out.println(DataFrames.std(desc,column));
        
        }*/
    }

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

        double m1 = (1 + 2 + 3) / 3.0;
        double s1 = new StandardDeviation().evaluate(new double[] {1, 2, 3});
        double m2 = (10 + 20 + 30) / 3.0;
        double s2 = new StandardDeviation().evaluate(new double[] {10, 20, 30});

        List<List<Writable>> expStandardize = new ArrayList<>();
        expStandardize.add(
                        Arrays.<Writable>asList(new DoubleWritable((1 - m1) / s1), new DoubleWritable((10 - m2) / s2)));
        expStandardize.add(
                        Arrays.<Writable>asList(new DoubleWritable((2 - m1) / s1), new DoubleWritable((20 - m2) / s2)));
        expStandardize.add(
                        Arrays.<Writable>asList(new DoubleWritable((3 - m1) / s1), new DoubleWritable((30 - m2) / s2)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        Schema schema = new Schema.Builder().addColumnInteger("c0").addColumnDouble("c1").build();


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

        for (int i = 0; i < expMinMax.size(); i++) {
            List<Writable> exp = expMinMax.get(i);
            List<Writable> act = c.get(i);

            for (int j = 0; j < exp.size(); j++) {
                assertEquals(exp.get(j).toDouble(), act.get(j).toDouble(), 1e-6);
            }
        }

        for (int i = 0; i < expStandardize.size(); i++) {
            List<Writable> exp = expStandardize.get(i);
            List<Writable> act = c2.get(i);

            for (int j = 0; j < exp.size(); j++) {
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

        Schema schema = new Schema.Builder().addColumnInteger("c0").addColumnDouble("c1").addColumnLong("c2").build();

        JavaRDD<List<List<Writable>>> rdd = sc.parallelize(sequences);

        JavaRDD<List<List<Writable>>> normalized = Normalization.normalizeSequence(schema, rdd, 0, 1);
        JavaRDD<List<List<Writable>>> standardized = Normalization.zeroMeanUnitVarianceSequence(schema, rdd);


        //Min/max normalization:
        List<List<Writable>> expSeq1MinMax = new ArrayList<>();
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((1 - 1.0) / (5.0 - 1.0)),
                        new DoubleWritable((10 - 10.0) / (50.0 - 10.0)),
                        new DoubleWritable((100 - 100.0) / (500.0 - 100.0))));
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((2 - 1.0) / (5.0 - 1.0)),
                        new DoubleWritable((20 - 10.0) / (50.0 - 10.0)),
                        new DoubleWritable((200 - 100.0) / (500.0 - 100.0))));
        expSeq1MinMax.add(Arrays.<Writable>asList(new DoubleWritable((3 - 1.0) / (5.0 - 1.0)),
                        new DoubleWritable((30 - 10.0) / (50.0 - 10.0)),
                        new DoubleWritable((300 - 100.0) / (500.0 - 100.0))));

        List<List<Writable>> expSeq2MinMax = new ArrayList<>();
        expSeq2MinMax.add(Arrays.<Writable>asList(new DoubleWritable((4 - 1.0) / (5.0 - 1.0)),
                        new DoubleWritable((40 - 10.0) / (50.0 - 10.0)),
                        new DoubleWritable((400 - 100.0) / (500.0 - 100.0))));
        expSeq2MinMax.add(Arrays.<Writable>asList(new DoubleWritable((5 - 1.0) / (5.0 - 1.0)),
                        new DoubleWritable((50 - 10.0) / (50.0 - 10.0)),
                        new DoubleWritable((500 - 100.0) / (500.0 - 100.0))));


        List<List<List<Writable>>> norm = new ArrayList<>(normalized.collect());
        Collections.sort(norm, new ComparatorSeqLength());
        assertEquals(2, norm.size());

        //        System.out.println(norm);

        for (int i = 0; i < 2; i++) {
            List<List<Writable>> seqExp = (i == 0 ? expSeq1MinMax : expSeq2MinMax);
            for (int j = 0; j < seqExp.size(); j++) {
                List<Writable> stepExp = seqExp.get(j);
                List<Writable> stepAct = norm.get(i).get(j);
                for (int k = 0; k < stepExp.size(); k++) {
                    assertEquals(stepExp.get(k).toDouble(), stepAct.get(k).toDouble(), 1e-6);
                }
            }
        }



        //Standardize:
        double m1 = (1 + 2 + 3 + 4 + 5) / 5.0;
        double s1 = new StandardDeviation().evaluate(new double[] {1, 2, 3, 4, 5});
        double m2 = (10 + 20 + 30 + 40 + 50) / 5.0;
        double s2 = new StandardDeviation().evaluate(new double[] {10, 20, 30, 40, 50});
        double m3 = (100 + 200 + 300 + 400 + 500) / 5.0;
        double s3 = new StandardDeviation().evaluate(new double[] {100, 200, 300, 400, 500});

        List<List<Writable>> expSeq1Std = new ArrayList<>();
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((1 - m1) / s1), new DoubleWritable((10 - m2) / s2),
                        new DoubleWritable((100 - m3) / s3)));
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((2 - m1) / s1), new DoubleWritable((20 - m2) / s2),
                        new DoubleWritable((200 - m3) / s3)));
        expSeq1Std.add(Arrays.<Writable>asList(new DoubleWritable((3 - m1) / s1), new DoubleWritable((30 - m2) / s2),
                        new DoubleWritable((300 - m3) / s3)));

        List<List<Writable>> expSeq2Std = new ArrayList<>();
        expSeq2Std.add(Arrays.<Writable>asList(new DoubleWritable((4 - m1) / s1), new DoubleWritable((40 - m2) / s2),
                        new DoubleWritable((400 - m3) / s3)));
        expSeq2Std.add(Arrays.<Writable>asList(new DoubleWritable((5 - m1) / s1), new DoubleWritable((50 - m2) / s2),
                        new DoubleWritable((500 - m3) / s3)));


        List<List<List<Writable>>> std = new ArrayList<>(standardized.collect());
        Collections.sort(std, new ComparatorSeqLength());
        assertEquals(2, std.size());

        //        System.out.println(std);

        for (int i = 0; i < 2; i++) {
            List<List<Writable>> seqExp = (i == 0 ? expSeq1Std : expSeq2Std);
            for (int j = 0; j < seqExp.size(); j++) {
                List<Writable> stepExp = seqExp.get(j);
                List<Writable> stepAct = std.get(i).get(j);
                for (int k = 0; k < stepExp.size(); k++) {
                    assertEquals(stepExp.get(k).toDouble(), stepAct.get(k).toDouble(), 1e-6);
                }
            }
        }
    }


    private static class ComparatorFirstCol implements Comparator<List<Writable>> {
        @Override
        public int compare(List<Writable> o1, List<Writable> o2) {
            return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
        }
    }

    private static class ComparatorSeqLength implements Comparator<List<List<Writable>>> {
        @Override
        public int compare(List<List<Writable>> o1, List<List<Writable>> o2) {
            return -Integer.compare(o1.size(), o2.size());
        }
    }
}
