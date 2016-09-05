package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.impl.common.CountPartitionsFunction;
import org.junit.Test;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Created by Alex on 03/07/2016.
 */
public class TestRepartitioning extends BaseSparkTest{

    @Test
    public void testRepartitioning(){
        List<String> list = new ArrayList<>();
        for( int i=0; i<1000; i++ ){
            list.add(String.valueOf(i));
        }

        JavaRDD<String> rdd = sc.parallelize(list);
        rdd = rdd.repartition(200);

        JavaRDD<String> rdd2 = SparkUtils.repartitionBalanceIfRequired(rdd, Repartition.Always, 100, 10);
        assertFalse(rdd == rdd2);   //Should be different objects due to repartitioning

        assertEquals(10, rdd2.partitions().size());
        for( int i=0; i<10; i++ ){
            List<String> partition = rdd2.collectPartitions(new int[]{i})[0];
            System.out.println("Partition " + i + " size: " + partition.size());
            assertEquals(100, partition.size());    //Should be exactly 100, for the util method (but NOT spark .repartition)
        }
    }

    @Test
    public void testRepartitioning2() throws Exception {

        int[] ns = {320, 321, 25600, 25601, 25615};

        for( int n : ns) {

            List<String> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                list.add(String.valueOf(i));
            }

            JavaRDD<String> rdd = sc.parallelize(list);
            rdd.repartition(65);

            int totalDataSetObjectCount = n;
            int dataSetObjectsPerSplit = 8 * 4 * 10;
            int valuesPerPartition = 10;
            int nPartitions = 32;

            JavaRDD<String>[] splits = org.deeplearning4j.spark.util.SparkUtils.balancedRandomSplit(totalDataSetObjectCount, dataSetObjectsPerSplit, rdd, new Random().nextLong());

            List<Integer> counts = new ArrayList<>();
            List<List<Tuple2<Integer, Integer>>> partitionCountList = new ArrayList<>();
//            System.out.println("------------------------");
//            System.out.println("Partitions Counts:");
            for (JavaRDD<String> split : splits) {
                JavaRDD<String> repartitioned = SparkUtils.repartition(split, Repartition.Always, RepartitionStrategy.Balanced, valuesPerPartition, nPartitions);
                List<Tuple2<Integer, Integer>> partitionCounts = repartitioned.mapPartitionsWithIndex(new CountPartitionsFunction<String>(), true).collect();
//                System.out.println(partitionCounts);
                partitionCountList.add(partitionCounts);
                counts.add((int) split.count());
            }

//            System.out.println(counts.size());
//            System.out.println(counts);


            int expNumPartitionsWithMore = totalDataSetObjectCount % nPartitions;
            int actNumPartitionsWithMore = 0;
            for (List<Tuple2<Integer, Integer>> l : partitionCountList) {
                assertEquals(nPartitions, l.size());

                for (Tuple2<Integer, Integer> t2 : l) {
                    int partitionSize = t2._2();
                    if (partitionSize > valuesPerPartition) actNumPartitionsWithMore++;
                }
            }

            assertEquals(expNumPartitionsWithMore, actNumPartitionsWithMore);
        }
    }
}
