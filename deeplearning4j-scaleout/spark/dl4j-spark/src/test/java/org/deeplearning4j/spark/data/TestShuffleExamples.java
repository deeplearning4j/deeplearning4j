package org.deeplearning4j.spark.data;

import org.apache.spark.HashPartitioner;
import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.data.shuffle.IntPartitioner;
import org.deeplearning4j.spark.util.SparkUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 06/01/2017.
 */
public class TestShuffleExamples extends BaseSparkTest {

    @Test
    public void testShuffle() {
        List<DataSet> list = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            INDArray f = Nd4j.valueArrayOf(new int[] {10, 1}, i);
            INDArray l = f.dup();

            DataSet ds = new DataSet(f, l);
            list.add(ds);
        }

        JavaRDD<DataSet> rdd = sc.parallelize(list);

        JavaRDD<DataSet> shuffled = SparkUtils.shuffleExamples(rdd, 10, 10);

        List<DataSet> shuffledList = shuffled.collect();

        int totalExampleCount = 0;
        for (DataSet ds : shuffledList) {
            totalExampleCount += ds.getFeatures().length();
            System.out.println(Arrays.toString(ds.getFeatures().data().asFloat()));

            assertEquals(ds.getFeatures(), ds.getLabels());
        }

        assertEquals(100, totalExampleCount);
    }

    @Test
    public void testIntPartitioner() {
        int nTest = 10000;
        int nPartitions = 42;

        Partitioner intPartitioner = new IntPartitioner(nPartitions);
        Partitioner hashPartitioner = new HashPartitioner(nPartitions);

        Random r = new Random();

        List<Integer> samples = new ArrayList();
        for (int i = 0; i < nTest; i++) {
            samples.add(Math.abs(r.nextInt()));
        }

        for (int i : samples) {
            assertTrue("Found intPartitioner " + intPartitioner.getPartition(i) + " for value " + i
                            + " with hashPartitioner " + hashPartitioner.getPartition(i),
                            intPartitioner.getPartition(i) == hashPartitioner.getPartition(i));
        }

    }
}
