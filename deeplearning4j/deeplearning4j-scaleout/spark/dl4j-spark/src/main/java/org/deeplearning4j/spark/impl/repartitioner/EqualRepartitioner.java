package org.deeplearning4j.spark.impl.repartitioner;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.api.Repartitioner;
import org.deeplearning4j.spark.impl.common.CountPartitionsFunction;
import org.deeplearning4j.spark.impl.common.repartition.EqualPartitioner;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.util.MathUtils;
import scala.Tuple2;

import java.util.List;
import java.util.Random;

@Slf4j
public class EqualRepartitioner implements Repartitioner {
    @Override
    public <T> JavaRDD<T> repartition(JavaRDD<T> rdd, int minObjectsPerPartition, int numExecutors) {
        //minObjectsPerPartition: intentionally not used here

        //Repartition: either always, or origNumPartitions != numWorkers

        //First: count number of elements in each partition. Need to know this so we can work out how to properly index each example,
        // so we can in turn create properly balanced partitions after repartitioning
        //Because the objects (DataSets etc) should be small, this should be OK

        //Count each partition...
        List<Tuple2<Integer, Integer>> partitionCounts =
                rdd.mapPartitionsWithIndex(new CountPartitionsFunction<T>(), true).collect();
        return repartition(rdd, numExecutors, partitionCounts);
    }


    public static <T> JavaRDD<T> repartition(JavaRDD<T> rdd, int numPartitions, List<Tuple2<Integer, Integer>> partitionCounts){
        int totalObjects = 0;
        int initialPartitions = partitionCounts.size();

        for (Tuple2<Integer, Integer> t2 : partitionCounts) {
            totalObjects += t2._2();
        }

        //Check if already correct
        int minAllowable = (int)Math.floor(totalObjects / (double) numPartitions);
        int maxAllowable = (int)Math.ceil(totalObjects / (double) numPartitions);

        boolean repartitionRequired = false;
        for (Tuple2<Integer, Integer> t2 : partitionCounts) {
            if(t2._2() < minAllowable || t2._2() > maxAllowable ){
                repartitionRequired = true;
                break;
            }
        }

        if (initialPartitions == numPartitions && !repartitionRequired) {
            //log.info("Did not repartition - all correct size: initial={}, numPartitions={}", initialPartitions, numExecutors);
            //Don't need to do any repartitioning here - already in the format we want
            return rdd;
        }

        //Index each element for repartitioning (can only do manual repartitioning on a JavaPairRDD)
        JavaPairRDD<Integer, T> pairIndexed = SparkUtils.indexedRDD(rdd);

        //Handle remainder.
        //We'll randomly allocate one of these to a single partition, with no partition getting more than 1 (otherwise, imbalanced)
        //Given that we don't know exactly how Spark will allocate partitions to workers, we are probably better off doing
        // this randomly rather than "first N get +1" or "every M get +1" as this could introduce poor load balancing
        int remainder = totalObjects % numPartitions;
        int[] remainderPartitions = null;
        if (remainder > 0) {
            remainderPartitions = new int[remainder];
            int[] temp = new int[numPartitions];
            for( int i=0; i< temp.length; i++ ){
                temp[i] = i;
            }
            MathUtils.shuffleArray(temp, new Random());
            for( int i=0; i<remainder; i++ ){
                remainderPartitions[i] = temp[i];
            }
        }

        int partitionSizeExRemainder = totalObjects / numPartitions;
        pairIndexed = pairIndexed.partitionBy(new EqualPartitioner(numPartitions, partitionSizeExRemainder, remainderPartitions));

//        //DEBUGGING
//        List<Tuple2<Integer, Integer>> partitionCounts2 =
//                rdd.mapPartitionsWithIndex(new CountPartitionsFunction<T>(), true).collect();
//        List<Tuple2<Integer, Integer>> partitionCounts3 =
//                pairIndexed.values().mapPartitionsWithIndex(new CountPartitionsFunction<T>(), true).collect();
//        log.info("Partition counts - BEFORE: {}", partitionCounts2);
//        log.info("Partition counts - AFTER: {}", partitionCounts3);
        return pairIndexed.values();
    }
}
