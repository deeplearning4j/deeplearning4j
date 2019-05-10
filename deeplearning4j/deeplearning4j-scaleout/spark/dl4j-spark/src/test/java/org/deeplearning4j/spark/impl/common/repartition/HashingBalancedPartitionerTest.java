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

package org.deeplearning4j.spark.impl.common.repartition;

import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.common.repartition.HashingBalancedPartitioner.LinearCongruentialGenerator;
import org.junit.Test;
import scala.Tuple2;

import java.util.*;

import static org.junit.Assert.assertTrue;


/**
 * Created by huitseeker on 4/4/17.
 */
public class HashingBalancedPartitionerTest extends BaseSparkTest {

    // e.g. we have 3 partitions, with red and blue elements, red is indexed by 0, blue by 1:
    //  [ r, r, r, r, b, b, b ], [r, b, b], [b, b, b, b, b, r, r]
    // avg # red elems per partition : 2.33
    // avg # blue elems per partition : 3.33
    // partitionWeightsByClass = [[1.714, .429, .857], [0.9, 0.6, 1.5]]

    @Test
    public void hashingBalancedPartitionerDoesBalance() {
        // partitionWeightsByClass = [[1.714, .429, .857], [0.9, 0.6, 1.5]]
        List<Double> reds = Arrays.asList(1.714D, 0.429D, .857D);
        List<Double> blues = Arrays.asList(0.9D, 0.6D, 1.5D);
        List<List<Double>> partitionWeights = Arrays.asList(reds, blues);

        HashingBalancedPartitioner hbp = new HashingBalancedPartitioner(partitionWeights);
        List<Tuple2<Integer, String>> l = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            l.add(new Tuple2<Integer, String>(0, "red"));
        }
        for (int i = 0; i < 3; i++) {
            l.add(new Tuple2<Integer, String>(0, "blue"));
        }
        for (int i = 0; i < 1; i++) {
            l.add(new Tuple2<Integer, String>(1, "red"));
        }
        for (int i = 0; i < 2; i++) {
            l.add(new Tuple2<Integer, String>(1, "blue"));
        }
        for (int i = 0; i < 2; i++) {
            l.add(new Tuple2<Integer, String>(2, "red"));
        }
        for (int i = 0; i < 5; i++) {
            l.add(new Tuple2<Integer, String>(2, "blue"));
        }
        // This should give exactly the sought distribution
        JavaPairRDD<Integer, String> rdd =
                        JavaPairRDD.fromJavaRDD(sc.parallelize(l)).partitionBy(new HashPartitioner(3));

        // Let's reproduce UIDs
        JavaPairRDD<Tuple2<Long, Integer>, String> indexedRDD = rdd.zipWithUniqueId().mapToPair(
                        new PairFunction<Tuple2<Tuple2<Integer, String>, Long>, Tuple2<Long, Integer>, String>() {
                            @Override
                            public Tuple2<Tuple2<Long, Integer>, String> call(
                                            Tuple2<Tuple2<Integer, String>, Long> payLoadNuid) {
                                Long uid = payLoadNuid._2();
                                String value = payLoadNuid._1()._2();
                                Integer elemClass = value.equals("red") ? 0 : 1;
                                return new Tuple2<Tuple2<Long, Integer>, String>(
                                                new Tuple2<Long, Integer>(uid, elemClass), value);
                            }
                        });

        List<Tuple2<Tuple2<Long, Integer>, String>> testList = indexedRDD.collect();

        int[][] colorCountsByPartition = new int[3][2];
        for (final Tuple2<Tuple2<Long, Integer>, String> val : testList) {
            System.out.println(val);
            Integer partition = hbp.getPartition(val._1());
            System.out.println(partition);

            if (val._2().equals("red"))
                colorCountsByPartition[partition][0] += 1;
            else
                colorCountsByPartition[partition][1] += 1;
        }

        for (int i = 0; i < 3; i++) {
            System.out.println(Arrays.toString(colorCountsByPartition[i]));
        }
        for (int i = 0; i < 3; i++) {
            // avg red per partition : 2.33
            assertTrue(colorCountsByPartition[i][0] >= 1 && colorCountsByPartition[i][0] < 4);
            // avg blue per partition : 3.33
            assertTrue(colorCountsByPartition[i][1] >= 2 && colorCountsByPartition[i][1] < 5);
        }

    }

    @Test
    public void hashPartitionerBalancesAtScale() {
        LinearCongruentialGenerator r = new LinearCongruentialGenerator(10000);
        List<String> elements = new ArrayList<String>();
        for (int i = 0; i < 10000; i++) {
            // The red occur towards the end
            if (r.nextDouble() < ((double) i / 10000D))
                elements.add("red");
            // The blue occur towards the front
            if (r.nextDouble() < (1 - (double) i / 10000D))
                elements.add("blue");
        }
        Integer countRed = 0;
        Integer countBlue = 0;
        for (String elem : elements) {
            if (elem.equals("red"))
                countRed++;
            else
                countBlue++;
        }
        JavaRDD<String> rdd = sc.parallelize(elements);
        JavaPairRDD<Tuple2<Long, Integer>, String> indexedRDD = rdd.zipWithUniqueId()
                        .mapToPair(new PairFunction<Tuple2<String, Long>, Tuple2<Long, Integer>, String>() {
                            @Override
                            public Tuple2<Tuple2<Long, Integer>, String> call(Tuple2<String, Long> stringLongTuple2)
                                            throws Exception {
                                Integer elemClass = stringLongTuple2._1().equals("red") ? 0 : 1;
                                return new Tuple2<Tuple2<Long, Integer>, String>(
                                                new Tuple2<Long, Integer>(stringLongTuple2._2(), elemClass),
                                                stringLongTuple2._1());
                            }
                        });

        Integer numPartitions = indexedRDD.getNumPartitions();

        // rdd and indexedRDD have the same partition distribution
        List<Tuple2<Integer, Integer>> partitionTuples =
                        rdd.mapPartitionsWithIndex(new CountRedBluePartitionsFunction(), true).collect();
        List<Double> redWeights = new ArrayList<Double>();
        List<Double> blueWeights = new ArrayList<Double>();
        Float avgRed = (float) countRed / numPartitions;
        Float avgBlue = (float) countBlue / numPartitions;
        for (int i = 0; i < partitionTuples.size(); i++) {
            Tuple2<Integer, Integer> counts = partitionTuples.get(i);
            redWeights.add((double) counts._1() / avgRed);
            blueWeights.add((double) counts._2() / avgBlue);
        }
        List<List<Double>> partitionWeights = Arrays.asList(redWeights, blueWeights);


        HashingBalancedPartitioner hbp = new HashingBalancedPartitioner(partitionWeights);

        List<Tuple2<Tuple2<Long, Integer>, String>> testList = indexedRDD.collect();

        int[][] colorCountsByPartition = new int[numPartitions][2];
        for (final Tuple2<Tuple2<Long, Integer>, String> val : testList) {
            Integer partition = hbp.getPartition(val._1());

            if (val._2().equals("red"))
                colorCountsByPartition[partition][0] += 1;
            else
                colorCountsByPartition[partition][1] += 1;
        }

        for (int i = 0; i < numPartitions; i++) {
            System.out.println(Arrays.toString(colorCountsByPartition[i]));
        }

        System.out.println("Ideal red # per partition: " + avgRed);
        System.out.println("Ideal blue # per partition: " + avgBlue);

        for (int i = 0; i < numPartitions; i++) {
            // avg red per partition : 2.33
            assertTrue(colorCountsByPartition[i][0] >= Math.round(avgRed * .99)
                            && colorCountsByPartition[i][0] < Math.round(avgRed * 1.01) + 1);
            // avg blue per partition : 3.33
            assertTrue(colorCountsByPartition[i][1] >= Math.round(avgBlue * .99)
                            && colorCountsByPartition[i][1] < Math.round(avgBlue * 1.01) + 1);
        }


    }

    class CountRedBluePartitionsFunction
                    implements Function2<Integer, Iterator<String>, Iterator<Tuple2<Integer, Integer>>> {
        @Override
        public Iterator<Tuple2<Integer, Integer>> call(Integer v1, Iterator<String> v2) throws Exception {

            int redCount = 0;
            int blueCount = 0;
            while (v2.hasNext()) {
                String elem = v2.next();
                if (elem.equals("red"))
                    redCount++;
                else
                    blueCount++;
            }

            return Collections.singletonList(new Tuple2<>(redCount, blueCount)).iterator();
        }
    }

}
