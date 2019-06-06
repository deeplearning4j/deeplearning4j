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

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.spark.text.accumulators.MaxPerPartitionAccumulator;
import org.nd4j.linalg.primitives.Counter;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
@SuppressWarnings("unchecked")
public class CountCumSum {

    // Starting variables
    private JavaSparkContext sc;
    private JavaRDD<AtomicLong> sentenceCountRDD;

    // Variables to fill in as we go
    private JavaRDD<AtomicLong> foldWithinPartitionRDD;
    private Broadcast<Counter<Integer>> broadcastedMaxPerPartitionCounter;
    private JavaRDD<Long> cumSumRDD;

    // Constructor
    public CountCumSum(JavaRDD<AtomicLong> sentenceCountRDD) {
        this.sentenceCountRDD = sentenceCountRDD;
        this.sc = new JavaSparkContext(sentenceCountRDD.context());
    }

    // Getter
    public JavaRDD<Long> getCumSumRDD() {
        if (cumSumRDD != null) {
            return cumSumRDD;
        } else {
            throw new IllegalAccessError("Cumulative Sum list not defined. Call buildCumSum() first.");
        }
    }

    // For each equivalent for partitions
    public void actionForMapPartition(JavaRDD rdd) {
        // Action to fill the accumulator
        rdd.foreachPartition(new MapPerPartitionVoidFunction());
    }

    // Do cum sum within the partition
    public void cumSumWithinPartition() {

        // Accumulator to get the max of the cumulative sum in each partition
        final Accumulator<Counter<Integer>> maxPerPartitionAcc =
                        sc.accumulator(new Counter<Integer>(), new MaxPerPartitionAccumulator());
        // Partition mapping to fold within partition
        foldWithinPartitionRDD = sentenceCountRDD
                        .mapPartitionsWithIndex(new FoldWithinPartitionFunction(maxPerPartitionAcc), true).cache();
        actionForMapPartition(foldWithinPartitionRDD);

        // Broadcast the counter (partition index : sum of count) to all workers
        broadcastedMaxPerPartitionCounter = sc.broadcast(maxPerPartitionAcc.value());
    }

    public void cumSumBetweenPartition() {

        cumSumRDD = foldWithinPartitionRDD
                        .mapPartitionsWithIndex(new FoldBetweenPartitionFunction(broadcastedMaxPerPartitionCounter),
                                        true)
                        .setName("cumSumRDD").cache();
        foldWithinPartitionRDD.unpersist();
    }

    public JavaRDD<Long> buildCumSum() {
        cumSumWithinPartition();
        cumSumBetweenPartition();
        return getCumSumRDD();
    }
}
