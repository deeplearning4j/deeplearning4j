package org.deeplearning4j.spark.text;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
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
        rdd.foreachPartition(new VoidFunction<Iterator<?>>() {
            @Override
            public void call(Iterator<?> integerIterator) throws Exception {}
        });
    }

    // Do cum sum within the partition
    public void cumSumWithinPartition() {

        // Accumulator to get the max of the cumulative sum in each partition
        final Accumulator<Counter<Integer>> maxPerPartitionAcc = sc.accumulator(new Counter<Integer>(),
                                                                                new MaxPerPartitionAccumulator());

        // Cumulative sum in each partition
        Function2 foldWithinPartition = new Function2<Integer, Iterator<AtomicLong>, Iterator<AtomicLong>>(){
            @Override
            public Iterator<AtomicLong> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {

                List<AtomicLong> foldedItemList = new ArrayList<AtomicLong>() {{ add(new AtomicLong(0L)); }};

                // Recurrent state implementation of cum sum
                int foldedItemListSize = 1;
                while (partition.hasNext()) {
                    long curPartitionItem = partition.next().get();
                    int lastFoldedIndex = foldedItemListSize - 1;
                    long lastFoldedItem = foldedItemList.get(lastFoldedIndex).get();
                    AtomicLong sumLastCurrent = new AtomicLong(curPartitionItem + lastFoldedItem);

                    foldedItemList.set(lastFoldedIndex, sumLastCurrent);
                    foldedItemList.add(sumLastCurrent);
                    foldedItemListSize += 1;
                }

                // Update Accumulator
                long maxFoldedItem = foldedItemList.remove(foldedItemListSize - 1).get();
                Counter<Integer> partitionIndex2maxItemCounter = new Counter<>();
                partitionIndex2maxItemCounter.incrementCount(ind, maxFoldedItem);
                maxPerPartitionAcc.add(partitionIndex2maxItemCounter);

                return foldedItemList.iterator();
            }
        };
        // Partition mapping to fold within partition
        foldWithinPartitionRDD = sentenceCountRDD.mapPartitionsWithIndex(foldWithinPartition, true).cache();
        actionForMapPartition(foldWithinPartitionRDD);

        // Broadcast the counter (partition index : sum of count) to all workers
        broadcastedMaxPerPartitionCounter = sc.broadcast(maxPerPartitionAcc.value());
    }

    public void cumSumBetweenPartition() {

        Function2 foldBetweenPartitions = new Function2<Integer, Iterator<AtomicLong>, Iterator<Long>>() {
            @Override
            public Iterator<Long> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {
                int sumToAdd = 0;
                Counter<Integer> maxPerPartitionCounterInScope = broadcastedMaxPerPartitionCounter.value();

                // Add the sum of counts of all the partition with an index lower than the current one
                if (ind != 0) {
                    for (int i=0; i < ind; i++) { sumToAdd += maxPerPartitionCounterInScope.getCount(i); }
                }

                // Add the sum of counts to each element of the partition
                List<Long> itemsAddedToList = new ArrayList<>();
                while (partition.hasNext()) {
                    itemsAddedToList.add(partition.next().get() + sumToAdd);
                }

                return itemsAddedToList.iterator();
            }
        };
        cumSumRDD = foldWithinPartitionRDD.mapPartitionsWithIndex(foldBetweenPartitions, true)
                                          .setName("cumSumRDD").cache();
        foldWithinPartitionRDD.unpersist();
    }

    public JavaRDD<Long> buildCumSum() {
        cumSumWithinPartition();
        cumSumBetweenPartition();
        return getCumSumRDD();
    }
}
