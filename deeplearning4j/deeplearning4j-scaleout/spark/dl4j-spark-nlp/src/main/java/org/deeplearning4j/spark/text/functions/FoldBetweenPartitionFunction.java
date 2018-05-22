package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.nd4j.linalg.primitives.Counter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FoldBetweenPartitionFunction implements Function2<Integer, Iterator<AtomicLong>, Iterator<Long>> {
    private Broadcast<Counter<Integer>> broadcastedMaxPerPartitionCounter;

    public FoldBetweenPartitionFunction(Broadcast<Counter<Integer>> broadcastedMaxPerPartitionCounter) {
        this.broadcastedMaxPerPartitionCounter = broadcastedMaxPerPartitionCounter;
    }

    @Override
    public Iterator<Long> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {
        int sumToAdd = 0;
        Counter<Integer> maxPerPartitionCounterInScope = broadcastedMaxPerPartitionCounter.value();

        // Add the sum of counts of all the partition with an index lower than the current one
        if (ind != 0) {
            for (int i = 0; i < ind; i++) {
                sumToAdd += maxPerPartitionCounterInScope.getCount(i);
            }
        }

        // Add the sum of counts to each element of the partition
        List<Long> itemsAddedToList = new ArrayList<>();
        while (partition.hasNext()) {
            itemsAddedToList.add(partition.next().get() + sumToAdd);
        }

        return itemsAddedToList.iterator();
    }
}
