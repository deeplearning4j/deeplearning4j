package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function2;
import org.nd4j.linalg.primitives.Counter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FoldWithinPartitionFunction implements Function2<Integer, Iterator<AtomicLong>, Iterator<AtomicLong>> {

    public FoldWithinPartitionFunction(Accumulator<Counter<Integer>> maxPartitionAcc) {
        this.maxPerPartitionAcc = maxPartitionAcc;
    }

    private Accumulator<Counter<Integer>> maxPerPartitionAcc;


    @Override
    public Iterator<AtomicLong> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {

        List<AtomicLong> foldedItemList = new ArrayList<AtomicLong>() {
            {
                add(new AtomicLong(0L));
            }
        };

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
}
