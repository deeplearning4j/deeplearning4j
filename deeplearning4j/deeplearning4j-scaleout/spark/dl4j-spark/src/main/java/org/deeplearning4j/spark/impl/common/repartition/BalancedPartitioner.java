package org.deeplearning4j.spark.impl.common.repartition;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Partitioner;

import java.util.Random;

/**
 * This is a custom partitioner (used in conjunction with {@link AssignIndexFunction} to repartition a RDD.
 * Unlike a standard .repartition() call (which assigns partitions like [2,3,4,1,2,3,4,1,2,...] for 4 partitions],
 * this function attempts to keep contiguous elements (i.e., those elements originally in the same partition) together
 * much more frequently. Furthermore, it is less prone to producing larger or smaller than expected partitions, as
 * it is entirely deterministic, whereas .repartition() has a degree of randomness (i.e., start index) which can result in
 * a large degree of variance when the number of elements in the original partitions is small (as is the case generally in DL4J)
 *
 * @author Alex Black
 */
@Slf4j
public class BalancedPartitioner extends Partitioner {
    private final int numPartitions; //Total number of partitions
    private final int elementsPerPartition;
    private final int remainder;
    private Random r;

    public BalancedPartitioner(int numPartitions, int elementsPerPartition, int remainder) {
        this.numPartitions = numPartitions;
        this.elementsPerPartition = elementsPerPartition;
        this.remainder = remainder;
    }

    @Override
    public int numPartitions() {
        return numPartitions;
    }

    @Override
    public int getPartition(Object key) {
        int elementIdx = key.hashCode();

        //First 'remainder' executors get elementsPerPartition+1 each; the remainder get
        // elementsPerPartition each. This is because the total number of examples might not be an exact multiple
        // of the number of cores in the cluster

        //Work out: which partition it belongs to...
        if (elementIdx <= (elementsPerPartition + 1) * remainder) {
            //This goes into one of the larger partitions (of size elementsPerPartition+1)
            int outputPartition = elementIdx / (elementsPerPartition + 1);
            if (outputPartition >= numPartitions) {
                //Should never happen, unless there's some up-stream problem with calculating elementsPerPartition
                outputPartition = getRandom().nextInt(numPartitions);
                log.warn("**** Random partition assigned (1): elementIdx={}, numPartitions={}, elementsPerPartition={}, remainder={}",
                        elementIdx, numPartitions, elementsPerPartition, remainder);
            }
            return outputPartition;
        } else {
            //This goes into one of the standard size partitions (of size elementsPerPartition)
            int numValsInLargerPartitions = remainder * (elementsPerPartition + 1);
            int idxInSmallerPartitions = elementIdx - numValsInLargerPartitions;
            int smallPartitionIdx = idxInSmallerPartitions / elementsPerPartition;
            int outputPartition = remainder + smallPartitionIdx;
            if (outputPartition >= numPartitions) {
                //Should never happen, unless there's some up-stream problem with calculating elementsPerPartition
                outputPartition = getRandom().nextInt(numPartitions);
                log.warn("**** Random partition assigned (2): elementIdx={}, numPartitions={}, elementsPerPartition={}, remainder={}",
                        elementIdx, numPartitions, elementsPerPartition, remainder);
            }
            return outputPartition;
        }
    }

    private synchronized Random getRandom() {
        if (r == null)
            r = new Random();
        return r;
    }
}
