package org.deeplearning4j.spark.impl.common.repartition;

import lombok.AllArgsConstructor;
import org.apache.spark.Partitioner;

/**
 * Created by Alex on 03/07/2016.
 */
@AllArgsConstructor
public class BalancedPartitioner extends Partitioner {
    private final int numPartitions;            //Total number of partitions
    private final int numStandardPartitions;    //Number of partitions of standard size; these are 1 larger than the others (==numPartitions where total number of examples is divisible into numPartitions without remainder)
    private final int elementsPerPartition;

    @Override
    public int numPartitions() {
        return numPartitions;
    }

    @Override
    public int getPartition(Object key) {
        int elementIdx = (Integer)key;

        //First 'numStandardPartitions' executors get "elementsPerPartition" each;

        //Work out: which partition it belongs to...
        if(elementIdx <= elementsPerPartition * numStandardPartitions){
            //This goes into one of the standard partitions (of size 'elementsPerPartition')
            return elementIdx / elementsPerPartition;
        } else {
            //This goes into one of the smaller partitions (of size elementsPerPartition - 1)
            int numValsInStdPartitions = elementsPerPartition * numStandardPartitions;
            int idxInSmallerPartitions = elementIdx - numValsInStdPartitions;
            int smallPartitionIdx = idxInSmallerPartitions / (elementsPerPartition-1);
            return numStandardPartitions + smallPartitionIdx;
        }
    }
}
