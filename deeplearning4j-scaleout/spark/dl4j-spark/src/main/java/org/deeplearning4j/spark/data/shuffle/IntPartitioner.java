package org.deeplearning4j.spark.data.shuffle;

import lombok.AllArgsConstructor;
import org.apache.spark.Partitioner;

/**
 * A very simple partitioner that assumes integer keys.
 * Maps each value to key % numPartitions
 *
 * @author Alex Black
 * @deprecated Use {@link org.apache.spark.HashPartitioner} instead
 */
@Deprecated
@AllArgsConstructor
public class IntPartitioner extends Partitioner {

    private final int numPartitions;

    @Override
    public int numPartitions() {
        return numPartitions;
    }

    @Override
    public int getPartition(Object key) {
        return (Integer) key % numPartitions;
    }
}
