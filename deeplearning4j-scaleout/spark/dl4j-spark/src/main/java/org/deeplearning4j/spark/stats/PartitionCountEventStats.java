package org.deeplearning4j.spark.stats;

import lombok.Getter;

/**
 * Event stats implementation with partition count
 *
 * @author Alex Black
 */
public class PartitionCountEventStats extends BaseEventStats {

    @Getter
    private final int numPartitions;

    public PartitionCountEventStats(long startTime, long durationMs, int numPartitions) {
        super(startTime, durationMs);
        this.numPartitions = numPartitions;
    }

    public PartitionCountEventStats(String machineId, String jvmId, long threadId, long startTime, long durationMs,
                    int numPartitions) {
        super(machineId, jvmId, threadId, startTime, durationMs);
        this.numPartitions = numPartitions;
    }

    @Override
    public String asString(String delimiter) {
        return super.asString(delimiter) + delimiter + numPartitions;
    }

    @Override
    public String getStringHeader(String delimiter) {
        return super.getStringHeader(delimiter) + delimiter + "numPartitions";
    }
}
