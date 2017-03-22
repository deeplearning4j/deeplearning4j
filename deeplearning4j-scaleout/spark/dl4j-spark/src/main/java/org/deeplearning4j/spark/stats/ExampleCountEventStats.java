package org.deeplearning4j.spark.stats;

import lombok.Getter;

/**
 * Event stats implementation with number of examples
 *
 * @author Alex Black
 */
public class ExampleCountEventStats extends BaseEventStats {

    @Getter
    private final int totalExampleCount;

    public ExampleCountEventStats(long startTime, long durationMs, int totalExampleCount) {
        super(startTime, durationMs);
        this.totalExampleCount = totalExampleCount;
    }

    public ExampleCountEventStats(String machineId, String jvmId, long threadId, long startTime, long durationMs,
                    int totalExampleCount) {
        super(machineId, jvmId, threadId, startTime, durationMs);
        this.totalExampleCount = totalExampleCount;
    }

    @Override
    public String asString(String delimiter) {
        return super.asString(delimiter) + delimiter + totalExampleCount;
    }

    @Override
    public String getStringHeader(String delimiter) {
        return super.getStringHeader(delimiter) + delimiter + "totalExampleCount";
    }
}
