package org.deeplearning4j.spark.stats;

import lombok.Getter;

/**
 * Created by Alex on 26/06/2016.
 */
public class ExampleCountEventStats extends BaseEventStats {

    @Getter private final int totalExampleCount;

    public ExampleCountEventStats(long startTime, long durationMs, int totalExampleCount) {
        super(startTime, durationMs);
        this.totalExampleCount = totalExampleCount;
    }

    public ExampleCountEventStats(String machineId, String jvmId, long threadId, long startTime, long durationMs, int totalExampleCount){
        super(machineId, jvmId, threadId, startTime, durationMs);
        this.totalExampleCount = totalExampleCount;
    }

}
