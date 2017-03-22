package org.deeplearning4j.spark.stats;

import org.deeplearning4j.util.UIDProvider;

/**
 * Created by Alex on 26/06/2016.
 */
public class BaseEventStats implements EventStats {

    protected final String machineId;
    protected final String jvmId;
    protected final long threadId;
    protected final long startTime;
    protected final long durationMs;

    public BaseEventStats(long startTime, long durationMs) {
        this(UIDProvider.getHardwareUID(), UIDProvider.getJVMUID(), Thread.currentThread().getId(), startTime,
                        durationMs);
    }

    public BaseEventStats(String machineId, String jvmId, long threadId, long startTime, long durationMs) {
        this.machineId = machineId;
        this.jvmId = jvmId;
        this.threadId = threadId;
        this.startTime = startTime;
        this.durationMs = durationMs;
    }

    @Override
    public String getMachineID() {
        return machineId;
    }

    @Override
    public String getJvmID() {
        return jvmId;
    }

    @Override
    public long getThreadID() {
        return threadId;
    }

    @Override
    public long getStartTime() {
        return startTime;
    }

    @Override
    public long getDurationMs() {
        return durationMs;
    }

    @Override
    public String asString(String delimiter) {
        return machineId + delimiter + jvmId + delimiter + threadId + delimiter + startTime + delimiter + durationMs;
    }

    @Override
    public String getStringHeader(String delimiter) {
        return "machineId" + delimiter + "jvmId" + delimiter + "threadId" + delimiter + "startTime" + delimiter
                        + "durationMs";
    }
}
