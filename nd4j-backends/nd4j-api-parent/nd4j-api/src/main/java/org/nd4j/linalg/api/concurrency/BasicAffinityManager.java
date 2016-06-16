package org.nd4j.linalg.api.concurrency;

/**
 * @author raver119@gmail.com
 */
public abstract class BasicAffinityManager implements AffinityManager {
    @Override
    public Integer getDeviceForCurrentThread() {
        return 0;
    }

    @Override
    public Integer getDeviceForThread(Thread thread) {
        return 0;
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        return 0;
    }

    @Override
    public void attachThreadToDevice(Thread thread, Integer deviceId) {
        // no-op
    }

    @Override
    public void attachThreadToDevice(long threadId, Integer deviceId) {
        // no-op
    }
}
