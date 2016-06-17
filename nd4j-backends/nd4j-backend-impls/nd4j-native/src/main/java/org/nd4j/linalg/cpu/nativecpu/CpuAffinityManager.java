package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.concurrency.BasicAffinityManager;

/**
 * @author raver119@gmail.com
 */
public class CpuAffinityManager extends BasicAffinityManager {
    @Override
    public Integer getDeviceForCurrentThread() {
        return super.getDeviceForCurrentThread();
    }

    @Override
    public Integer getDeviceForThread(Thread thread) {
        return super.getDeviceForThread(thread);
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        return super.getDeviceForThread(threadId);
    }

    @Override
    public void attachThreadToDevice(Thread thread, Integer deviceId) {
        super.attachThreadToDevice(thread, deviceId);
    }

    @Override
    public void attachThreadToDevice(long threadId, Integer deviceId) {
        super.attachThreadToDevice(threadId, deviceId);
    }
}
