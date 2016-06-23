package org.nd4j.jita.concurrency;

import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
public class CudaAffinityManager extends BasicAffinityManager {

    private static final Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

    private static Logger logger = LoggerFactory.getLogger(CudaAffinityManager.class);

    private Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();
    private AtomicInteger devPtr = new AtomicInteger(0);

    @Override
    public Integer getDeviceForCurrentThread() {
        return getDeviceForThread(Thread.currentThread().getId());
    }

    @Override
    public Integer getDeviceForThread(Thread thread) {
        return getDeviceForThread(thread.getId());
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        if (!affinityMap.containsKey(threadId)) {
            Integer deviceId = getNextDevice();
            affinityMap.put(threadId, deviceId);
            return deviceId;
        }
        return affinityMap.get(threadId);
    }

    @Override
    public void attachThreadToDevice(Thread thread, Integer deviceId) {
        attachThreadToDevice(thread.getId(), deviceId);
    }

    @Override
    public void attachThreadToDevice(long threadId, Integer deviceId) {
        affinityMap.put(threadId, deviceId);
    }

    protected Integer getNextDevice() {
        List<Integer> devices = new ArrayList<>(configuration.getAvailableDevices());
        Integer device = null;
        if (!configuration.isForcedSingleGPU()) {
            // simple round-robin here
            synchronized (this) {
                device = devices.get(devPtr.getAndIncrement());
                if (devPtr.get() >= devices.size())
                    devPtr.set(0);

                logger.debug("Mapping to device [{}], out of [{}] devices...", device, devices.size());
            }
        } else {
            device = configuration.getAvailableDevices().get(0);
            logger.debug("Single device is forced, mapping to device [{}]", device);
        }

        return device;
    }
}
