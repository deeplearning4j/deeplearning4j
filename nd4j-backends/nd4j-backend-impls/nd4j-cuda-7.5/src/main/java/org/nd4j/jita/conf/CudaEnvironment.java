package org.nd4j.jita.conf;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import lombok.Data;
import lombok.NonNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;

/**
 * @author raver119@gmail.com
 */
@Data
public class CudaEnvironment extends Environment {
    private Map<Integer, DeviceInformation> availableDevices = new HashMap<>();

    private boolean wasReset = false;

    private static Logger log = LoggerFactory.getLogger(CudaEnvironment.class);

    public CudaEnvironment(@NonNull Configuration configuration) {

        /*
            We don't expect that CUDA was initialized somewhere else
         */

        cuInit(0);


        int count[] = new int[1];
        cuDeviceGetCount(count);

        log.info("Devices found: " + count[0]);

        for (int x = 0; x < count[0]; x++) {
            cudaDeviceProp deviceProperties = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(deviceProperties, x);



            DeviceInformation information = new DeviceInformation();
            information.setTotalMemory(deviceProperties.totalGlobalMem);
            information.setDeviceId(x);
            information.setCcMinor(deviceProperties.minor);
            information.setCcMajor(deviceProperties.major);
            information.setCanMapHostMemory(deviceProperties.canMapHostMemory != 0);
            information.setOverlappedKernels(deviceProperties.deviceOverlap != 0);
            information.setConcurrentKernels(deviceProperties.concurrentKernels != 0);
            information.setSharedMemPerBlock(deviceProperties.sharedMemPerBlock);
            information.setSharedMemPerMP(deviceProperties.sharedMemPerMultiprocessor);
            information.setWarpSize(deviceProperties.warpSize);

            JCuda.cudaSetDevice(x);

            // FIXME: remove this later
            JCuda.cudaDeviceSetLimit(0,10000);
            JCuda.cudaDeviceSetLimit(2,10000);

            long[] freemem = new long[1];
            long[] totalmem = new long[1];

            JCuda.cudaMemGetInfo(freemem, totalmem);

            log.debug("Device [" + x + "]: Free: " + freemem[0] + " Total memory: " + totalmem[0]);

            information.setAvailableMemory(freemem[0] - (64 * 1024 * 1024L));

            configuration.setMaximumDeviceAllocation(Math.min(configuration.getMaximumDeviceAllocation(), information.getAvailableMemory()));

            availableDevices.put(x, information);
        }

        JCuda.cudaSetDevice(0);

    }

    public void banDevice(Integer deviceId) {
        availableDevices.remove(deviceId);
    }

    public DeviceInformation getDeviceInformation(Integer deviceId) {
        return availableDevices.get(deviceId);
    }

    /**
     * This method exists only for debug purposes, and shouldn't be ever used in real world environment.
     *
     * @param deviceInformation
     */
    @Deprecated
    public void addDevice(@NonNull DeviceInformation deviceInformation) {
        if (!wasReset) {
            availableDevices.clear();
            wasReset = true;
            log.warn("Clearing available devices list");
        }
        if (!availableDevices.containsKey(deviceInformation.getDeviceId())) {
            availableDevices.put(deviceInformation.getDeviceId(), deviceInformation);
        }
    }

    public long getTotalMemoryForDevice(Integer device) {
        if (availableDevices.containsKey(device)) {
            return availableDevices.get(device).getTotalMemory();
        } else throw new IllegalStateException("Requested device does not exist: ["+ device+"]");
    }

    public long getAvailableMemoryForDevice(Integer device) {
        if (availableDevices.containsKey(device)) {
            return availableDevices.get(device).getAvailableMemory();
        } else throw new IllegalStateException("Requested device does not exist: ["+ device+"]");
    }

    public long getAllocatedMemoryForDevice(Integer device) {
        if (availableDevices.containsKey(device)) {
            return availableDevices.get(device).getAllocatedMemory().get();
        } else throw new IllegalStateException("Requested device does not exist: ["+ device+"]");
    }

    /**
     * This method adds
     *
     * @param deviceId
     * @param memorySize
     */
    public void trackAllocatedMemory(Integer deviceId, long memorySize) {
        // TODO: device-level lock should be considered here
        if (availableDevices.containsKey(deviceId)) {
            DeviceInformation information = availableDevices.get(deviceId);
            information.getAllocatedMemory().set(information.getAllocatedMemory().get() + memorySize);

        } else throw new IllegalStateException("Requested device does not exist: ["+ deviceId+"]");
    }
}
