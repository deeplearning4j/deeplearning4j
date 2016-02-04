package org.nd4j.jita.conf;

import lombok.Data;
import lombok.NonNull;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Data
public class CudaEnvironment {
    private Map<Integer, DeviceInformation> availableDevices = new ConcurrentHashMap<>();


    public void addDevice(@NonNull DeviceInformation deviceInformation) {
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
            return availableDevices.get(device).getAllocatedMemory();
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
            information.setAllocatedMemory(information.getAllocatedMemory() + memorySize);

        } else throw new IllegalStateException("Requested device does not exist: ["+ deviceId+"]");
    }
}
