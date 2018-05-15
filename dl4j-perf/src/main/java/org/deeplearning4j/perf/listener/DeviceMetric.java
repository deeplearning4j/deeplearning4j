package org.deeplearning4j.perf.listener;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

@Data
@Builder
public class DeviceMetric implements Serializable {

    private double load;
    private double totalMemory;
    private String deviceName;
    private double temp;
    private double memAvailable;
    private long bandwidthDeviceToHost,bandwidthHostToDevice,bandwidthDeviceToDevice;

}
