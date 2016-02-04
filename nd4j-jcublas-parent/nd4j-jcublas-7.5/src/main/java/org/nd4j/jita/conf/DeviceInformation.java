package org.nd4j.jita.conf;

import lombok.Data;

/**
 * @author raver119@gmail.com
 */
@Data
public class DeviceInformation {
    private int deviceId;

    private int ccMajor = 0;
    private int ccMinor = 0;

    /**
     * Total amount of memory available on current specific device
     */
    private long totalMemory = 0;

    /**
     * Available RAM
     */
    private long availableMemory = 0;

    /**
     * This is amount of RAM allocated within current JVM process
     */
    private long allocatedMemory = 0;

    /*
        Key features we care about: hostMapped, overlapped exec, number of cores/sm
     */
}
