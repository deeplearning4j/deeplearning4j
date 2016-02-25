package org.nd4j.jita.conf;

import lombok.Data;

import java.util.concurrent.atomic.AtomicLong;

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
    private AtomicLong allocatedMemory = new AtomicLong(0);

    /*
        Key features we care about: hostMapped, overlapped exec, number of cores/sm
     */
    private boolean canMapHostMemory = false;

    private boolean overlappedKernels = false;

    private boolean concurrentKernels = false;

    private long sharedMemPerBlock = 0;

    private long sharedMemPerMP = 0;

    private int warpSize = 0;
}
