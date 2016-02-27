package org.nd4j.linalg.jcublas.device.conf;

import org.nd4j.linalg.jcublas.buffer.allocation.MemoryStrategies;
import org.nd4j.linalg.jcublas.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.GpuInformation;

import java.util.Properties;

/**
 * Configuration for a device.
 * This includes the memory allocation strategy
 * among other things.
 *
 *
 * @author Adam Gibson
 */
public class DeviceConfiguration {
    public final static String NAME_SPACE = DeviceConfiguration.class.getPackage().getName();
    public final static String MEMORY = NAME_SPACE + ".memory";

    private MemoryStrategy memoryStrategy;
    private int deviceNumber = 0;

    /**
     * Initialize the configuration with
     * the given properties
     * @param deviceNumber the device number to initialize
     * @param props the properties to initialize with
     *
     */
    public DeviceConfiguration(int deviceNumber,Properties props) {
        this.deviceNumber = deviceNumber;
        GpuInformation info = ContextHolder.getInstance().getInfoFor(deviceNumber);
        if(info == null)
            throw new IllegalStateException("No configuration found for " + deviceNumber);

        if(props.containsKey(MEMORY)) {
            String val = props.getProperty(MEMORY);
            MemoryStrategy strat = MemoryStrategies.getStrategy(MemoryStrategies.MemoryMode.valueOf(val));
            this.memoryStrategy = strat;
        }

        else {
            //query the device capabilities and get the strategy
            MemoryStrategy strat = MemoryStrategies.getStrategy(deviceNumber);
            this.memoryStrategy = strat;
        }

    }

    /**
     * Initialize the device configuration
     * based on the device information
     * @param deviceNumber the device number
     */
    public DeviceConfiguration(int deviceNumber) {
       this(deviceNumber,new Properties());
    }

    public MemoryStrategy getMemoryStrategy() {
        return memoryStrategy;
    }

    public void setMemoryStrategy(MemoryStrategy memoryStrategy) {
        this.memoryStrategy = memoryStrategy;
    }

    public int getDeviceNumber() {
        return deviceNumber;
    }

    public void setDeviceNumber(int deviceNumber) {
        this.deviceNumber = deviceNumber;
    }
}
