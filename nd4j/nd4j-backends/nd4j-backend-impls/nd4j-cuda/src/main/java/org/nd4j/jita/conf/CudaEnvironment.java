package org.nd4j.jita.conf;

import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * The cuda environment contains information
 * for a given {@link Configuration}
 * singleton.
 *
 * @author raver119@gmail.com
 */
public class CudaEnvironment {
    private static final CudaEnvironment INSTANCE = new CudaEnvironment();
    private static volatile Configuration configuration;
    private static Map<Integer, Integer> arch = new ConcurrentHashMap<>();

    private CudaEnvironment() {
        configuration = new Configuration();

    }

    public static CudaEnvironment getInstance() {
        return INSTANCE;
    }

    /**
     * Get the {@link Configuration}
     * for the environment
     * @return
     */
    public Configuration getConfiguration() {
        return configuration;
    }

    /**
     * Get the current device architecture
     * @return the major/minor version of
     * the current device
     */
    public int getCurrentDeviceArchitecture() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (!arch.containsKey(deviceId)) {
            int major = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMajor(new CudaPointer(deviceId));
            int minor = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMinor(new CudaPointer(deviceId));
            Integer cc = Integer.parseInt(new String("" + major + minor));
            arch.put(deviceId, cc);
            return cc;
        }

        return arch.get(deviceId);
    }

    public void notifyConfigurationApplied() {
        configuration.updateDevice();
    }
}
