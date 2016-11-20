package org.nd4j.jita.conf;

import lombok.NonNull;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class CudaEnvironment {
    private static final CudaEnvironment INSTANCE = new CudaEnvironment();
    private static volatile Configuration configuration;
    private static Logger logger = LoggerFactory.getLogger(CudaEnvironment.class);
    private static Map<Integer, Integer> arch = new ConcurrentHashMap<>();

    private CudaEnvironment() {
        configuration = new Configuration();
        configuration.enableDebug(configuration.isDebug());
        configuration.setVerbose(configuration.isVerbose());
        configuration.allowCrossDeviceAccess(configuration.isCrossDeviceAccessAllowed());
        configuration.setMaximumGridSize(configuration.getMaximumGridSize());
        configuration.setMaximumBlockSize(configuration.getMaximumBlockSize());
        configuration.setMinimumBlockSize(configuration.getMinimumBlockSize());
    }

    public static CudaEnvironment getInstance() {
        return INSTANCE;
    }

    public Configuration getConfiguration() {
        return configuration;
    }

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
}
