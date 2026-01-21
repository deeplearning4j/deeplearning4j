package org.eclipse.deeplearning4j.nd4j.linalg.multidevice;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.jita.conf.CudaEnvironment;

public class DeviceCountDebugTest {
    @Test
    public void testDeviceCount() {
        System.out.println("=== Device Count Debug ===");
        
        int nativeDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        System.out.println("Native getAvailableDevices(): " + nativeDevices);
        
        int configDevices = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size();
        System.out.println("Configuration availableDevices.size(): " + configDevices);
        System.out.println("Configuration availableDevices: " + CudaEnvironment.getInstance().getConfiguration().getAvailableDevices());
        
        int affinityDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        System.out.println("AffinityManager.getNumberOfDevices(): " + affinityDevices);
        
        System.out.println("=== End Debug ===");
    }
}
