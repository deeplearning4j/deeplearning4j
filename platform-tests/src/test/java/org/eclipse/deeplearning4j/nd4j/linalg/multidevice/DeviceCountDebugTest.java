package org.eclipse.deeplearning4j.nd4j.linalg.multidevice;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.List;

public class DeviceCountDebugTest {
    @Test
    public void testDeviceCount() {
        System.out.println("=== Device Count Debug ===");

        int nativeDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        System.out.println("Native getAvailableDevices(): " + nativeDevices);

        List<Integer> availableDeviceIds = Nd4j.getAffinityManager().getAvailableDeviceIds();
        System.out.println("AffinityManager availableDeviceIds.size(): " + availableDeviceIds.size());
        System.out.println("AffinityManager availableDeviceIds: " + availableDeviceIds);

        int affinityDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        System.out.println("AffinityManager.getNumberOfDevices(): " + affinityDevices);

        System.out.println("=== End Debug ===");
    }
}
