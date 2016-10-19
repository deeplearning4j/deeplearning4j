package org.nd4j.linalg.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class DeviceLocalNDArray extends DeviceLocal<INDArray> {

    public DeviceLocalNDArray() {
        super();
    }

    public DeviceLocalNDArray(INDArray array) {
        super();
        fill(array);
    }

    /**
     * This method duplicates array, and stores it to all devices
     *
     * @param array
     */
    public void fill(INDArray array) {
        if (array == null)
            return;

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int i = 0; i < numDevices; i++) {
            // if current thread equal to this device - we just save it, without duplication
            if (Nd4j.getAffinityManager().getDeviceForCurrentThread() == i) {
                set(i, array);
            } else {
                set(i, Nd4j.getAffinityManager().replicateToDevice(i, array));
            }

        }
    }
}
