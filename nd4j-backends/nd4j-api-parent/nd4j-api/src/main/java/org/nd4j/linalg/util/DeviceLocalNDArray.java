package org.nd4j.linalg.util;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * DeviceLocal implementation for INDArray, with special broadcast method
 * @author raver119@gmail.com
 */
@Slf4j
public class DeviceLocalNDArray extends DeviceLocal<INDArray> {

    public DeviceLocalNDArray() {
        super();
    }

    public DeviceLocalNDArray(INDArray array) {
        super();

        broadcast(array);
    }

    /**
     * This method duplicates array, and stores it to all devices
     *
     * @param array
     */
    public void broadcast(INDArray array) {
        if (array == null)
            return;

        Nd4j.getExecutioner().commit();

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
