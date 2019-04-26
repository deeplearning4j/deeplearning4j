package org.nd4j.jita.allocator;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
public class DeviceLocalNDArrayTests {

    @Test
    public void testDeviceLocalArray() throws Exception{
        val arr = Nd4j.create(DataType.DOUBLE, 10, 10);

        val dl = new DeviceLocalNDArray(arr);

        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    dl.get().addi(1.0);
                    Nd4j.getExecutioner().commit();
                }
            });
            Nd4j.getAffinityManager().attachThreadToDevice(t, e);
            t.start();
            t.join();
        }
    }
}
