/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.serde.ModelLoadingContext;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/** Regression coverage for device-authoritative arrays passed through model loading. */
public class ModelLoadingContextDeviceResidencyTest {

    @Test
    public void deviceAuthoritativeArrayIsNotOverwrittenByHostPublication() {
        int deviceIndex = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        DeviceDescriptor target = Nd4j.getBackendDeviceType() == DeviceType.CPU
                ? DeviceDescriptor.cpu() : DeviceDescriptor.cuda(deviceIndex);

        try (INDArray array = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 32 * 1024);
             ModelLoadingContext context = ModelLoadingContext.builder()
                     .targetDevice(target)
                     .asyncEnabled(true)
                     .parallelTransfers(2)
                     .useBatchedNativeTransfer(true)
                     .build()) {
            // Materialize the original host values, then mutate on the active backend. On CUDA,
            // the device now owns values 2, 4, ... while the host still contains 1, 2, ....
            assertEquals(1.0f, array.getFloat(0), 0.0f);
            array.muli(2.0f);

            AffinityManager.Location location = Nd4j.getAffinityManager().getActiveLocation(array);
            if (target.getDeviceType().isGpu()) {
                assertNotEquals(AffinityManager.Location.HOST, location,
                        "CUDA in-place output must be device-authoritative before publication");
            }

            context.scheduleTransfer(array);
            context.awaitTransfers();

            assertEquals(2.0f, array.getFloat(0), 0.0f,
                    "model-loader publication restored stale host bytes over the device result");
            assertEquals(65536.0f, array.getFloat(array.length() - 1), 0.0f,
                    "the full device-authoritative buffer must survive publication");
        }
    }
}
