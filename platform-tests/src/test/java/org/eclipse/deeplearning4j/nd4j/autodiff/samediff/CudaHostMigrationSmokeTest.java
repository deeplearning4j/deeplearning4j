package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Checks the array operations used to prepare inputs for the DSP isolation gate. */
class CudaHostMigrationSmokeTest {
    @Test
    void onesValues() {
        try (INDArray values = Nd4j.ones(DataType.FLOAT, 1, 4)) {
            for (int i = 0; i < 4; i++) {
                assertEquals(1.0f, values.getFloat(i), 0.0f, "ones element " + i);
            }
        }
    }

    @Test
    void scalarMultiplyValues() {
        try (INDArray input = Nd4j.createFromArray(new float[]{1, 1, 1, 1});
             INDArray output = input.mul(10.0f)) {
            int launchStatus = org.bytedeco.cuda.global.cudart.cudaPeekAtLastError();
            assertEquals(0, launchStatus, "CUDA launch: "
                    + org.bytedeco.cuda.global.cudart.cudaGetErrorString(launchStatus).getString());
            for (int i = 0; i < 4; i++) {
                assertEquals(10.0f, output.getFloat(i), 0.0f, "multiply element " + i);
            }
        }
    }

    @Test
    void identityValues() {
        try (INDArray identity = Nd4j.eye(4)) {
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    assertEquals(row == col ? 1.0f : 0.0f,
                            identity.getFloat(row, col), 0.0f, "identity " + row + "," + col);
                }
            }
        }
    }
}
