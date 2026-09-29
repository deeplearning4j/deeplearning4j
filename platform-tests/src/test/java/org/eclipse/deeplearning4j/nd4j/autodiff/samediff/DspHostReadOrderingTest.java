package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A host read during a DSP plan's first (slot-by-slot) execution must see the device writes
 * queued before it on the plan's stream. evaluate_reduction_shape reads its shape input on the
 * host, and that input comes from shape_of, a kernel on the plan's non-blocking stream. The copy
 * back once ran on legacy stream 0, which does not wait for non-blocking streams: behind a queue
 * of matmuls it read the buffer before shape_of wrote it (zeros made the reshape [n, 1] instead
 * of [1, n]; a recycled pool block produced garbage shape info).
 */
class DspHostReadOrderingTest {

    @ParameterizedTest(name = "n={0}")
    @ValueSource(ints = {512, 1024, 2048})
    void hostReadSeesPlanStreamWrites(int n) {
        SameDiff sd = SameDiff.create();
        try {
            SDVariable a = sd.placeHolder("a", DataType.FLOAT, n, n);
            // Queue milliseconds of work on the plan stream ahead of shape_of.
            SDVariable prod = a;
            for (int i = 0; i < 6; i++) {
                prod = prod.mmul(a);
            }
            SDVariable axis = sd.constant("axis", Nd4j.createFromArray(0L));
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, n);
            SDVariable out = SameDiffUtils.reductionBroadcastableWithOrigShape(prod, axis, v);

            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("a", Nd4j.rand(DataType.FLOAT, n, n).divi(n));
            INDArray values = Nd4j.linspace(DataType.FLOAT, 1, n, 1);
            placeholders.put("v", values);
            INDArray reshaped = sd.output(placeholders, out.name()).get(out.name());
            assertArrayEquals(new long[]{1, n}, reshaped.shape(), "reduction shape of [n, n] over axis 0");
            assertEquals(values.reshape(1, n), reshaped);
        } finally {
            sd.close();
        }
    }
}
