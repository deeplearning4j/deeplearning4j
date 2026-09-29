package org.eclipse.deeplearning4j.nd4j.linalg.reduce;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * An along-dimension reduction of one row must not depend on how many rows the call
 * reduces: row r of a [1, W, H, D] reduction over D equals the reduction of row r alone.
 * The CUDA launch once sized its thread block from the number of outputs, so a 5-row
 * verification window and a width-1 decode step summed the same 128 values in different
 * orders (the Qwen3.6-27B GDN q/k norm sums differed by 1 ulp).
 */
class ReduceAlongDimensionRowInvarianceTest {
    private static final int HEADS = 16;

    @ParameterizedTest(name = "{0} D={1} W={2}")
    @CsvSource({"sum,128,5", "sum,128,64", "sum,40,5", "sum,300,5", "mean,128,5", "max,128,5",
            "norm2,128,5", "argmax,128,5", "any,128,5", "countNonZero,128,5"})
    void rowResultIndependentOfRowCount(String op, int depth, int window) {
        Random random = new Random(20260929L + depth * 31L + window);
        float[] values = new float[window * HEADS * depth];
        for (int i = 0; i < values.length; i++) {
            // Many exponents so that a different accumulation order changes the rounding.
            values[i] = (random.nextFloat() * 2f - 1f) * (float) Math.pow(2, random.nextInt(12) - 6);
            if (op.equals("any") || op.equals("countNonZero")) values[i] = random.nextInt(4) == 0 ? 0f : values[i];
        }
        INDArray all = Nd4j.create(values, new long[]{1, window, HEADS, depth}, 'c');
        Function<INDArray, INDArray> reduce = reduction(op);
        INDArray windowResult = reduce.apply(all);
        for (int row = 0; row < window; row++) {
            INDArray single = all.get(NDArrayIndex.all(), NDArrayIndex.interval(row, row + 1), NDArrayIndex.all(),
                    NDArrayIndex.all()).dup('c');
            INDArray rowResult = reduce.apply(single);
            INDArray expected = windowResult.get(NDArrayIndex.all(), NDArrayIndex.interval(row, row + 1)).dup('c');
            assertArrayEquals(expected.shape(), rowResult.shape(), op + " shape");
            assertEquals(expected.dataType(), rowResult.dataType(), op + " dtype");
            double[] e = expected.castTo(DataType.DOUBLE).data().asDouble();
            double[] a = rowResult.castTo(DataType.DOUBLE).data().asDouble();
            for (int i = 0; i < e.length; i++) {
                assertEquals(Double.doubleToRawLongBits(e[i]), Double.doubleToRawLongBits(a[i]),
                        op + " row " + row + " head " + i + ": window=" + e[i] + " single=" + a[i]);
            }
        }
    }

    private static Function<INDArray, INDArray> reduction(String op) {
        switch (op) {
            case "sum": return x -> x.sum(true, 3);
            case "mean": return x -> x.mean(true, 3);
            case "max": return x -> x.max(true, 3);
            case "norm2": return x -> x.norm2(true, 3);
            case "argmax": return x -> Nd4j.argMax(x, 3);
            case "any": return x -> Nd4j.getExecutioner().exec(
                    new org.nd4j.linalg.api.ops.impl.reduce.bool.Any(x, true, 3L));
            case "countNonZero": return x -> Nd4j.getExecutioner().exec(
                    new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(x, true, new long[]{3}));
            default: throw new IllegalArgumentException(op);
        }
    }
}
