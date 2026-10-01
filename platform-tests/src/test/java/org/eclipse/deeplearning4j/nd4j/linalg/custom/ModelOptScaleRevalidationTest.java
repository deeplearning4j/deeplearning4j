package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptFp8Linear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptNvfp4Linear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Host validation of ModelOpt checkpoint scales has to follow a scale's contents, not its
 * address. A scale that was valid when first seen and is then rewritten in place, or another
 * view of the same storage, must still be rejected on the host before any launch. Letting it
 * through leaves it to the kernels' device-side check, which traps and leaves a sticky CUDA
 * error behind for the rest of the process, so every case also runs a valid call afterwards.
 */
@NativeTag
public class ModelOptScaleRevalidationTest extends BaseNd4jTestWithBackends {
    /** Text shared by the host validator's messages; a device trap reports a CUDA error instead. */
    private static final String HOST_REJECTION = "must be positive and finite";
    private static final byte[] BLOCKS = {0x38, 0x39, 0x30, 0x40, 0x28, 0x3c};

    @Override
    public char ordering() { return 'c'; }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void rewrittenBlockScaleIsRevalidated(Nd4jBackend backend) {
        INDArray x = activations();
        INDArray w = raw(DataType.UBYTE, nvfp4Weights(), 3, 16);
        INDArray scale = raw(DataType.FLOAT8, BLOCKS.clone(), 3, 2);
        INDArray second = scalar(.1003f);
        INDArray expected = run(true, x, w, scale, second);

        for (int bad : new int[]{0, 0xb8, 0x7f}) {
            overwrite(scale, 5, (byte) bad);
            assertHostRejection(true, x, w, scale, second);
        }

        overwrite(scale, 5, BLOCKS[5]);
        assertEquals(expected, run(true, x, w, scale, second));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void rewrittenFp8ScaleIsRevalidated(Nd4jBackend backend) {
        INDArray x = activations();
        INDArray w = raw(DataType.FLOAT8, fp8Weights(), 3, 32);
        INDArray scale = scalar(.7f);
        INDArray second = scalar(.3f);
        INDArray expected = run(false, x, w, scale, second);

        for (float bad : new float[]{0, -1, Float.NaN, Float.POSITIVE_INFINITY}) {
            overwrite(scale, bad);
            assertHostRejection(false, x, w, scale, second);
        }

        overwrite(scale, .7f);
        assertEquals(expected, run(false, x, w, scale, second));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyViewOfValidatedStorageIsValidated(Nd4jBackend backend) {
        // Two [3, 2] block-scale tensors in one allocation; only the second holds a zero scale.
        byte[] both = new byte[2 * BLOCKS.length];
        System.arraycopy(BLOCKS, 0, both, 0, BLOCKS.length);
        System.arraycopy(BLOCKS, 0, both, BLOCKS.length, BLOCKS.length);
        both[both.length - 1] = 0;
        INDArray storage = raw(DataType.FLOAT8, both, 6, 2);
        INDArray valid = storage.get(NDArrayIndex.interval(0, 3), NDArrayIndex.all());
        INDArray invalid = storage.get(NDArrayIndex.interval(3, 6), NDArrayIndex.all());

        INDArray x = activations();
        INDArray w = raw(DataType.UBYTE, nvfp4Weights(), 3, 16);
        INDArray second = scalar(.1003f);
        INDArray expected = run(true, x, w, valid, second);

        assertHostRejection(true, x, w, invalid, second);
        assertEquals(expected, run(true, x, w, valid, second));
    }

    private static INDArray run(boolean nv, INDArray x, INDArray w, INDArray scale, INDArray second) {
        DynamicCustomOp op = nv ? new ModelOptNvfp4Linear(x, w, scale, second, true)
                : new ModelOptFp8Linear(x, w, scale, second, true);
        Nd4j.exec(op);
        return op.outputArguments().get(0);
    }

    private static void assertHostRejection(boolean nv, INDArray x, INDArray w, INDArray scale, INDArray second) {
        RuntimeException e = assertThrows(RuntimeException.class, () -> run(nv, x, w, scale, second));
        StringBuilder messages = new StringBuilder();
        for (Throwable t = e; t != null; t = t.getCause()) messages.append(t.getMessage()).append('\n');
        assertTrue(messages.toString().contains(HOST_REJECTION),
                "invalid scale was not rejected by host validation before launch: " + messages);
    }

    private static INDArray activations() {
        float[] values = new float[6 * 32];
        for (int i = 0; i < values.length; i++) values[i] = ((i * 7) % 31 - 15) / 8.0f;
        return Nd4j.create(values, new long[]{6, 32}, DataType.FLOAT);
    }

    private static byte[] nvfp4Weights() {
        byte[] bytes = new byte[3 * 16];
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) ((i & 15) | (((i + 5) & 15) << 4));
        return bytes;
    }

    private static byte[] fp8Weights() {
        byte[] bytes = new byte[3 * 32];
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) ((0x28 + i % 32) | (i % 3 == 0 ? 128 : 0));
        return bytes;
    }

    /** Writes one byte of host storage in place and marks the host copy as the newest. */
    private static void overwrite(INDArray array, long index, byte value) {
        new BytePointer(array.data().pointer()).capacity(array.length()).put(index, value);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
    }

    /** Writes a FLOAT scalar's host storage in place and marks the host copy as the newest. */
    private static void overwrite(INDArray scalar, float value) {
        new FloatPointer(scalar.data().pointer()).capacity(1).put(0, value);
        Nd4j.getAffinityManager().tagLocation(scalar, AffinityManager.Location.HOST);
    }

    private static INDArray raw(DataType dtype, byte[] bytes, long... shape) {
        INDArray array = Nd4j.createUninitialized(dtype, shape, 'c');
        assertEquals(1, dtype.width(), "raw helper accepts only one-byte storage");
        assertEquals(bytes.length, array.length(), "raw payload must exactly fill its typed allocation");
        // Typed storage is filled byte for byte; no numeric conversion to FP8.
        new BytePointer(array.data().pointer()).capacity(bytes.length).put(bytes);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }

    private static INDArray scalar(float value) {
        INDArray result = Nd4j.scalar(DataType.FLOAT, value);
        Nd4j.getAffinityManager().tagLocation(result, AffinityManager.Location.HOST);
        return result;
    }
}
