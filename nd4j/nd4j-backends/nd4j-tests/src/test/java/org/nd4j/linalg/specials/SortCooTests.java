package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class SortCooTests extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public SortCooTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @After
    public void setDown() {
        Nd4j.setDataType(initialType);
    }

    @Test
    public void sortSparseCooIndicesSort1() throws Exception {
        // FIXME: we don't want this test running on cuda for now
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        val indices = new long[] {
                1, 0, 0,
                0, 1, 1,
                0, 1, 0,
                1, 1, 1};

        // we don't care about
        double values[] = new double[] {2, 1, 0, 3};
        val expIndices = new long[] {
                0, 1, 0,
                0, 1, 1,
                1, 0, 0,
                1, 1, 1};
        double expValues[] = new double[] {0, 1, 2, 3};

        DataBuffer idx = Nd4j.getDataBufferFactory().createLong(indices);
        DataBuffer val = Nd4j.createBuffer(values);

        log.info("Old indices: {}", Arrays.toString(idx.asInt()));

        NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesFloat(null, (LongPointer) idx.addressPointer(),
                        (FloatPointer) val.addressPointer(), 4, 3);


        log.info("New indices: {}", Arrays.toString(idx.asInt()));

        assertArrayEquals(expIndices, idx.asInt());
        assertArrayEquals(expValues, val.asDouble(), 1e-5);
    }

    @Test
    public void sortSparseCooIndicesSort2() throws Exception {
        // FIXME: we don't want this test running on cuda for now
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        val indices = new long[] {
                0, 0, 0,
                2, 2, 2,
                1, 1, 1};

        // we don't care about
        double values[] = new double[] {2, 1, 3};
        val expIndices = new long[] {
                0, 0, 0,
                1, 1, 1,
                2, 2, 2};
        double expValues[] = new double[] {2, 3, 1};

        DataBuffer idx = Nd4j.getDataBufferFactory().createLong(indices);
        DataBuffer val = Nd4j.createBuffer(values);

        NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesFloat(null, (LongPointer) idx.addressPointer(),
                        (FloatPointer) val.addressPointer(), 3, 3);

        assertArrayEquals(expIndices, idx.asInt());
        assertArrayEquals(expValues, val.asDouble(), 1e-5);
    }

    @Test
    public void sortSparseCooIndicesSort3() throws Exception {
        // FIXME: we don't want this test running on cuda for now
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        Random rng = Nd4j.getRandom();
        rng.setSeed(12040483421383L);
        long shape[] = {50,50,50};
        int nnz = 100;
        val indices = Nd4j.rand(new int[]{nnz, shape.length}, rng).muli(50).ravel().toLongVector();
        val values = Nd4j.rand(new long[]{nnz}).ravel().toDoubleVector();


        DataBuffer indiceBuffer = Nd4j.getDataBufferFactory().createLong(indices);
        DataBuffer valueBuffer = Nd4j.createBuffer(values);
        INDArray indMatrix = Nd4j.create(indiceBuffer).reshape(new long[]{nnz, shape.length});

        NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesFloat(null, (LongPointer) indiceBuffer.addressPointer(),
                (FloatPointer) valueBuffer.addressPointer(), nnz, 3);

        for (long i = 1; i < nnz; ++i){
            for(long j = 0; j < shape.length; ++j){
                long prev = indiceBuffer.getLong(((i - 1) * shape.length + j));
                long current = indiceBuffer.getLong((i * shape.length + j));
                if (prev < current){
                    break;
                } else if(prev > current){
                    long[] prevRow = indiceBuffer.getLongsAt((i - 1) * shape.length, shape.length);
                    long[] currentRow = indiceBuffer.getLongsAt(i * shape.length, shape.length);
                    throw new AssertionError(String.format("indices are not correctly sorted between element %d and %d. %s > %s",
                            i - 1, i, Arrays.toString(prevRow), Arrays.toString(currentRow)));
                }
            }

        }
    }

    @Test
    public void sortSparseCooIndicesSort4() throws Exception {
        // FIXME: we don't want this test running on cuda for now
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        val indices = new long[] {
                0,2,7,
                2,36,35,
                3,30,17,
                5,12,22,
                5,43,45,
                6,32,11,
                8,8,32,
                9,29,11,
                5,11,22,
                15,26,16,
                17,48,49,
                24,28,31,
                26,6,23,
                31,21,31,
                35,46,45,
                37,13,14,
                6,38,18,
                7,28,20,
                8,29,39,
                8,32,30,
                9,42,43,
                11,15,18,
                13,18,45,
                29,26,39,
                30,8,25,
                42,31,24,
                28,33,5,
                31,27,1,
                35,43,26,
                36,8,37,
                39,22,14,
                39,24,42,
                42,48,2,
                43,26,48,
                44,23,49,
                45,18,34,
                46,28,5,
                46,32,17,
                48,34,44,
                49,38,39,
        };

        val expIndices = new long[] {
                0, 2, 7,
                2, 36, 35,
                3, 30, 17,
                5, 11, 22,
                5, 12, 22,
                5, 43, 45,
                6, 32, 11,
                6, 38, 18,
                7, 28, 20,
                8, 8, 32,
                8, 29, 39,
                8, 32, 30,
                9, 29, 11,
                9, 42, 43,
                11, 15, 18,
                13, 18, 45,
                15, 26, 16,
                17, 48, 49,
                24, 28, 31,
                26, 6, 23,
                28, 33, 5,
                29, 26, 39,
                30, 8, 25,
                31, 21, 31,
                31, 27, 1,
                35, 43, 26,
                35, 46, 45,
                36, 8, 37,
                37, 13, 14,
                39, 22, 14,
                39, 24, 42,
                42, 31, 24,
                42, 48, 2,
                43, 26, 48,
                44, 23, 49,
                45, 18, 34,
                46, 28, 5,
                46, 32, 17,
                48, 34, 44,
                49, 38, 39,
        };

        double values[] = new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};

        DataBuffer idx = Nd4j.getDataBufferFactory().createLong(indices);
        DataBuffer val = Nd4j.createBuffer(values);

        NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesFloat(null, (LongPointer) idx.addressPointer(),
                (FloatPointer) val.addressPointer(), 40, 3);

        // just check the indices. sortSparseCooIndicesSort1 and sortSparseCooIndicesSort2 checks that
        // indices and values are both swapped. This test just makes sure index sort works for larger arrays.
        assertArrayEquals(expIndices, idx.asInt());
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
