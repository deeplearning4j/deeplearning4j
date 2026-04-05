package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterNdUpdate;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for the SmolDocling vision encoder crash on CPU.
 * The crash is SIGSEGV in CopyPws<long> during scatter_nd_update.
 */
public class VisionEncoderScatterIsolationTest {

    @Test
    public void testScatterNdUpdateTinyFloat() {
        // Same test with FLOAT32
        INDArray data = Nd4j.zeros(DataType.FLOAT, 1, 4);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 0}, {0, 1}, {0, 2}, {0, 3}});
        INDArray updates = Nd4j.createFromArray(new float[]{10, 20, 30, 40});

        System.out.println("FLOAT data shape: " + java.util.Arrays.toString(data.shape()));
        INDArray result = Nd4j.getExecutioner().exec(new ScatterNdUpdate(data, indices, updates))[0];
        assertNotNull(result);
        assertEquals(10f, result.getFloat(0, 0), 1e-5);
        System.out.println("FLOAT test passed!");
    }

    @Test
    public void testScatterNdUpdate1D() {
        // 1D output — indices must be [N, 1] for scatter_nd
        INDArray data = Nd4j.zeros(DataType.FLOAT, 4);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0}, {1}, {2}, {3}});
        INDArray updates = Nd4j.createFromArray(new float[]{10, 20, 30, 40});

        System.out.println("1D data shape: " + java.util.Arrays.toString(data.shape()));
        INDArray result = Nd4j.getExecutioner().exec(new ScatterNdUpdate(data, indices, updates))[0];
        assertNotNull(result);
        assertEquals(10f, result.getFloat(0), 1e-5);
        assertEquals(40f, result.getFloat(3), 1e-5);
        System.out.println("1D test passed!");
    }

    @Test
    public void testScatterNdUpdate2DRowSelect() {
        // 2D output with 1D indices (row selection) — like the TritonGraphBackendTest
        INDArray data = Nd4j.zeros(DataType.FLOAT, 4, 2);
        INDArray indices = Nd4j.createFromArray(new int[][]{{0}, {2}});
        INDArray updates = Nd4j.ones(DataType.FLOAT, 2, 2);

        System.out.println("2D-row data shape: " + java.util.Arrays.toString(data.shape()));
        INDArray result = Nd4j.getExecutioner().exec(new ScatterNdUpdate(data, indices, updates))[0];
        assertNotNull(result);
        assertEquals(1f, result.getFloat(0, 0), 1e-5);
        assertEquals(0f, result.getFloat(1, 0), 1e-5); // row 1 untouched
        assertEquals(1f, result.getFloat(2, 0), 1e-5);
        System.out.println("2D-row test passed!");
    }

    @Test
    public void testScatterNdUpdateTiny() {
        // Minimal case: [1, 4] data, 4 indices, 4 updates
        INDArray data = Nd4j.zeros(DataType.INT64, 1, 4);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 0}, {0, 1}, {0, 2}, {0, 3}});
        INDArray updates = Nd4j.createFromArray(new long[]{10, 20, 30, 40});

        System.out.println("data shape: " + java.util.Arrays.toString(data.shape()) + " buffer: " + data.data().length());
        System.out.println("indices shape: " + java.util.Arrays.toString(indices.shape()));
        System.out.println("updates shape: " + java.util.Arrays.toString(updates.shape()) + " buffer: " + updates.data().length());

        INDArray result = Nd4j.getExecutioner().exec(
                new ScatterNdUpdate(data, indices, updates))[0];
        assertNotNull(result);
        assertEquals(4, result.length());
        assertEquals(10, result.getLong(0, 0));
        assertEquals(40, result.getLong(0, 3));
    }

    @Test
    public void testScatterNdUpdateBasic() {
        INDArray data = Nd4j.zeros(DataType.INT64, 1, 1024);
        INDArray indices = Nd4j.zeros(DataType.INT64, 1024, 2);
        for (int i = 0; i < 1024; i++) {
            indices.putScalar(i, 0, 0);
            indices.putScalar(i, 1, i);
        }
        INDArray updates = Nd4j.arange(1024).castTo(DataType.INT64);

        System.out.println("data shape: " + java.util.Arrays.toString(data.shape()) + " buffer: " + data.data().length());
        System.out.println("indices shape: " + java.util.Arrays.toString(indices.shape()));
        System.out.println("updates shape: " + java.util.Arrays.toString(updates.shape()) + " buffer: " + updates.data().length());

        INDArray result = Nd4j.getExecutioner().exec(
                new ScatterNdUpdate(data, indices, updates))[0];
        assertNotNull(result);
        assertEquals(1024, result.length());
        assertEquals(0, result.getLong(0, 0));
        assertEquals(1, result.getLong(0, 1));
        assertEquals(1023, result.getLong(0, 1023));
    }

    @Test
    public void testCreateToScatterNdSameDiff() {
        SameDiff sd = SameDiff.create();

        SDVariable shapeTensor = sd.placeHolder("shape", DataType.INT64, 2);
        SDVariable data = sd.create("data", shapeTensor, DataType.INT64, "c", true);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, -1, 2);
        SDVariable updates = sd.placeHolder("updates", DataType.INT64, -1);
        SDVariable result = sd.scatterNdUpdate("result", data, indices, updates);

        INDArray shapeArr = Nd4j.createFromArray(new long[]{1, 1024});
        INDArray indicesArr = Nd4j.zeros(DataType.INT64, 1024, 2);
        for (int i = 0; i < 1024; i++) {
            indicesArr.putScalar(i, 0, 0);
            indicesArr.putScalar(i, 1, i);
        }
        INDArray updatesArr = Nd4j.arange(1024).castTo(DataType.INT64);

        Map<String, INDArray> out = sd.output(
                Map.of("shape", shapeArr, "indices", indicesArr, "updates", updatesArr),
                "result");

        INDArray res = out.get("result");
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 1024}, res.shape());
        assertEquals(0, res.getLong(0, 0));
        assertEquals(999, res.getLong(0, 999));
    }

    @Test
    public void testWhereToCreateToScatter() {
        SameDiff sd = SameDiff.create();

        SDVariable shapeRaw = sd.placeHolder("shape_raw", DataType.INT64, 2);
        SDVariable negOnes = sd.constant("neg_ones", Nd4j.createFromArray(new long[]{-1, -1}));
        SDVariable ones = sd.constant("ones", Nd4j.createFromArray(new long[]{1, 1}));

        SDVariable eq = sd.eq(shapeRaw, negOnes).castTo(DataType.BOOL);
        SDVariable shapeResolved = sd.where(ones, shapeRaw, eq);

        SDVariable data = sd.create("data", shapeResolved, DataType.INT64, "c", true);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, -1, 2);
        SDVariable updates = sd.placeHolder("updates", DataType.INT64, -1);
        SDVariable result = sd.scatterNdUpdate("result", data, indices, updates);

        INDArray shapeRawArr = Nd4j.createFromArray(new long[]{1, 1024});
        INDArray indicesArr = Nd4j.zeros(DataType.INT64, 1024, 2);
        for (int i = 0; i < 1024; i++) {
            indicesArr.putScalar(i, 0, 0);
            indicesArr.putScalar(i, 1, i);
        }
        INDArray updatesArr = Nd4j.arange(1024).castTo(DataType.INT64);

        Map<String, INDArray> out = sd.output(
                Map.of("shape_raw", shapeRawArr, "indices", indicesArr, "updates", updatesArr),
                "result");

        INDArray res = out.get("result");
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 1024}, res.shape());
    }

    @Test
    public void testReshapeNoCopyWithDynamicInput() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.INT64, -1, -1);
        SDVariable target = sd.constant("target", Nd4j.createFromArray(new long[]{-1}));
        SDVariable reshaped = sd.reshape(input, target);

        INDArray inputArr = Nd4j.arange(1024).castTo(DataType.INT64).reshape(1024, 1);
        Map<String, INDArray> out = sd.output(Map.of("input", inputArr), reshaped.name());

        INDArray res = out.get(reshaped.name());
        assertNotNull(res);
        assertEquals(1024, res.length());
        assertArrayEquals(new long[]{1024}, res.shape());
    }

    @Test
    public void testRepeatedCreateScatterExecution() {
        SameDiff sd = SameDiff.create();

        SDVariable shapeTensor = sd.placeHolder("shape", DataType.INT64, 2);
        SDVariable data = sd.create("data", shapeTensor, DataType.INT64, "c", true);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, -1, 2);
        SDVariable updates = sd.placeHolder("updates", DataType.INT64, -1);
        SDVariable result = sd.scatterNdUpdate("result", data, indices, updates);

        for (int iter = 0; iter < 5; iter++) {
            INDArray shapeArr = Nd4j.createFromArray(new long[]{1, 1024});
            INDArray indicesArr = Nd4j.zeros(DataType.INT64, 1024, 2);
            for (int i = 0; i < 1024; i++) {
                indicesArr.putScalar(i, 0, 0);
                indicesArr.putScalar(i, 1, i);
            }
            INDArray updatesArr = Nd4j.arange(1024).castTo(DataType.INT64).addi(iter * 1000);

            Map<String, INDArray> out = sd.output(
                    Map.of("shape", shapeArr, "indices", indicesArr, "updates", updatesArr),
                    "result");

            INDArray res = out.get("result");
            assertNotNull(res, "Iteration " + iter);
            assertArrayEquals(new long[]{1, 1024}, res.shape(), "Iteration " + iter);
            assertEquals(iter * 1000, res.getLong(0, 0), "Iteration " + iter);
        }
    }

    /**
     * Test the full chain matching the vision encoder graph exactly:
     * flatten_2d(axis=1) → flatten_2d(axis=2) → gather(axis=0) → reshape_no_copy([-1]) → scatter_nd_update
     */
    @Test
    public void testFullVisionEncoderPositionChain() {
        SameDiff sd = SameDiff.create();

        // Position grid [1, 32, 32] → reshape → [1, 1024]
        SDVariable posGrid = sd.placeHolder("pos_grid", DataType.INT64, 1, 32, 32);
        SDVariable flat1 = sd.reshape("flat1", posGrid, 1, 1024);

        // Reshape [1, 1024] → [1024, 1]
        SDVariable flat2 = sd.reshape("flat2", flat1, 1024, 1);

        // Gather with indices [1024] from flat2 [1024, 1] on axis=0
        SDVariable gatherIdx = sd.placeHolder("gather_idx", DataType.INT64, -1);
        SDVariable gathered = sd.gather("gathered", flat2, gatherIdx, 0);

        // Reshape to [-1]
        SDVariable reshapeTarget = sd.constant("reshape_target", Nd4j.createFromArray(new long[]{-1}));
        SDVariable reshaped = sd.reshape("reshaped", gathered, reshapeTarget);

        // Create target buffer [1, 1024] and scatter
        SDVariable shapeTensor = sd.placeHolder("scatter_shape", DataType.INT64, 2);
        SDVariable scatterData = sd.create("scatter_data", shapeTensor, DataType.INT64, "c", true);
        SDVariable scatterIdx = sd.placeHolder("scatter_idx", DataType.INT64, -1, 2);
        SDVariable result = sd.scatterNdUpdate("result", scatterData, scatterIdx, reshaped);

        // Execute
        INDArray posGridArr = Nd4j.arange(1024).castTo(DataType.INT64).reshape(1, 32, 32);
        INDArray gatherIdxArr = Nd4j.arange(1024).castTo(DataType.INT64);
        INDArray scatterShapeArr = Nd4j.createFromArray(new long[]{1, 1024});
        INDArray scatterIdxArr = Nd4j.zeros(DataType.INT64, 1024, 2);
        for (int i = 0; i < 1024; i++) {
            scatterIdxArr.putScalar(i, 0, 0);
            scatterIdxArr.putScalar(i, 1, i);
        }

        Map<String, INDArray> out = sd.output(
                Map.of("pos_grid", posGridArr, "gather_idx", gatherIdxArr,
                       "scatter_shape", scatterShapeArr, "scatter_idx", scatterIdxArr),
                "result");

        INDArray res = out.get("result");
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 1024}, res.shape());
        assertEquals(0, res.getLong(0, 0));
        assertEquals(1023, res.getLong(0, 1023));
    }
}
