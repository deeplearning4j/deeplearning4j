package org.nd4j.linalg.shape;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Test;

import org.apache.commons.math3.util.Pair;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class TADTests extends BaseNd4jTest {

    public TADTests(Nd4jBackend backend) {
        super(backend);
    }


    /**
     * This test checks for TADs equality between Java & native
     *
     * @throws Exception
     */
    @Test
    public void testEquality1() throws Exception {

        char[] order = new char[]{'c','f'};
        int[] dim_e = new int[]{0, 2};
        int[] dim_x = new int[]{1, 3};

        List<int[]> dim_3 = Arrays.asList(new int[]{0, 2, 3}, new int[]{0, 1, 2}, new int[]{1, 2, 3}, new int[]{0, 1, 3});


        for (char o: order) {
            INDArray array = Nd4j.create(new int[]{3, 5, 7, 9}, o);
            for (int e : dim_e) {
                for (int x : dim_x) {

                    int[] shape = new int[]{e, x};
                    Arrays.sort(shape);

                    DataBuffer tadShape_N = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(array, shape).getFirst();
                    DataBuffer tadShape_J = array.tensorAlongDimension(0, shape).shapeInfoDataBuffer();

                    log.info("Original order: {}; Dimensions: {}; Original shape: {};", o, Arrays.toString(shape), Arrays.toString(array.shapeInfoDataBuffer().asInt()));
                    log.info("Java shape: {}; Native shape: {}", Arrays.toString(tadShape_J.asInt()), Arrays.toString(tadShape_N.asInt()));
                    System.out.println();

                    assertEquals(true, compareShapes(tadShape_N, tadShape_J));
                }
            }
        }

        log.info("3D TADs:");
        for(char o: order) {
            INDArray array = Nd4j.create(new int[]{9, 7, 5, 3}, o);
            for (int[] shape: dim_3) {
                Arrays.sort(shape);

                DataBuffer tadShape_N = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(array, shape).getFirst();
                DataBuffer tadShape_J = array.tensorAlongDimension(0, shape).shapeInfoDataBuffer();

                log.info("Original order: {}; Dimensions: {}; Original shape: {};", o, Arrays.toString(shape), Arrays.toString(array.shapeInfoDataBuffer().asInt()));
                log.info("Java shape: {}; Native shape: {}", Arrays.toString(tadShape_J.asInt()), Arrays.toString(tadShape_N.asInt()));
                System.out.println();

                assertEquals(true, compareShapes(tadShape_N, tadShape_J));
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * this method compares rank, shape and stride for two given shapeBuffers
     * @param shapeA
     * @param shapeB
     * @return
     */
    protected boolean compareShapes(@NonNull DataBuffer shapeA, @NonNull DataBuffer shapeB) {
        if (shapeA.dataType() != DataBuffer.Type.INT)
            throw new IllegalStateException("ShapeBuffer should have dataType of INT");

        if (shapeA.dataType() != shapeB.dataType())
            return false;

        int rank = shapeA.getInt(0);
        if (rank != shapeB.getInt(0))
            return false;

        for (int e = 1; e <= rank * 2; e++) {
            if (shapeA.getInt(e) != shapeB.getInt(e))
                return false;
        }

        return true;
    }
}
