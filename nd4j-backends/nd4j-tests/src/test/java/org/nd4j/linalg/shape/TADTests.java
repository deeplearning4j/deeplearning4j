package org.nd4j.linalg.shape;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import org.apache.commons.math3.util.Pair;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

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
        int[] dim_e = new int[]{0, 1};
        int[] dim_x = new int[]{2, 3};


        for (char o: order) {
            INDArray array = Nd4j.create(new int[]{3, 5, 7, 9}, o);
            for (int e : dim_e) {
                for (int x : dim_x) {

                    DataBuffer tadShape_N = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(array, new int[]{e, x}).getFirst();
                    DataBuffer tadShape_J = array.tensorAlongDimension(0, e, x).shapeInfoDataBuffer();

                    log.info("Original order: {}; Dimensions: {}; Original shape: {};", o, Arrays.toString(new int[]{e, x}), Arrays.toString(array.shapeInfoDataBuffer().asInt()));
                    log.info("Java shape: {}; Native shape: {}", Arrays.toString(tadShape_J.asInt()), Arrays.toString(tadShape_N.asInt()));
                    System.out.println();
                }
            }
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
