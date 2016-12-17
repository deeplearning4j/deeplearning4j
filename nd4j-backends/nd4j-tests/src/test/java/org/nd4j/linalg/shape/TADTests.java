package org.nd4j.linalg.shape;

import lombok.extern.slf4j.Slf4j;
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
            }
        }
    }


    @Test
    public void testNavigation1() throws Exception {
        INDArray array = Nd4j.linspace(1, 945, 945).reshape('c', new int[]{3, 5, 7, 9});

        int numTensors = array.tensorssAlongDimension(1,2);
        log.info("tensor shapeInfo: {}", Arrays.toString(array.tensorAlongDimension(0, 1, 2).shapeInfoDataBuffer().asInt()));
        for (int t = 0; t < 2; t++) {
            INDArray tensor = array.tensorAlongDimension(t, 1, 2);

            log.info("Tensor {}:\n{}", t, Arrays.toString(tensor.dup().data().asFloat()));
        }

        INDArray bc02 = Nd4j.create(5,7);

        Nd4j.getExecutioner().exec(new BroadcastMulOp(array, bc02, array.dup(array.ordering()), 1, 2));
    }

    @Test
    public void testNavigation2() throws Exception {
        INDArray array = Nd4j.linspace(1, 945, 945).reshape('c', new int[]{3, 5, 7, 9}).dup('f');

        int []shape = new int[]{1, 3};

        int numTensors = array.tensorssAlongDimension(shape);
        log.info("tensor shapeInfo: {}", Arrays.toString(array.tensorAlongDimension(0, shape).shapeInfoDataBuffer().asInt()));
        for (int t = 0; t < 2; t++) {
            INDArray tensor = array.tensorAlongDimension(t, shape);

            log.info("Tensor {}:\n{}", t, tensor);
            log.info("Tensor linear {}", Arrays.toString(tensor.dup(tensor.ordering()).data().asFloat()));
        }

        INDArray bc02 = Nd4j.create(shape[0],shape[1]);

        Nd4j.getExecutioner().exec(new BroadcastMulOp(array, bc02, array.dup(array.ordering()), shape));
    }

    @Test
    public void testNavigation3() throws Exception {
        INDArray array = Nd4j.linspace(1, 60, 60).reshape('c', new int[]{3, 4, 5}).dup('f');

        int []shape = new int[]{0, 1};

        int numTensors = array.tensorssAlongDimension(shape);
        log.info("tensor shapeInfo: {}", Arrays.toString(array.tensorAlongDimension(0, shape).shapeInfoDataBuffer().asInt()));
        for (int t = 0; t < 2; t++) {
            INDArray tensor = array.tensorAlongDimension(t, shape);

            log.info("Tensor {}:\n{}", t, tensor);
            log.info("linear: {}", Arrays.toString(tensor.dup(tensor.ordering()).data().asFloat()));
        }

        INDArray bc02 = Nd4j.linspace(1, 12, 12).reshape('c', new int[]{3,4}).dup('c');
        log.info("bc: {}", Arrays.toString(bc02.data().asFloat()));

        Nd4j.getExecutioner().exec(new BroadcastMulOp(array, bc02, array.dup(array.ordering()), shape));
    }

    @Test
    public void testNavigation4() throws Exception {
        INDArray arrOrig = Nd4j.ones(3,4,5,6).dup('c');
        INDArray bc13 = Nd4j.create(new double[][]{
                {1,1,1,1,1},
                {0,1,1,1,1},
                {1,0,0,1,1},
                {1,1,1,0,0}}).dup('c');

        INDArray result13 = arrOrig.dup('c');
        Nd4j.getExecutioner().exec(new BroadcastMulOp(result13,bc13,result13, 1, 3));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
