package jcuda.jcublas.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
@Slf4j
public class ArrayUtilsTests {

    @Test
    public void testArrayRemoveIndex1() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{0,1});

        assertEquals(6, dst.length);
        assertEquals(3, dst[0]);
    }

    @Test
    public void testArrayRemoveIndex2() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{0,7});

        assertEquals(6, dst.length);
        assertEquals(2, dst[0]);
        assertEquals(7, dst[5]);
    }

    @Test
    public void testArrayRemoveIndex4() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{0});

        assertEquals(7, dst.length);
        assertEquals(2, dst[0]);
        assertEquals(8, dst[6]);
    }

    @Test
    @Ignore
    public void testArrayRemoveIndexX() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{11});

        assertEquals(8, dst.length);
        assertEquals(1, dst[0]);
        assertEquals(8, dst[7]);
    }

    @Test
    @Ignore
    public void testArrayRemoveIndex5() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{Integer.MAX_VALUE});

        assertEquals(8, dst.length);
        assertEquals(1, dst[0]);
        assertEquals(8, dst[7]);
    }

    @Test
    public void testArrayFlatten1() {
        INDArray arrayC = Nd4j.create(new double[][]{{3, 5}, {4, 6}}, 'c');
        INDArray arrayF = Nd4j.create(new double[][]{{3, 5}, {4, 6}}, 'f');

        System.out.println("C: " + Arrays.toString(arrayC.data().asFloat()));
        System.out.println("F: " + Arrays.toString(arrayF.data().asFloat()));

        assertEquals(arrayC, arrayF);

    }


    @Test
    public void testInterleavedVector1() {
        int[] vector = ArrayUtil.buildInterleavedVector(new Random(), 11);

        log.error("Vector: {}", vector);
    }

    @Test
    public void testHalfVector1() {
        int[] vector = ArrayUtil.buildHalfVector(new Random(), 12);

        log.error("Vector: {}", vector);
    }
}
