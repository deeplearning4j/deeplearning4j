package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
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
    public void testArrayRemoveIndexX() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{11});

        assertEquals(8, dst.length);
        assertEquals(1, dst[0]);
        assertEquals(8, dst[7]);
    }

    @Test
    public void testArrayRemoveIndex5() throws Exception {
        //INDArray arraySource = Nd4j.create(new float[]{1,2,3,4,5,6,7,8});
        int[] arraySource = new int[] {1,2,3,4,5,6,7,8};

        int[] dst = ArrayUtil.removeIndex(arraySource, new int[]{Integer.MAX_VALUE});

        assertEquals(8, dst.length);
        assertEquals(1, dst[0]);
        assertEquals(8, dst[7]);
    }
}
