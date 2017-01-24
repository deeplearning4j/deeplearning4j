package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaBlasTests {

    @Test
    public void testMMuli1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 250, 250).reshape(new int[]{5, 50});

        System.out.println("array1: " + array1);

        INDArray array2 = Nd4j.linspace(1, 250, 250).reshape(new int[]{50, 5});

        System.out.println("array2: " + array2);

        INDArray result = Nd4j.create(new int[]{5, 5});

        System.out.println("Order1: " + array1.ordering());
        System.out.println("Order2: " + array2.ordering());
        System.out.println("Result order: " + result.ordering());

        array1.mmul(array2, result);

        System.out.println("Result: " + result);

        Thread.sleep(100000000000L);
    }

    @Test
    public void testDup1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 250, 250).reshape(new int[]{5, 50}).dup('f');

        INDArray array2 = array1.dup();

        System.out.println("array1 ordering: " + array1.ordering());  //  F
        System.out.println("array2 ordering: " + array2.ordering()); //  C
        System.out.println("array1 eq array2: " + array1.equals(array2)); // true

        //assertEquals(array1.getFloat(17), array2.getFloat(17), 0.001f );
    }

    @Test
    public void testForAlex() throws Exception {
        int[][] shape1s = new int[][]{{10240, 10240}};
        int[][] shape2s = new int[][]{{10240, 10240}};

        int[] nTestsArr = new int[]{5};

        for(int test=0; test<shape1s.length; test++ ) {

            int[] shape1 = shape1s[test];
            int[] shape2 = shape2s[test];

            int nTests = nTestsArr[test];

            INDArray c1 = Nd4j.create(shape1, 'c');
            INDArray c2 = Nd4j.create(shape2, 'c');

            CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

            AtomicAllocator.getInstance().getPointer(c1, context);
            AtomicAllocator.getInstance().getPointer(c2, context);

            //CC
            long startCC = System.currentTimeMillis();
            for (int i = 0; i < nTests; i++) {
                c1.mmul(c2);
            }
            long endCC = System.currentTimeMillis();
            System.out.println("cc");


            System.out.println("mmul: " + Arrays.toString(shape1) + "x" + Arrays.toString(shape2) + ", " + nTests + " runs");
            System.out.println("cc: " + (endCC - startCC));
        }
    }
}
