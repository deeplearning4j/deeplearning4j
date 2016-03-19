package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
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
}
