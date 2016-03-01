package org.nd4j.linalg.shape.ones;
import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class LeadingAndTrailingOnesC extends BaseNd4jTest  {

    public LeadingAndTrailingOnesC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testCreateLeadingAndTrailingOnes() {
        INDArray arr = Nd4j.create(1, 10, 1, 1);
        arr.assign(1);
        System.out.println(arr);
    }

    @Test
    public void testMatrix() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray slice1 = arr.slice(1);
        System.out.println(arr.slice(1));
        INDArray oneInMiddle = Nd4j.linspace(1,4,4).reshape(2,1,2);
        INDArray otherSlice = oneInMiddle.slice(1);
        assertEquals(2, otherSlice.offset());
        System.out.println(otherSlice);
        INDArray twoOnesInMiddle = Nd4j.linspace(1,4,4).reshape(2,1,1,2);
        INDArray sub = twoOnesInMiddle.get(NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all());
        assertEquals(2,sub.offset());

    }

    @Test
    public void testMultipleOnesInMiddle() {
        INDArray tensor = Nd4j.linspace(1,144,144).reshape(2, 2, 1, 1, 6, 6);
        INDArray tensorSlice1 = tensor.slice(1);
        INDArray tensorSlice1Slice1 = tensorSlice1.slice(1);
        System.out.println(tensor);
    }

    @Test
    public void testOnesInMiddleTensor() {
        INDArray im2colAssertion = Nd4j.create(new double[]{
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 0.0, 0.0, 0.0, 0.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        }, new int[]{2, 2, 1, 1, 6, 6});
        System.out.println(im2colAssertion);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
