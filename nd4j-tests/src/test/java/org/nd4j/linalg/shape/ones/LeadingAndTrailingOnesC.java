package org.nd4j.linalg.shape.ones;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class LeadingAndTrailingOnesC extends BaseNd4jTest  {
    public LeadingAndTrailingOnesC() {
    }

    public LeadingAndTrailingOnesC(String name) {
        super(name);
    }

    public LeadingAndTrailingOnesC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

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
        assertEquals(2,otherSlice.offset());
        System.out.println(otherSlice);
    }

    @Test
    public void testMultipleOnesInMiddle() {
        INDArray tensor = Nd4j.linspace(1,144,144).reshape(2,2,1,1,6,6);
        INDArray tensorSlice1 = tensor.slice(1);
        INDArray tensorSlice1Slice1 = tensorSlice1.slice(1);
        System.out.println(tensor);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
