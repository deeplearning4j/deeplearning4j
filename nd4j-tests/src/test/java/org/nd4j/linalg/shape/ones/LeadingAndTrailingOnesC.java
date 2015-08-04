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


    @Override
    public char ordering() {
        return 'c';
    }
}
