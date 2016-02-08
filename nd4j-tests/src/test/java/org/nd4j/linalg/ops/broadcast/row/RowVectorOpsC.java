package org.nd4j.linalg.ops.broadcast.row;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class RowVectorOpsC extends BaseNd4jTest {
    public RowVectorOpsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public RowVectorOpsC() {
    }

    public RowVectorOpsC(Nd4jBackend backend) {
        super(backend);
    }

    public RowVectorOpsC(String name) {
        super(name);
    }

    @Test
    public void testAddi() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        arr.addiRowVector(Nd4j.create(new double[]{1,2}));
        INDArray assertion  = Nd4j.create(new double[][]{
                {2,4},
                {4,6}
        });
        assertEquals(assertion,arr);
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
