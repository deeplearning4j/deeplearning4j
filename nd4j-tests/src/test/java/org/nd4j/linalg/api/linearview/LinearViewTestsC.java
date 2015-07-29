package org.nd4j.linalg.api.linearview;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class LinearViewTestsC extends BaseNd4jTest {
    public LinearViewTestsC() {
        super();
    }

    public LinearViewTestsC(String name) {
        super(name);
    }

    public LinearViewTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public LinearViewTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }


    @Test
    public void testMoreReshape() {
        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9,

                10, 11, 12}, new int[]{2, 6});


        INDArray ndv = nd.getRow(0);
        INDArray other = ndv.reshape(2, 3);

        INDArray otherLinear = other.linearView();
        assertEquals(ndv.linearView(),otherLinear);

        INDArray otherVec = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6});
        assertEquals(ndv,otherVec);
    }
}
