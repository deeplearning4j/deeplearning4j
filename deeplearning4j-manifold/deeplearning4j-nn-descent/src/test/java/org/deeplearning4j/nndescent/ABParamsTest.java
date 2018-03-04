package org.deeplearning4j.nndescent;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;

import static org.junit.Assert.assertEquals;

public class ABParamsTest {
    @Test
    public void testAbParams() {
        INDArray[] abParams = ABParams.builder()
                .minDistance(1.0).spread(1.0).build().solve();
        double[] assertions = {0.11497568308423263, 1.9292371454400927};
        assertEquals(assertions[0],abParams[0].getDouble(0),1e-3);
        assertEquals(assertions[1],abParams[1].getDouble(0),1e-3);


    }

}
