package org.nd4j.linalg.ops.transforms;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Created by agibsonccc on 9/6/14.
 */
public abstract class TransformTests {



    @Test
    public void testPooling() {
        INDArray twoByTwo = Nd4j.ones(new int[]{2,2,2});
        INDArray pool = Transforms.pool(twoByTwo,new int[]{1,2});

    }






}
