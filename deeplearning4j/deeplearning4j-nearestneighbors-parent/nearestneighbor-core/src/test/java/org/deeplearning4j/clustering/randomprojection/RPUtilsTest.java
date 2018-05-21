package org.deeplearning4j.clustering.randomprojection;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class RPUtilsTest {

    @Test
    public void testDistanceComputeBatch() {
        INDArray x = Nd4j.linspace(1,4,4);
        INDArray y = Nd4j.linspace(1,16,16).reshape(4,4);
        INDArray result = Nd4j.create(4);
        INDArray distances = RPUtils.computeDistanceMulti("euclidean",x,y,result);
        INDArray scalarResult = Nd4j.scalar(1.0);
        for(int i = 0; i < result.length(); i++) {
            assertEquals(RPUtils.computeDistance("euclidean",x,y.slice(i),scalarResult),distances.getDouble(i),1e-3);
        }
    }

}
