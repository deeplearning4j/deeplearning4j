package org.deeplearning4j.spark.parameterserver.accumulation;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingAccumulationFunctionTest {
    @Before
    public void setUp() throws Exception {}

    @Test
    public void testAccumulation1() throws Exception {
        INDArray updates1 = Nd4j.create(1000).assign(1.0);
        INDArray updates2 = Nd4j.create(1000).assign(2.0);
        INDArray expUpdates = Nd4j.create(1000).assign(3.0);

        SharedTrainingAccumulationTuple tuple1 = SharedTrainingAccumulationTuple.builder().updaterStateArray(updates1)
                        .scoreSum(1.0).aggregationsCount(1).build();

        SharedTrainingAccumulationTuple tuple2 = SharedTrainingAccumulationTuple.builder().updaterStateArray(updates2)
                        .scoreSum(2.0).aggregationsCount(1).build();

        SharedTrainingAccumulationFunction accumulationFunction = new SharedTrainingAccumulationFunction();

        SharedTrainingAccumulationTuple tupleE = accumulationFunction.call(null, tuple1);

        // testing null + tuple accumulation
        assertEquals(1, tupleE.getAggregationsCount());
        assertEquals(1.0, tupleE.getScoreSum(), 0.01);
        assertEquals(updates1, tupleE.getUpdaterStateArray());


        // testing tuple + tuple accumulation
        SharedTrainingAccumulationTuple tupleResult = accumulationFunction.call(tuple1, tuple2);
        assertEquals(2, tupleResult.getAggregationsCount());
        assertEquals(3.0, tupleResult.getScoreSum(), 0.01);
        assertEquals(expUpdates, tupleResult.getUpdaterStateArray());

    }
}
