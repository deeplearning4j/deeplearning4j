package org.deeplearning4j.spark.parameterserver.accumulation;

import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingAggregateFunctionTest {
    @Before
    public void setUp() throws Exception {
        //
    }

    @Test
    public void testAggregate1() throws Exception {
        INDArray updates1 = Nd4j.create(1000).assign(1.0);
        INDArray updates2 = Nd4j.create(1000).assign(2.0);
        INDArray expUpdates = Nd4j.create(1000).assign(3.0);

        SharedTrainingResult result1 = SharedTrainingResult.builder().updaterStateArray(updates1).aggregationsCount(1)
                        .scoreSum(1.0).build();

        SharedTrainingResult result2 = SharedTrainingResult.builder().updaterStateArray(updates2).aggregationsCount(1)
                        .scoreSum(2.0).build();

        // testing null + result
        SharedTrainingAggregateFunction aggregateFunction = new SharedTrainingAggregateFunction();
        SharedTrainingAccumulationTuple tuple1 = aggregateFunction.call(null, result1);


        // testing tuple + result
        SharedTrainingAccumulationTuple tuple2 = aggregateFunction.call(tuple1, result2);


        // testing final result
        assertEquals(2, tuple2.getAggregationsCount());
        assertEquals(3.0, tuple2.getScoreSum(), 0.001);
        assertEquals(expUpdates, tuple2.getUpdaterStateArray());
    }
}
