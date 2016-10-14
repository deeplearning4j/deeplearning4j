package org.nd4j.linalg.aggregates;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateAxpy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class AggregatesTests extends BaseNd4jTest {

    public AggregatesTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testAggregate1() throws Exception {
        INDArray arrayX = Nd4j.ones(10);
        INDArray arrayY = Nd4j.zeros(10);

        INDArray exp1 = Nd4j.ones(10);

        AggregateAxpy axpy = new AggregateAxpy(arrayX, arrayY, 1.0f);

        Nd4j.getExecutioner().exec(axpy);

        assertEquals(exp1, arrayY);
    }


    @Test
    public void testBatchedAggregate1() throws Exception {
        INDArray arrayX1 = Nd4j.ones(10);
        INDArray arrayY1 = Nd4j.zeros(10);

        INDArray arrayX2 = Nd4j.ones(10);
        INDArray arrayY2 = Nd4j.zeros(10);

        INDArray exp1 = Nd4j.create(10).assign(1f);
        INDArray exp2 = Nd4j.create(10).assign(1f);

        AggregateAxpy axpy1 = new AggregateAxpy(arrayX1, arrayY1, 1.0f);
        AggregateAxpy axpy2 = new AggregateAxpy(arrayX2, arrayY2, 1.0f);

        Batch batch = new Batch();
        batch.enqueueAggregate(axpy1);
        batch.enqueueAggregate(axpy2);

        Nd4j.getExecutioner().exec(batch);

        assertEquals(exp1, arrayY1);
        assertEquals(exp2, arrayY2);
    }

    @Test
    public void testBatchedAggregate2() throws Exception {
        INDArray arrayX1 = Nd4j.ones(10);
        INDArray arrayY1 = Nd4j.zeros(10).assign(2.0f);

        INDArray arrayX2 = Nd4j.ones(10);
        INDArray arrayY2 = Nd4j.zeros(10).assign(2.0f);

        INDArray arrayX3 = Nd4j.ones(10);
        INDArray arrayY3 = Nd4j.ones(10);

        INDArray exp1 = Nd4j.create(10).assign(4f);
        INDArray exp2 = Nd4j.create(10).assign(3f);
        INDArray exp3 = Nd4j.create(10).assign(3f);

        AggregateAxpy axpy1 = new AggregateAxpy(arrayX1, arrayY1, 2.0f);
        AggregateAxpy axpy2 = new AggregateAxpy(arrayX2, arrayY2, 1.0f);
        AggregateAxpy axpy3 = new AggregateAxpy(arrayX3, arrayY3, 2.0f);

        Batch batch = new Batch();
        batch.enqueueAggregate(axpy1);
        batch.enqueueAggregate(axpy2);
        batch.enqueueAggregate(axpy3);

        Nd4j.getExecutioner().exec(batch);

        assertEquals(exp1, arrayY1);
        assertEquals(exp2, arrayY2);
        assertEquals(exp3, arrayY3);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
