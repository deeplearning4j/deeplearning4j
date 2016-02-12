package org.nd4j.linalg.cpu.javacpp;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Bias;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        Loop loop = new Loop();
        float[] add1 = new float[1000000];
        float[] add2 = new float[1];
        long start = System.nanoTime();
        loop.execFloatTransform(add1,3,0,0,1,1,"exp",add2,add1);
        long end = System.nanoTime();
        System.out.println((TimeUnit.MILLISECONDS.convert(Math.abs(end - start),TimeUnit.NANOSECONDS)));
        loop.execScalarFloat(add1,add1,add1.length,0,0,1,1,"div_scalar",new float[]{1});

    }

    @Test
    public void testCreate() {
        INDArray arr = Nd4j.create(new double[10]);
    }

    @Test
    public void testShape() {
        INDArray arr = Nd4j.create(new int[]{10});
        System.out.println(Arrays.toString(arr.shape()));
    }


    @Test
    public void testDup() {
        INDArray arr = Nd4j.ones(3);
        System.out.println(arr.dup());
    }


    @Test
    public void testSumWithRow2(){
        //All sums in this method execute without exceptions.
        INDArray array3d = Nd4j.ones(2,10,10);
        array3d.sum(0);
        array3d.sum(1);
        array3d.sum(2);

        INDArray array4d = Nd4j.ones(2, 10, 10, 10);
        array4d.sum(0);
        array4d.sum(1);
        array4d.sum(2);
        array4d.sum(3);

        INDArray array5d = Nd4j.ones(2, 10, 10, 10, 10);
        array5d.sum(0);
        array5d.sum(1);
        array5d.sum(2);
        array5d.sum(3);
        array5d.sum(4);
    }

    @Test
    public void testGetColumnFortran() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column = Nd4j.create(new float[]{1, 2});
        INDArray column2 = Nd4j.create(new float[]{3, 4});
        INDArray testColumn = n.getColumn(0);
        INDArray testColumn1 = n.getColumn(1);
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);

    }

    @Test
    public void testDescriptiveStats() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0, mean.currentResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(2.5, variance.currentResult().doubleValue(), 1e-1);
    }


    @Test
    public void testBias() {
        INDArray bias = Nd4j.linspace(1, 4, 4);
        Bias biaOp = new Bias(bias);
        Nd4j.getExecutioner().exec(biaOp);
        assertEquals(0.0,biaOp.currentResult().doubleValue(),1e-1);
    }


    @Test
    public void testLength() {
        INDArray values = Nd4j.create(2, 2);
        INDArray values2 = Nd4j.create(2, 2);

        values.put(0, 0, 0);
        values2.put(0, 0, 2);
        values.put(1, 0, 0);
        values2.put(1, 0, 2);
        values.put(0, 1, 0);
        values2.put(0, 1, 0);
        values.put(1, 1, 2);
        values2.put(1, 1, 2);


        INDArray expected = Nd4j.repeat(Nd4j.scalar(2), 2).reshape(2,1);

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }

}
