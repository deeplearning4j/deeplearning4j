package org.nd4j.linalg.crash;

import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * This set of test launches different ops in different order, to check for possible data corruption cases
 *
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CrashTest extends BaseNd4jTest {
    public CrashTest(Nd4jBackend backend) {
        super(backend);
    }

    private static final int ITERATIONS = 100;
    private static final boolean[] paramsA = new boolean[] {true, false};
    private static final boolean[] paramsB = new boolean[] {true, false};

    @Test
    public void testArrays1() {
        INDArray x = Nd4j.create(1024, 64);
        INDArray y = Nd4j.create(64, 1024);

        for(int i = 0; i < ITERATIONS; i++) {
            op(x, y, i);
        }
    }

    @Test
    public void testViews1() {
        INDArray x = Nd4j.create(64, 1024, 64);
        INDArray y = Nd4j.create(64, 64, 1024);

        for(int i = 0; i < ITERATIONS; i++) {
            int slice = RandomUtils.nextInt(0, x.shape()[0]);
            op(x.tensorAlongDimension(slice, 1, 2), y.tensorAlongDimension(slice, 1, 2), i);
        }
    }

    protected void op(INDArray x, INDArray y, int i) {
        // broadcast along row & column
        INDArray row = Nd4j.ones(64);
        INDArray column = Nd4j.ones(1024, 1);

        x.addiRowVector(row);
        x.addiColumnVector(column);

        // casual scalar
        x.addi(i * 2);

        // reduction along all dimensions
        float sum = x.sumNumber().floatValue();

        // index reduction
        IMax imax = new IMax(x);
        Nd4j.getExecutioner().exec(imax, Integer.MAX_VALUE);
        int max = imax.getFinalResult();

        // casual transform
        Nd4j.getExecutioner().exec(new Sqrt(x, x));

        //  dup
        INDArray x1 = x.dup();
        INDArray x2 = x.dup();
        INDArray x3 = x.dup();
        INDArray x4 = x.dup();

        // vstack && hstack
        INDArray vstack = Nd4j.vstack(x1, x2, x3, x4);
        INDArray hstack = Nd4j.hstack(x1, x2, x3, x4);

        // reduce3 call
        Nd4j.getExecutioner().exec(new ManhattanDistance(x, x2));

        // flatten call
        INDArray flat = Nd4j.toFlattened(x, x1, x2, x3, x4);


        // reduction along dimension: row & column
        INDArray max_0 = x.max(0);
        INDArray max_1 = x.max(1);

        // index reduction along dimension: row & column
        INDArray imax_0 = Nd4j.argMax(x, 0);
        INDArray imax_1 = Nd4j.argMax(x, 1);

        // logisoftmax, softmax & softmax derivative
        Nd4j.getExecutioner().exec(new SoftMax(x));
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(x));
        Nd4j.getExecutioner().exec(new LogSoftMax(x));

        // BooleanIndexing
        BooleanIndexing.replaceWhere(x, 5f, Conditions.lessThan(8f));

        // assing on view
        BooleanIndexing.assignIf(x, x1, Conditions.greaterThan(-1000000000f));

        // blas call
        float dot = (float) Nd4j.getBlasWrapper().dot(x, x1);

        // mmul
        for (boolean tA : paramsA) {
            for (boolean tB : paramsB) {

                INDArray xT = tA ? x.dup() : x.dup().transpose();
                INDArray yT = tB ? y.dup() : y.dup().transpose();

                Nd4j.gemm(xT, yT, tA, tB);
            }
        }

        // specially for views, checking here without dup and rollover
        Nd4j.gemm(x, y, false, false);

        System.out.println("Iteration passed: " + i);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
