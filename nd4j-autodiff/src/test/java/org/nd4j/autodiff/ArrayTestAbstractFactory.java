package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.autodiff.DifferentialFunction;
import org.nd4j.autodiff.autodiff.DifferentialFunctionFactory;
import org.nd4j.autodiff.autodiff.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ArrayTestAbstractFactory
        extends AbstractFactoriesTest<ArrayField> {

    private static final double EQUAL_DELTA = 1e-12;

    public ArrayTestAbstractFactory() {
        super(EQUAL_DELTA);
    }

    @Override
    protected AbstractFactory<ArrayField> getFactory() {
        return ArrayFactory.instance();
    }


    @Test
    public void testAutoDiff() {
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(ArrayFactory.instance());
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x", new ArrayField(Nd4j.scalar(1.0)));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(Nd4j.scalar(1.0)));
        DifferentialFunction<ArrayField> h = x.mul(x).mul( arrayFieldDifferentialFunctionFactory.cos( x.mul(y) ).plus(y) );
        System.out.println(h.diff(x));

    }

}
