package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.autodiff.DifferentialFunction;
import org.nd4j.autodiff.autodiff.DifferentialFunctionFactory;
import org.nd4j.autodiff.autodiff.Variable;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
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
        Graph<NDArrayInformation,OpState> graph = new Graph<>();
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayInformation yInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("y").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);
        NDArrayVertex arrayVertex = new NDArrayVertex(1,yInfo);

        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, graph));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(arrayVertex, graph));
        DifferentialFunction<ArrayField> h = x.mul(x).mul( arrayFieldDifferentialFunctionFactory.cos(x.mul(y) ).plus(y));
        System.out.println(h.diff(x).getValue().getOps().numVertices());

    }

}
