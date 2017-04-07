package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.autodiff.DifferentialFunction;
import org.nd4j.autodiff.autodiff.DifferentialFunctionFactory;
import org.nd4j.autodiff.autodiff.Variable;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;

import static org.junit.Assert.assertEquals;

public class ArrayTestAbstractFactory
        extends AbstractFactoriesTest<ArrayField> {

    private static final double EQUAL_DELTA = 1e-12;

    public ArrayTestAbstractFactory() {
        super(EQUAL_DELTA);
    }

    @Override
    protected AbstractFactory<ArrayField> getFactory() {
        return new ArrayFactory();
    }


    @Test
    public void testAutoDiff() {
        Graph<NDArrayInformation,OpState> graph = new Graph<>();
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
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
        System.out.println(h.diff(x).getValue().getOps());

    }


    @Test
    public void testAutoDiffSimple() {
        Graph<NDArrayInformation,OpState> graph = new Graph<>();
        ArrayFactory arrayFactory = new ArrayFactory(graph);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);

        //2 * x
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, graph));
        DifferentialFunction<ArrayField> h = x.mul(x);
        //x and result are the vertices
        assertEquals(2,graph.numVertices());
        //x * x - edges for only 1 vertex
        assertEquals(1,graph.getEdges().size());
        //2 edges
        assertEquals(2,graph.getEdges().get(0).size());
        System.out.println("Pre graph " + graph);
        // for(int i = 0; i < 8; i++)
        System.out.println(h.diff(x));
        System.out.println(graph);
    }


}
