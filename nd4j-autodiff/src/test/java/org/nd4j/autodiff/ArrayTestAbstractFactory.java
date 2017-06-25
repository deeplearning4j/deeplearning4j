package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.functions.Constant;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecOrder;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.autodiff.tensorgrad.TensorGradGraph;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

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
        TensorGradGraph graph = new TensorGradGraph();
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph, arrayFactory);
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
        NDArrayVertex xVertex = new NDArrayVertex(0, xInfo);
        NDArrayVertex arrayVertex = new NDArrayVertex(1, yInfo);

        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x", new ArrayField(xVertex, graph));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(arrayVertex, graph));
        DifferentialFunction<ArrayField> h = x.mul(x).mul(arrayFieldDifferentialFunctionFactory.cos(x.mul(y)).add(y));

        DifferentialFunction<ArrayField> diff = h.diff(x);
        ArrayField value = diff.getValue();
        Graph<NDArrayInformation, OpState> ops = value.getOps();
        String s = ops.toString();

        assertNotNull(s);
    }

    @Test
    public void testVariables() {
       TensorGradGraph graph = new TensorGradGraph();
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, graph));
        DifferentialFunction<ArrayField> h = x.mul(x);
        System.out.println(h.diff(x).getValue(false).getClass());

    }


    @Test
    public void testPairWiseOp() throws Exception {
       TensorGradGraph graph = new TensorGradGraph();
        ArrayFactory arrayFactory = new ArrayFactory(graph);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);

        NDArrayInformation xInfo = NDArrayInformation.builder().shape(new int[]{1,1}).id("x").build();
        NDArrayInformation yInfo = NDArrayInformation.builder().shape(new int[]{1, 1}).id("y").build();

        NDArrayVertex xVertex = new NDArrayVertex(graph.nextVertexId(), xInfo);
        NDArrayVertex yVertex = new NDArrayVertex(graph.nextVertexId(), yInfo);

        //2 * x
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, graph));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(yVertex, graph));
        DifferentialFunction<ArrayField> h = x.mul(y);
        System.out.println(h.getFormula(new ArrayList<>()));
        //x, x as the duplicate input and result are the vertices
        assertEquals(3,graph.numVertices());
        //x * x - edges for only 1 vertex and 1 duplicate
        assertEquals(2,graph.getEdges().size());
        //2 edges
        assertEquals(1,graph.getEdges().get(x.getVertexId()).size());
        graph.print(new File(System.getProperty("java.io.tmpdir"),"graph.png"));

    }

    @Test
    public void testConstant() {
        TensorGradGraph graph = new TensorGradGraph();
        ArrayFactory arrayFactory = new ArrayFactory(graph);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        Constant<ArrayField> constant  = arrayFieldDifferentialFunctionFactory.zero();
        Constant<ArrayField> one = arrayFieldDifferentialFunctionFactory.zero();
        assertEquals(2,graph.numVertices());
        DifferentialFunction<ArrayField> mul = one.mul(constant);
        assertEquals(2/*why not one?*/,graph.getEdges().size());
        assertEquals(2/*why not one?*/,graph.getVertexInDegree(mul.getVertexId()));

        Variable<ArrayField> variable = arrayFieldDifferentialFunctionFactory.var("x",constant.getValue());
        assertEquals(3,graph.numVertices());
        System.out.println(mul.diff(variable).getFormula(new ArrayList<>()));

    }


    @Test
    public void testGrad() throws Exception {
        TensorGrad tensorGrad = TensorGrad.create();
        TensorGradVariable var = tensorGrad.var("x", Nd4j.valueArrayOf(1,2.0));
        TensorGradVariable xTimesX = var.mul(var);
        TensorGradVariable grad = tensorGrad.grad(xTimesX,var);
        assertEquals("( 2.0 * Math.pow(x,1)", grad.getFormula());
        OpExecOrder opExecOrder = tensorGrad.graph().getOpOrder();
        List<OpState> opStates = opExecOrder.opStates();
        List<Op> ops = tensorGrad.exec();
        assertEquals(opStates.size(),ops.size());
        assertArrayEquals(new int[]{1,1},ops.get(ops.size() - 1).z().shape());
        assertEquals(4.0,ops.get(ops.size() - 1).z().getDouble(0),1e-1);
        System.out.println(ops);
        //tensorGrad.graph().print(new File("/tmp/graph.png"));
    }

    @Test
    public void testSingleTransformOp() throws Exception {
       TensorGradGraph graph = new TensorGradGraph();
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

        Field<ArrayField> h = arrayFactory.abs(x.getValue());

        //x, x as the duplicate input and result are the vertices
        assertEquals(2,graph.numVertices());
        //x * x - edges for only 1 vertex and 1 duplicate
        assertEquals(1,graph.getEdges().size());
        //2 edges
        assertEquals(1,graph.getEdges().get(0).size());
        graph.print(new File(System.getProperty("java.io.tmpdir"),"graph.png"));

    }


    @Test
    public void testAutoDiffSimple() throws Exception {
       TensorGradGraph graph = new TensorGradGraph();
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
        System.out.println(h.getFormula(new ArrayList<>()));
        //x and result are the vertices
        assertEquals(3, graph.numVertices());
        //x * x - edges for only 1 vertex
        assertEquals(2, graph.getEdges().size());
        //2 edges
        assertEquals(1,graph.getEdges().get(0).size());
        System.out.println("Pre graph " + graph);
        //the polynomial doesn't create edges (power,one,..)
        DifferentialFunction<ArrayField> dif = h.diff(x);
        System.out.println("Formula  " + dif.getFormula(new ArrayList<>()));
        assertEquals(3,graph.getEdges().get(0).size());
        //This accumulates the edges from both x * x and 2 * (x,1) ^ 1 (the derivative)
        System.out.println(graph.toString());
        dif.getValue();
        //getValue shouldn't change graph
        assertEquals(3,graph.getEdges().get(0).size());
        dif.getFormula(new ArrayList<>());
        //getFormula shouldn't change graph
        assertEquals(3,graph.getEdges().get(0).size());
        //should have polynomial edges from 2 to 4 and 2 to 5
        assertEquals(1,graph.getEdges().get(2).size());
        graph.print(new File(System.getProperty("java.io.tmpdir"),"graph.png"));
        for(List<Edge<OpState>> edges : graph.getEdges().values()) {
        }
    }


}
