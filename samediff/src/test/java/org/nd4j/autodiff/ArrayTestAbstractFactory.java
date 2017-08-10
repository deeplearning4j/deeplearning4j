package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.functions.Constant;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.Variable;

import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecOrder;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
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
        return new ArrayFactory(SameDiff.create());
    }


    @Test
    public void testAutoDiff() {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);
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

        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x", new ArrayField(xVertex, sameDiff));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(arrayVertex, sameDiff));
        DifferentialFunction<ArrayField> h = x.mul(x).mul(arrayFieldDifferentialFunctionFactory.cos(x.mul(y)).add(y));

        DifferentialFunction<ArrayField> diff = h.diff(x);
        ArrayField value = diff.getValue(true);
        SameDiff ops = value.getOps();
        String s = ops.toString();

        assertNotNull(s);
    }

    @Test
    public void testVariables() {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);
        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, sameDiff));
        DifferentialFunction<ArrayField> h = x.mul(x);
        System.out.println(h.diff(x).getValue(false).getClass());

    }


    @Test
    public void testPairWiseOp() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);

        NDArrayInformation xInfo = NDArrayInformation.builder().shape(new int[]{1,1}).id("x").build();
        NDArrayInformation yInfo = NDArrayInformation.builder().shape(new int[]{1, 1}).id("y").build();
        NDArrayVertex xVertex = new NDArrayVertex(sameDiff.getGraph().nextVertexId(), xInfo);
        NDArrayVertex yVertex = new NDArrayVertex(sameDiff.getGraph().nextVertexId(), yInfo);

        //2 * x
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, sameDiff));
        Variable<ArrayField> y = arrayFieldDifferentialFunctionFactory.var("y", new ArrayField(yVertex, sameDiff));
        DifferentialFunction<ArrayField> h = x.mul(y);
        System.out.println(h.getFormula(new ArrayList<>()));
        //x, x as the duplicate input and result are the vertices
        assertEquals(3,sameDiff.getGraph().numVertices());
        //x * x - edges for only 1 vertex and 1 duplicate
        assertEquals(2,sameDiff.getGraph().getEdges().size());
        //2 edges
        assertEquals(1,sameDiff.getGraph().getEdges().get(x.getVertexId()).size());
        sameDiff.getGraph().print(new File(System.getProperty("java.io.tmpdir"),"sameDiff.png"));

    }

    @Test
    public void testConstant() {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);
        Constant<ArrayField> constant  = arrayFieldDifferentialFunctionFactory.zero(new int[]{1,2});
        Constant<ArrayField> one = arrayFieldDifferentialFunctionFactory.zero(new int[]{1,2});
        assertEquals(2,sameDiff.getGraph().numVertices());
        DifferentialFunction<ArrayField> mul = one.mul(constant);
        assertEquals(2/*why not one?*/,sameDiff.getGraph().getEdges().size());
        assertEquals(2/*why not one?*/,sameDiff.getGraph().getVertexInDegree(mul.getVertexId()));

        Variable<ArrayField> variable = arrayFieldDifferentialFunctionFactory.var("x",constant.getValue(true));
        assertEquals(3,sameDiff.getGraph().numVertices());
        System.out.println(mul.diff(variable).getFormula(new ArrayList<>()));

    }


    @Test
    public void testGrad() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        SDVariable var = sameDiff.var("x", Nd4j.valueArrayOf(1,2.0));
        SDVariable xTimesX = var.mul(var);
        SDVariable grad = sameDiff.grad(xTimesX,var);
        assertEquals("( 2.0 * Math.pow(x,1)", grad.getFormula());
        OpExecOrder opExecOrder = sameDiff.getGraph().getOpOrder();
        List<OpState> opStates = opExecOrder.opStates();
        List<Op> ops = sameDiff.exec();
        assertEquals(opStates.size(),ops.size());
        assertArrayEquals(new int[]{1,1},ops.get(ops.size() - 1).z().shape());
        assertEquals(4.0,ops.get(ops.size() - 1).z().getDouble(0),1e-1);
        System.out.println(ops);
        //tensorGrad.sameDiff().print(new File("/tmp/sameDiff.png"));
    }

    @Test
    public void testSingleTransformOp() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);


        //2 * x
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, sameDiff));

        Field<ArrayField> h = arrayFactory.abs(x.getValue(true));

        //x, x as the duplicate input and result are the vertices
        assertEquals(2,sameDiff.getGraph().numVertices());
        //x * x - edges for only 1 vertex and 1 duplicate
        assertEquals(1,sameDiff.getGraph().getEdges().size());
        //2 edges
        assertEquals(1,sameDiff.getGraph().getEdges().get(0).size());
        sameDiff.getGraph().print(new File(System.getProperty("java.io.tmpdir"),"sameDiff.png"));

    }


    @Test
    public void testAutoDiffSimple() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        ArrayFactory arrayFactory = new ArrayFactory(sameDiff);

        DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(sameDiff);
        NDArrayInformation xInfo = NDArrayInformation.
                builder().
                shape(new int[]{1,1}).
                id("x").
                build();
        NDArrayVertex xVertex = new NDArrayVertex(0,xInfo);

        //2 * x
        Variable<ArrayField> x = arrayFieldDifferentialFunctionFactory.var("x",new ArrayField(xVertex, sameDiff));
        DifferentialFunction<ArrayField> h = x.mul(x);
        System.out.println(h.getFormula(new ArrayList<>()));
        //x and result are the vertices
        assertEquals(3, sameDiff.getGraph().numVertices());
        //x * x - edges for only 1 vertex
        assertEquals(2, sameDiff.getGraph().getEdges().size());
        //2 edges
        assertEquals(1,sameDiff.getGraph().getEdges().get(0).size());
        System.out.println("Pre sameDiff " + sameDiff);
        //the polynomial doesn't create edges (power,one,..)
        DifferentialFunction<ArrayField> dif = h.diff(x);
        System.out.println("Formula  " + dif.getFormula(new ArrayList<>()));
        assertEquals(3,sameDiff.getGraph().getEdges().get(0).size());
        //This accumulates the edges from both x * x and 2 * (x,1) ^ 1 (the derivative)
        System.out.println(sameDiff.toString());
        dif.getValue(true);
        //getValue shouldn't change sameDiff
        assertEquals(3,sameDiff.getGraph().getEdges().get(0).size());
        dif.getFormula(new ArrayList<>());
        //getFormula shouldn't change sameDiff
        assertEquals(3,sameDiff.getGraph().getEdges().get(0).size());
        //should have polynomial edges from 2 to 4 and 2 to 5
        assertEquals(1,sameDiff.getGraph().getEdges().get(2).size());
        sameDiff.getGraph().print(new File(System.getProperty("java.io.tmpdir"),"sameDiff.png"));
       
    }


}
