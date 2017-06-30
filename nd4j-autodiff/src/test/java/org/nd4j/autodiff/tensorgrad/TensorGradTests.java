package org.nd4j.autodiff.tensorgrad;

import org.junit.Test;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 4/11/17.
 */
public class TensorGradTests {
    @Test
    public void testSigmoid() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable sigmoid = tensorGrad.sigmoid(x);
        assertEquals("sigmoid(x)",sigmoid.getVarName());
        assertEquals(2,tensorGrad.graph().numVertices());
        assertEquals(1,tensorGrad.graph().getEdges().size());
        assertArrayEquals(arr.shape(), sigmoid.getShape());
        assertEquals(1,tensorGrad.graph().getVertexInDegree(sigmoid.getDifferentialFunction().getVertexId()));
        int[] sorted = new int[] { x.getArrayField().getVertexId(), sigmoid.getDifferentialFunction().getVertexId() };
        assertArrayEquals(sorted, tensorGrad.graph().topologicalSort());
        assertEquals(1,tensorGrad.graph().getOpOrder().getActions().size());
        OpState opState = tensorGrad.graph().getOpOrder().getActions().get(0).getOpState();
        assertEquals("sigmoid",opState.getOpName());
        tensorGrad.allocate();
        Op op = tensorGrad.createOp(OpState.OpType.TRANSFORM,tensorGrad.graph().getOpOrder().getActions().get(0));
        assertTrue(op instanceof Sigmoid);
        Nd4j.getExecutioner().exec(op);
        assertEquals(Transforms.sigmoid(Nd4j.linspace(1,4,4)),op.z());
    }

    @Test
    public void testSum() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        TensorGradVariable x = tensorGrad.var("x", arr);
        TensorGradVariable result = tensorGrad.sum(x, 1);
        assertEquals("sum(x)", result.getVarName());
        assertEquals(2, tensorGrad.graph().numVertices());
        assertEquals(1, tensorGrad.graph().getEdges().size());
        assertArrayEquals(arr.shape(),result.getShape());
        assertArrayEquals(new int[] { 1, 2 }, tensorGrad.graph().topologicalSort());
    }

    @Test
    public void testReshape() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable result = tensorGrad.reshape(x, 2, 2);
        assertEquals("reshape(x)",result.getVarName());
        assertEquals(2,tensorGrad.graph().numVertices());
        assertEquals(1,tensorGrad.graph().getEdges().size());
        assertArrayEquals(new int[]{2,2},result.getShape());

    }

    @Test
    public void testTranspose() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4));
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable result = tensorGrad.transpose(x);
        assertEquals("transpose(x)",result.getVarName());
        assertEquals(2,tensorGrad.graph().numVertices());
        assertEquals(1,tensorGrad.graph().getEdges().size());
        assertArrayEquals(new int[]{4,1},result.getShape());

    }

    @Test
    public void testDistance() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.cosineSimilarity(x,y,1);
        TensorGradVariable addResult = result.add(result);

        assertEquals("cosineSimilarity(x,y)",result.getVarName());
        assertEquals(3,tensorGrad.graph().numVertices());
        assertEquals(2,tensorGrad.graph().getEdges().get(0).size());
        assertArrayEquals(new int[]{1,2},result.getShape());
    }

    @Test
    public void testTensorGradMmul() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.mmul(0,x,y);
        TensorGradVariable otherResult = result.add(result);
        assertEquals("mmul(x,y)",result.getVarName());
        assertEquals(5,tensorGrad.graph().numVertices()); // XXX: Why 5 instead of 3?
        assertEquals(3,tensorGrad.graph().getEdges().size()); // XXX: Why 3 instead of 2?
        assertArrayEquals(new int[]{2,2},result.getShape());
    }


    @Test
    public void testGetInputs() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.mmul(0,x,y);
        TensorGradVariable otherResult = result.add(result);
        assertEquals(2,tensorGrad.graph().getInputs().size());
    }

    @Test
    public void testGetOutputs() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.mmul(0,x,y);
        TensorGradVariable otherResult = result.add(result);
        assertEquals(2,tensorGrad.graph().getOutputs().size());
    }

    @Test
    public void testEval() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable sigmoid = tensorGrad.sigmoid(x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray[] eval = tensorGrad.eval(Collections.singletonMap("x",arr));
        assertEquals(assertion,eval[0]);

    }

    @Test
    public void testEvalAddSelf() {
        /**
         * Note this test fails yet due to needing
         * to validate simple cases like x * x
         * matching number of inputs.
         */
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable sigmoid = x.mul(x);
        INDArray assertion = arr.mul(arr);
        INDArray[] eval = tensorGrad.eval(Collections.singletonMap("x",arr));
        assertEquals(assertion,eval[0]);

    }

    @Test
    public void testEvalAdd() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Nd4j.linspace(1,4,4);
        INDArray yArr = arr.dup();
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",yArr);

        TensorGradVariable sigmoid = x.mul(y);
        INDArray assertion = arr.mul(arr);
        Map<String,INDArray> vars = new HashMap<>();
        vars.put("x",arr);
        vars.put("y",yArr);
        INDArray[] eval = tensorGrad.eval(vars);
        assertEquals(assertion,eval[0]);

    }




    @Test
    public void testTensorGradTensorMmul() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,8,8)).reshape(2,2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.tensorMmul(x,y,new int[][]{{0},{1}},0);
        assertEquals("tensorMmul(x,y)",result.getVarName());
        assertEquals(3,tensorGrad.graph().numVertices());
        assertEquals(2,tensorGrad.graph().getEdges().size());
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new int[]{2,2,2},new int[]{2,2,2},new int[][]{{0},{1}}),result.getShape());
        assertEquals(32,tensorGrad.numElements());
    }

    @Test
    public void testDup() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,8,8)).reshape(2,2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGrad tg2 = tensorGrad.dup();
        assertEquals(tensorGrad, tg2);
    }

    @Test
    public void testOpExecutionWithAutoDiff() {
        TensorGrad tensorGrad = TensorGrad.create();

        INDArray arr = Nd4j.linspace(1,4,4);

        TensorGradVariable x = tensorGrad.var("x", arr);
        TensorGradVariable sigmoid = tensorGrad.sigmoid(x);
        TensorGradVariable grad = tensorGrad.grad(sigmoid, x);

        List<OpExecAction> actions = tensorGrad.graph().getOpOrder().getActions();

        OpState opState = actions.get(0).getOpState();
        assertEquals("sigmoid", opState.getOpName());

        OpState opState2 = actions.get(1).getOpState();
        assertEquals("sigmoidderivative", opState2.getOpName());

        tensorGrad.allocate();

        Op op1 = tensorGrad.createOp(actions.get(0).getOpState().getOpType(), actions.get(0));
        assertTrue(op1 instanceof Sigmoid);
        Nd4j.getExecutioner().exec(op1);
        assertEquals(Transforms.sigmoid(arr), op1.z());

        Op op2 = tensorGrad.createOp(actions.get(1).getOpState().getOpType(), actions.get(1));
        assertTrue(op2 instanceof SigmoidDerivative);
        Nd4j.getExecutioner().exec(op2);
    }


    @Test
    public void testLogisticRegression() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12,  0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{1,1,0,0},2);

        INDArray weights = Nd4j.zeros(3).transpose();

        TensorGradVariable x = tensorGrad.var("x",inputs);
        TensorGradVariable y = tensorGrad.var("y",labels);
        TensorGradVariable w = tensorGrad.var("w",weights);

        TensorGradVariable learningRate = tensorGrad.scalar("lr",0.01);

        TensorGradVariable preOutput = tensorGrad.mmul(0,x,w);

        TensorGradVariable outputs = tensorGrad.sigmoid(preOutput);
        //    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        TensorGradVariable probs = outputs.mul(y).add(outputs.rsub(tensorGrad.scalar("one",1.0)).mul(y.rsub(tensorGrad.scalar("onetwo",1.0))));
   /*     TensorGradVariable outputGrad = tensorGrad.grad(probs,x);
        for(int i = 0; i < 1; i++) {
            w = w.sub(w.mul(outputGrad).mul(learningRate));
        }
*/
        List<Op> ops = tensorGrad.exec();
        System.out.println(ops);
    }

}
