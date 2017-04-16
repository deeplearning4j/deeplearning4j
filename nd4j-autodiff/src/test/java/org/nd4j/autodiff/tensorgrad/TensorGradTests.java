package org.nd4j.autodiff.tensorgrad;

import org.junit.Test;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

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
        assertArrayEquals(arr.shape(),sigmoid.getShape());
        assertEquals(1,tensorGrad.graph().getVertexInDegree(1));
        assertArrayEquals(new int[]{0,1},tensorGrad.graph().topologicalSort());
        assertEquals(1,tensorGrad.graph().getOpOrder().size());
        OpState opState = tensorGrad.graph().getOpOrder().get(0).getOpState();
        assertEquals("sigmoid",opState.getOpName());
        tensorGrad.allocate();
        Op op = tensorGrad.createOp(OpState.OpType.TRANSFORM,tensorGrad.graph().getOpOrder().get(0));
        assertTrue(op instanceof Sigmoid);
        Nd4j.getExecutioner().exec(op);
        assertEquals(Transforms.sigmoid(Nd4j.linspace(1,4,4)),op.z());

    }

    @Test
    public void testSum() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4));
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable result = tensorGrad.sum(x,1);
        assertEquals("sum(x)",result.getVarName());
        assertEquals(2,tensorGrad.graph().numVertices());
        assertEquals(1,tensorGrad.graph().getEdges().size());
        assertArrayEquals(arr.shape(),result.getShape());
        assertArrayEquals(new int[]{0,1},tensorGrad.graph().topologicalSort());


    }



    @Test
    public void testReshape() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable result = tensorGrad.reshape(x);
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
        assertEquals("cosineSimilarity(x,y)",result.getVarName());
        assertEquals(3,tensorGrad.graph().numVertices());
        assertEquals(2,tensorGrad.graph().getEdges().size());
        assertArrayEquals(new int[]{1,2},result.getShape());
    }

    @Test
    public void testTensorGradMmul() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2);
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable y = tensorGrad.var("y",arr);
        TensorGradVariable result = tensorGrad.mmul(0,x,y);
        assertEquals("mmul(x,y)",result.getVarName());
        assertEquals(3,tensorGrad.graph().numVertices());
        assertEquals(2,tensorGrad.graph().getEdges().size());
        assertArrayEquals(new int[]{2,2},result.getShape());
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

}
