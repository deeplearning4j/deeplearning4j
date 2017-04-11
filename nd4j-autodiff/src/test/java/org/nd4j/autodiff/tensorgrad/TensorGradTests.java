package org.nd4j.autodiff.tensorgrad;

import org.junit.Test;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 4/11/17.
 */
public class TensorGradTests {
    @Test
    public void testSigmoid() {
        TensorGrad tensorGrad = TensorGrad.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1,4,4));
        TensorGradVariable x = tensorGrad.var("x",arr);
        TensorGradVariable sigmoid = tensorGrad.sigmoid(x);
        assertEquals("sigmoid(x)",sigmoid.getVarName());
        assertEquals(2,tensorGrad.graph().numVertices());
        assertEquals(1,tensorGrad.graph().getEdges().size());
        assertArrayEquals(arr.shape(),sigmoid.getShape());

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

}
