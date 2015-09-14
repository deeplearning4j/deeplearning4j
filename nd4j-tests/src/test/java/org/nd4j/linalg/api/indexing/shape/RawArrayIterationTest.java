package org.nd4j.linalg.api.indexing.shape;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.loop.two.CopyLoopFunction;
import org.nd4j.linalg.api.shape.loop.two.RawArrayIterationInformation2;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * Created by agibsonccc on 9/13/15.
 */
public class RawArrayIterationTest extends BaseNd4jTest {

    public RawArrayIterationTest() {
    }

    public RawArrayIterationTest(String name) {
        super(name);
    }

    public RawArrayIterationTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public RawArrayIterationTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRawIteration() {
        INDArray a = Nd4j.linspace(1,6,6).reshape(2,3);
        INDArray b = Nd4j.linspace(1,6,6).reshape(2,3).add(1);
        RawArrayIterationInformation2 rawIter = Shape.prepareTwoRawArrayIter(b,a);
        int[] offsets = new int[2];
        int[] coords = new int[2];
        Shape.raw2dLoop(0,2,coords,rawIter.getShape(),rawIter.getAOffset(),rawIter.getAStrides(),rawIter.getBOffset(),rawIter.getBStrides(),rawIter,new CopyLoopFunction());



        assertEquals(a,b);

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
