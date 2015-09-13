package org.nd4j.linalg.api.indexing.shape;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
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
        Pair<INDArray,INDArray> rawIter = Shape.prepareTwoRawArrayIter(b,a);
        for(int i = 0; i < a.length(); i++) {
            rawIter.getSecond().data().put(rawIter.getSecond().offset() + i * rawIter.getSecond().stride(-1),rawIter.getFirst().data().getDouble(rawIter.getFirst().offset()  + i * rawIter.getFirst().stride(-1)));
        }

        System.out.println(b);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
