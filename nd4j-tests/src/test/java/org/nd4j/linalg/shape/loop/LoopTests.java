package org.nd4j.linalg.shape.loop;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.api.shape.loop.one.RawArrayIterationInformation1;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class LoopTests extends BaseNd4jTest {
    public LoopTests() {
    }

    public LoopTests(Nd4jBackend backend) {
        super(backend);
    }

    public LoopTests(String name) {
        super(name);
    }

    public LoopTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Test
    public void testLoop1d() {
        RawArrayIterationInformation1 iter = Shape.prepareRawArrayIter(Nd4j.linspace(1,4,4).reshape(2,2));
        System.out.println(iter);
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
