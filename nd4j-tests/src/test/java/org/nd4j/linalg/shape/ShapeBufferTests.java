package org.nd4j.linalg.shape;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.IntBuffer;

import static org.junit.Assert.*;

/**
 * Created by agibsoncccc on 1/30/16.
 */
public class ShapeBufferTests extends BaseNd4jTest {
    public ShapeBufferTests() {
        super();
    }

    public ShapeBufferTests(String name) {
        super(name);
    }

    public ShapeBufferTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public ShapeBufferTests(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testRank() {
        int[] shape = {2,4};
        int[] stride = {1,2};
        IntBuffer buff = Shape.createShapeInformation(shape, stride, 0, 1, 'c');
        int rank = 2;
        assertEquals(rank,Shape.rank(buff));

    }

    @Test
    public void testShape() {
        int[] shape = {2,4};
        int[] stride = {1,2};
        IntBuffer buff = Shape.createShapeInformation(shape, stride, 0, 1, 'c');
        IntBuffer shapeView = Shape.shapeOf(buff);
        assertTrue(Shape.contentEquals(shape,shapeView));
        IntBuffer strideView = Shape.stride(buff);
        assertTrue(Shape.contentEquals(stride,strideView));
        assertEquals('c',Shape.order(buff));
        assertEquals(1,Shape.elementWiseStride(buff));
        

    }


}
