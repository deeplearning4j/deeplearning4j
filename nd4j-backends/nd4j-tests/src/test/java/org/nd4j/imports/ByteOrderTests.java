package org.nd4j.imports;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ByteOrderTests  extends BaseNd4jTest {

    public ByteOrderTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testByteArrayOrder1() {
        val ndarray = Nd4j.create(2).assign(1);

        assertEquals(DataBuffer.Type.FLOAT, ndarray.data().dataType());

        val array = ndarray.data().asBytes();

        assertEquals(8, array.length);
    }

    @Test
    public void testByteArrayOrder2() {
        val original = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }


    @Test
    public void testByteArrayOrder3() {
        val original = Nd4j.linspace(1, 25, 25).reshape('f', 5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }

    @Test
    public void testShapeStridesOf1() {
        val buffer = new int[]{2, 5, 5, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5}, shape);
        assertArrayEquals(new int[]{5, 1}, strides);
    }

    @Test
    public void testShapeStridesOf2() {
        val buffer = new int[]{3, 5, 5, 5, 25, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5, 5}, shape);
        assertArrayEquals(new int[]{25, 5, 1}, strides);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
