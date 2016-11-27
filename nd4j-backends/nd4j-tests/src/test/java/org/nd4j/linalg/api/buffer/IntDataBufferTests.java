package org.nd4j.linalg.api.buffer;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Tests for INT INDArrays and DataBuffers serialization
 *
 * @author raver119@gmail.com
 */
public class IntDataBufferTests extends BaseNd4jTest {

    public IntDataBufferTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBasicSerde1() throws Exception {


        DataBuffer dataBuffer = Nd4j.createBuffer(new int[]{1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, 5});
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        File tempFile = File.createTempFile("test", "test");
        tempFile.deleteOnExit();

        Nd4j.saveBinary(intArray, tempFile);

        InputStream stream = new FileInputStream(tempFile);
        BufferedInputStream bis = new BufferedInputStream(stream);
        DataInputStream dis = new DataInputStream(bis);

        INDArray loaded = Nd4j.read(dis);

        assertEquals(DataBuffer.Type.INT, loaded.data().dataType());
        assertEquals(DataBuffer.Type.INT, loaded.shapeInfoDataBuffer().dataType());

        assertEquals(intArray.data().length(), loaded.data().length());

        assertArrayEquals(intArray.data().asInt(), loaded.data().asInt());
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testOpDiscarded() throws Exception {
        DataBuffer dataBuffer = Nd4j.createBuffer(new int[]{1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, 5});
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        intArray.add(10f);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
