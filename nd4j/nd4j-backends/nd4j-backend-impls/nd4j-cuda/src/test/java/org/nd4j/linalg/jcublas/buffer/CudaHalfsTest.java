package org.nd4j.linalg.jcublas.buffer;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaHalfsTest {

    @Before
    public void setUp() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
    }

    @Test
    public void testSerialization1() throws Exception {

        INDArray array = Nd4j.linspace(1, 5, 10);

        File tempFile = File.createTempFile("alpha", "11");
        tempFile.deleteOnExit();

        // now we serialize halfs, and we expect it to become floats on other side
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(tempFile.getAbsolutePath())))){
            Nd4j.write(array, dos);
        }

        // loading data back from file
        DataInputStream dis = new DataInputStream(new FileInputStream(tempFile.getAbsoluteFile()));

        INDArray restored = Nd4j.read(dis);

        assertEquals(array, restored);
    }


    @Test
    public void testSerialization2() throws Exception {

        INDArray array = Nd4j.linspace(1, 5, 10);

        File tempFile = File.createTempFile("alpha", "11");
        tempFile.deleteOnExit();

        // now we serialize halfs, and we expect it to become floats on other side
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(tempFile.getAbsolutePath())))){
            Nd4j.write(array, dos);
        }

        // loading data back from file
        DataInputStream dis = new DataInputStream(new FileInputStream(tempFile.getAbsoluteFile()));

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

        INDArray exp = Nd4j.linspace(1, 5, 10);

        INDArray restored = Nd4j.read(dis);

        assertArrayEquals(exp.data().asFloat(), restored.data().asFloat(), 0.1f);
        assertEquals(DataBuffer.Type.FLOAT, exp.data().dataType());
    }

    @Test
    public void test2D1() throws Exception {
        float[][] array = new float[5][5];

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                array[i][j] = i;
            }
        }

        // crate INDArray
        INDArray data = Nd4j.create(array);

        assertEquals(25, data.length());

        for (int i = 0; i < array.length; i++) {
            INDArray row = data.getRow(i);
            for (int x = 0; x < row.length(); x++) {
                assertEquals((float) i, row.getFloat(x), 0.1f);
            }
        }

        System.out.println(data);
    }

}
