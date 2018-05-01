package org.nd4j.linalg.compression;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.ByteArrayInputStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests for SerDe on compressed arrays
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionSerDeTests extends BaseNd4jTest {
    public CompressionSerDeTests(Nd4jBackend backend) {
        super(backend);
    }


    /*
        This test checks for automatic decompression after deserialization
     */
    @Test
    public void testAutoDecompression1() throws Exception {
        INDArray array = Nd4j.linspace(1, 250, 250);

        INDArray compressed = Nd4j.getCompressor().compress(array, "UINT8");

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(bos, compressed);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());

        INDArray result = Nd4j.read(bis);

        assertEquals(array, result);
    }

    @Test
    public void testManualDecompression1() throws Exception {
        INDArray array = Nd4j.linspace(1, 5, 10);

        INDArray compressed = Nd4j.getCompressor().compress(array, "FLOAT16");

        assertEquals(true, compressed.isCompressed());

        //        assertEquals(true, compressed.data() instanceof CompressedDataBuffer);

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(bos, compressed);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());

        INDArray result = Nd4j.read(bis);

        assertTrue(result.isCompressed());
        INDArray decomp = Nd4j.getCompressor().decompress(result);

        assertArrayEquals(array.data().asFloat(), decomp.data().asFloat(), 0.1f);
    }

    @Test
    public void testAutoDecompression2() throws Exception {
        INDArray array = Nd4j.linspace(1, 10, 11);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(bos, compressed);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());

        System.out.println("Restoring -------------------------");

        INDArray result = Nd4j.read(bis);

        System.out.println("Decomp -------------------------");

        INDArray decomp = Nd4j.getCompressor().decompress(result);

        assertEquals(array, decomp);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
