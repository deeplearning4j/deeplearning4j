package org.nd4j.linalg.compression;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionMagicTests extends BaseNd4jTest {
    public CompressionMagicTests(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() throws Exception {
        Nd4j.preventUnpack = true;
    }

    @Test
    public void testMagicDecompression1() throws Exception {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        assertTrue(compressed.isCompressed());
        compressed.muli(1.0);

        assertEquals(array, compressed);
    }

    @Test
    public void testMagicDecompression2() throws Exception {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "FLOAT16");

        assertTrue(compressed.isCompressed());
        compressed.muli(1.0);

        assertArrayEquals(array.data().asFloat(), compressed.data().asFloat(), 0.1f);
    }

    @Test
    public void testMagicDecompression3() throws Exception {
        INDArray array = Nd4j.linspace(1, 2500, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "INT16");

        Nd4j.getExecutioner().commit();;

        assertTrue(compressed.isCompressed());
        compressed.muli(1.0);

        assertEquals(array, compressed);
    }


    @Test
    public void testMagicDecompression4() throws Exception {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        for (int cnt = 0; cnt < array.length(); cnt++) {
            float a = array.getFloat(cnt);
            float c = compressed.getFloat(cnt);
            assertEquals(a, c, 0.01f);
        }

    }

    @Test
    public void testDupSkipDecompression1() {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup();
        assertTrue(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
    }

    @Test
    public void testDupSkipDecompression2() {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup('c');
        assertTrue(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
    }

    @Test
    public void testDupSkipDecompression3() {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup('f');
        assertFalse(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        //        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
        assertEquals('f', newArray.ordering());
        assertEquals('c', compressed.ordering());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
