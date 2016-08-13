package org.nd4j.linalg.compression;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionMagicTests extends BaseNd4jTest {
    public CompressionMagicTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testMagicDecompression1() throws Exception {
        INDArray array = Nd4j.linspace(1, 100, 2500);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        compressed.muli(1.0);

        assertEquals(array, compressed);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
