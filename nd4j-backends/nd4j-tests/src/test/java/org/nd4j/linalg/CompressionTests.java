package org.nd4j.linalg;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.compression.DataCompressor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionTests extends BaseNd4jTest  {

    public CompressionTests(Nd4jBackend backend) {
            super(backend);
    }


    @Test
    public void testFP16Compression1() {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

        DataCompressor.getInstance().setDefaultCompression("FP16");

        DataCompressor.getInstance().printAvailableCompressors();

        INDArray compr = DataCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = DataCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(5.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testFP16Compression2() {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});
        DataBuffer exp = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});

        DataCompressor.getInstance().setDefaultCompression("FP16");

        DataBuffer compr = DataCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = DataCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(5.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testUint8Compression1() {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});
        DataBuffer exp = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});

        DataCompressor.getInstance().setDefaultCompression("UINT8");

        DataBuffer compr = DataCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = DataCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(5.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testUint8Compression2() {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 1005f});
        DataBuffer exp = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 1005f});

        DataCompressor.getInstance().setDefaultCompression("UINT8");

        DataBuffer compr = DataCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = DataCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(255.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testGzipCompression1() {
        INDArray array = Nd4j.linspace(1, 10000, 20000);
        INDArray exp = array.dup();

        DataCompressor.getInstance().setDefaultCompression("GZIP");

        INDArray compr = DataCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = DataCompressor.getInstance().decompress(compr);

        assertEquals(exp, array);
        assertEquals(exp, decomp);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
