package org.nd4j.linalg.compression;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.compression.BasicNDArrayCompressor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionTests extends BaseNd4jTest {

    public CompressionTests(Nd4jBackend backend) {
            super(backend);
    }


    @Test
    public void testCompressionDescriptorSerde() {
        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(4);
        descriptor.setOriginalElementSize(4);
        descriptor.setNumberOfElements(4);
        descriptor.setCompressionAlgorithm("GZIP");
        descriptor.setOriginalLength(4);
        descriptor.setCompressionType(CompressionType.LOSSY);
        ByteBuffer toByteBuffer = descriptor.toByteBuffer();
        CompressionDescriptor fromByteBuffer = CompressionDescriptor.fromByteBuffer(toByteBuffer);
        assertEquals(descriptor,fromByteBuffer);
    }

    @Test
    public void testGzipInPlaceCompression() {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        Nd4j.getCompressor().setDefaultCompression("GZIP");
        Nd4j.getCompressor().compressi(array);
        assertTrue(array.isCompressed());
        Nd4j.getCompressor().decompressi(array);
        assertFalse(array.isCompressed());
     }

    @Test
    public void testFP16Compression1() {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("INT8");

        BasicNDArrayCompressor.getInstance().printAvailableCompressors();

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

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

        BasicNDArrayCompressor.getInstance().setDefaultCompression("FLOAT16");

        DataBuffer compr = BasicNDArrayCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(5.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testFP16Compression3() {
        INDArray buffer = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        INDArray exp = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("FLOAT16");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(buffer);

        assertEquals(false, buffer.isCompressed() );
        assertEquals(true, compr.isCompressed() );
        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

//        assertNotEquals(exp, compr);

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(false, decomp.isCompressed() );
        assertEquals(DataBuffer.Type.FLOAT, decomp.data().dataType());

        assertEquals(exp, decomp);
    }

    @Test
    public void testUint8Compression1() {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});
        DataBuffer exp = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("UINT8");

        DataBuffer compr = BasicNDArrayCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

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

        BasicNDArrayCompressor.getInstance().setDefaultCompression("UINT8");

        DataBuffer compr = BasicNDArrayCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(255.0f, decomp.getFloat(4), 0.01f);
    }

    @Test
    public void testInt8Compression1() {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 2f, 3f, 4f, 1005f, -3.7f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("INT8");

        DataBuffer compr = BasicNDArrayCompressor.getInstance().compress(buffer);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.dataType());

        DataBuffer decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(127.0f, decomp.getFloat(4), 0.01f);
        assertEquals(-3.0f, decomp.getFloat(5), 0.01f);
    }


    @Test
    public void testGzipCompression1() {
        INDArray array = Nd4j.linspace(1, 10000, 20000);
        INDArray exp = array.dup();

        BasicNDArrayCompressor.getInstance().setDefaultCompression("GZIP");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(exp, array);
        assertEquals(exp, decomp);
    }

    @Test
    public void testNoOpCompression1() {
        INDArray array = Nd4j.linspace(1, 10000, 20000);
        INDArray exp = array.dup();

        BasicNDArrayCompressor.getInstance().setDefaultCompression("NOOP");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(exp, array);
        assertEquals(exp, decomp);
    }


    @Test
    public void testFP8Compression1() {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("FLOAT8");

        BasicNDArrayCompressor.getInstance().printAvailableCompressors();

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataBuffer.Type.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(1.0f, decomp.getFloat(0), 0.01f);
        assertEquals(2.0f, decomp.getFloat(1), 0.01f);
        assertEquals(3.0f, decomp.getFloat(2), 0.01f);
        assertEquals(4.0f, decomp.getFloat(3), 0.01f);
        assertEquals(5.0f, decomp.getFloat(4), 0.01f);
    }


    @Test
    public void testJVMCompression1() throws Exception {
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("FLOAT16");

        INDArray compressed = BasicNDArrayCompressor.getInstance().compress(new float[]{1f, 2f, 3f, 4f, 5f});
        assertNotEquals(null, compressed.data());
        assertNotEquals(null, compressed.shapeInfoDataBuffer());
        assertTrue(compressed.isCompressed());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compressed);

        assertEquals(exp, decomp);
    }

    @Test
    public void testJVMCompression2() throws Exception {
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("INT8");

        INDArray compressed = BasicNDArrayCompressor.getInstance().compress(new float[]{1f, 2f, 3f, 4f, 5f});
        assertNotEquals(null, compressed.data());
        assertNotEquals(null, compressed.shapeInfoDataBuffer());
        assertTrue(compressed.isCompressed());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compressed);

        assertEquals(exp, decomp);
    }

    @Test
    public void testJVMCompression3() throws Exception {
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        BasicNDArrayCompressor.getInstance().setDefaultCompression("NOOP");

        INDArray compressed = BasicNDArrayCompressor.getInstance().compress(new float[]{1f, 2f, 3f, 4f, 5f});
        assertNotEquals(null, compressed.data());
        assertNotEquals(null, compressed.shapeInfoDataBuffer());
        assertTrue(compressed.isCompressed());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compressed);

        assertEquals(exp, decomp);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
