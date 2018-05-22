package org.nd4j.serde.binary;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.UUID;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 9/23/16.
 */
public class BinarySerdeTest {

    @Test
    public void testToAndFrom() {
        INDArray arr = Nd4j.scalar(1.0);
        ByteBuffer buffer = BinarySerde.toByteBuffer(arr);
        INDArray back = BinarySerde.toArray(buffer);
        assertEquals(arr, back);
    }

    @Test
    public void testToAndFromCompressed() {
        INDArray arr = Nd4j.scalar(1.0);
        INDArray compress = Nd4j.getCompressor().compress(arr, "GZIP");
        assertTrue(compress.isCompressed());
        ByteBuffer buffer = BinarySerde.toByteBuffer(compress);
        INDArray back = BinarySerde.toArray(buffer);
        INDArray decompressed = Nd4j.getCompressor().decompress(compress);
        assertEquals(arr, decompressed);
        assertEquals(arr, back);
    }


    @Test
    public void testToAndFromCompressedLarge() {
        INDArray arr = Nd4j.zeros((int) 1e7);
        INDArray compress = Nd4j.getCompressor().compress(arr, "GZIP");
        assertTrue(compress.isCompressed());
        ByteBuffer buffer = BinarySerde.toByteBuffer(compress);
        INDArray back = BinarySerde.toArray(buffer);
        INDArray decompressed = Nd4j.getCompressor().decompress(compress);
        assertEquals(arr, decompressed);
        assertEquals(arr, back);
    }


    @Test
    public void testReadWriteFile() throws Exception {
        File tmpFile = new File(System.getProperty("java.io.tmpdir"),
                        "ndarraytmp-" + UUID.randomUUID().toString() + " .bin");
        tmpFile.deleteOnExit();
        INDArray rand = Nd4j.randn(5, 5);
        BinarySerde.writeArrayToDisk(rand, tmpFile);
        INDArray fromDisk = BinarySerde.readFromDisk(tmpFile);
        assertEquals(rand, fromDisk);
    }

    @Test
    public void testReadShapeFile() throws Exception {
        File tmpFile = new File(System.getProperty("java.io.tmpdir"),
                        "ndarraytmp-" + UUID.randomUUID().toString() + " .bin");
        tmpFile.deleteOnExit();
        INDArray rand = Nd4j.randn(5, 5);
        BinarySerde.writeArrayToDisk(rand, tmpFile);
        DataBuffer buffer = BinarySerde.readShapeFromDisk(tmpFile);

        assertArrayEquals(rand.shapeInfoDataBuffer().asInt(), buffer.asInt());
    }

    @Test
    public void timeOldVsNew() throws Exception {
        int numTrials = 1000;
        long oldTotal = 0;
        long newTotal = 0;
        INDArray arr = Nd4j.create(100000);
        Nd4j.getCompressor().compressi(arr, "GZIP");
        for (int i = 0; i < numTrials; i++) {
            StopWatch oldStopWatch = new StopWatch();
            // FIXME: int cast
            BufferedOutputStream bos = new BufferedOutputStream(new ByteArrayOutputStream((int) arr.length()));
            DataOutputStream dos = new DataOutputStream(bos);
            oldStopWatch.start();
            Nd4j.write(arr, dos);
            oldStopWatch.stop();
            // System.out.println("Old " + oldStopWatch.getNanoTime());
            oldTotal += oldStopWatch.getNanoTime();
            StopWatch newStopWatch = new StopWatch();
            newStopWatch.start();
            BinarySerde.toByteBuffer(arr);
            newStopWatch.stop();
            //  System.out.println("New " + newStopWatch.getNanoTime());
            newTotal += newStopWatch.getNanoTime();

        }

        oldTotal /= numTrials;
        newTotal /= numTrials;
        System.out.println("Old avg " + oldTotal + " New avg " + newTotal);

    }

}
