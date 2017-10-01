package org.nd4j.compression.impl;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author raver119@gmail.com
 */
public class Gzip extends AbstractCompressor {
    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "GZIP";
    }

    /**
     * This method returns compression opType provided by specific NDArrayCompressor implementation
     *
     * @return
     */
    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSLESS;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {
        try {

            CompressedDataBuffer compressed = (CompressedDataBuffer) buffer;
            CompressionDescriptor descriptor = compressed.getCompressionDescriptor();

            BytePointer pointer = (BytePointer) compressed.addressPointer();
            ByteArrayInputStream bis = new ByteArrayInputStream(pointer.getStringBytes());
            GZIPInputStream gzip = new GZIPInputStream(bis);
            DataInputStream dis = new DataInputStream(gzip);

            DataBuffer bufferRestored = Nd4j.createBuffer(descriptor.getNumberOfElements());
            bufferRestored.read(dis);

            return bufferRestored;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        try {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            GZIPOutputStream gzip = new GZIPOutputStream(stream);
            DataOutputStream dos = new DataOutputStream(gzip);

            buffer.write(dos);
            dos.flush();
            dos.close();

            byte[] bytes = stream.toByteArray();
            //            logger.info("Bytes: {}", Arrays.toString(bytes));
            BytePointer pointer = new BytePointer(bytes);
            CompressionDescriptor descriptor = new CompressionDescriptor(buffer, this);
            descriptor.setCompressedLength(bytes.length);

            CompressedDataBuffer result = new CompressedDataBuffer(pointer, descriptor);

            return result;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataBuffer.TypeEx srcType, Pointer srcPointer, int length,
                    int elementSize) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
