package org.nd4j.linalg.compression;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * @author raver119@gmail.com
 */
public class CompressedDataBuffer extends BaseDataBuffer {
    @Getter @Setter protected CompressionDescriptor compressionDescriptor;
    @Getter @Setter protected Pointer pointer;
    private static Logger logger = LoggerFactory.getLogger(CompressedDataBuffer.class);

    public CompressedDataBuffer(Pointer pointer, @NonNull CompressionDescriptor descriptor) {
        this.compressionDescriptor = descriptor;
        this.pointer = pointer;

        initTypeAndSize();
    }

    /**
     * Initialize the type of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = -1;
        type = Type.COMPRESSED;
    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        logger.info("Writing out CompressedDataBuffer");
    }

    @Override
    public void read(DataInputStream s) {
        logger.info("Reading CompressedDataBuffer from DataInputStream");
    }

    @Override
    public void read(InputStream is) {
        logger.info("Reading CompressedDataBuffer from InputStream");
    }

    private void readObject(ObjectInputStream s) {
        logger.info("Reading CompressedDataBuffer readObject");
        doReadObject(s);
    }

    protected void doReadObject(ObjectInputStream s) {
        logger.info("Reading CompressedDataBuffer from doReadObject");
        try {
            s.defaultReadObject();
            read(s);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    /**
     * Create with length
     *
     * @param length a databuffer of the same type as
     *               this with the given length
     * @return a data buffer with the same length and datatype as this one
     */
    @Override
    protected DataBuffer create(long length) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(double[] data) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(float[] data) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(int[] data) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    @Override
    public IComplexFloat getComplexFloat(long i) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    @Override
    public IComplexDouble getComplexDouble(long i) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }
}
