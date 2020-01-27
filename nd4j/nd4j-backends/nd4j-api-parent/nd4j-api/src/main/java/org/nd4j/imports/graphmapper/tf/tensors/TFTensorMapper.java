package org.nd4j.imports.graphmapper.tf.tensors;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;
import java.nio.ByteBuffer;

/**
 * @param <J> Java array type
 * @param <B> Java buffer type
 */
public interface TFTensorMapper<J,B extends Buffer> {

    enum ValueSource {EMPTY, VALUE_COUNT, BINARY};

    DataType dataType();

    long[] shape();

    boolean isEmpty();

    ValueSource valueSource();

    int valueCount();

    J newArray(int length);

    B getBuffer(ByteBuffer bb);

    INDArray toNDArray();

    void getValue(J jArr, int i);

    void getValue(J jArr, B buffer, int i);

    INDArray arrayFor(long[] shape, J jArr);


}
