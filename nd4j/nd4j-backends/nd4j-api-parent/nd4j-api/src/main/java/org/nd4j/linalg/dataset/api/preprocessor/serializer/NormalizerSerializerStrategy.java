package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Strategy for serializing and unserializing a specific opType of normalizer
 *
 * @param <T> the opType of normalizer this strategy supports
 * @author Ede Meijer
 */
public interface NormalizerSerializerStrategy<T extends Normalizer> {
    /**
     * Serialize a normalizer to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    void write(T normalizer, OutputStream stream) throws IOException;

    /**
     * Restore a normalizer that was previously serialized by this strategy
     *
     * @param stream the stream to read serialized data from
     * @return the restored normalizer
     * @throws IOException
     */
    T restore(InputStream stream) throws IOException;

    /**
     * Get the enum opType of the supported normalizer
     *
     * @see Normalizer#getType()
     *
     * @return the enum opType
     */
    NormalizerType getSupportedType();
}
