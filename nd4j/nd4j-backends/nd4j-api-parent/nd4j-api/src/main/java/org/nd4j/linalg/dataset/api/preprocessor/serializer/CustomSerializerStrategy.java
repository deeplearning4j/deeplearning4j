package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;

/**
 * Base class for custom normalizer serializers
 */
public abstract class CustomSerializerStrategy<T extends Normalizer> implements NormalizerSerializerStrategy<T> {
    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.CUSTOM;
    }

    /**
     * Get the class of the supported custom serializer
     *
     * @return the class
     */
    public abstract Class<T> getSupportedClass();
}
