package org.nd4j.linalg.dataset.api.preprocessor.serializer;

/**
 * Enum representing the opType of a normalizer for serialization purposes
 */
public enum NormalizerType {
    STANDARDIZE, MIN_MAX, IMAGE_MIN_MAX, IMAGE_VGG16, MULTI_STANDARDIZE, MULTI_MIN_MAX, MULTI_HYBRID, CUSTOM,
}
