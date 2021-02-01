package org.nd4j.codegen.api

/**
 *  See org.nd4j.autodiff.los.LossReduce in nd4j for documentation.
 */
enum class LossReduce {
    NONE,
    SUM,
    MEAN_BY_WEIGHT,
    MEAN_BY_NONZERO_WEIGHT_COUNT
}
