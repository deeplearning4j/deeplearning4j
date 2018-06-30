package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.IUpdater;

public interface TrainingConfig {

    String getLayerName();

    boolean isPretrain();

    /**
     * Get the L1 coefficient for the given parameter.
     * Different parameters may have different L1 values, even for a single .l1(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName Parameter name
     * @return L1 value for that parameter
     */
    double getL1ByParam(String paramName);

    /**
     * Get the L2 coefficient for the given parameter.
     * Different parameters may have different L2 values, even for a single .l2(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName Parameter name
     * @return L2 value for that parameter
     */
    double getL2ByParam(String paramName);

    /**
     * Is the specified parameter a layerwise pretraining only parameter?<br>
     * For example, visible bias params in an autoencoder (or, decoder params in a variational autoencoder) aren't
     * used during supervised backprop.<br>
     * Layers (like DenseLayer, etc) with no pretrainable parameters will return false for all (valid) inputs.
     *
     * @param paramName Parameter name/key
     * @return True if the parameter is for layerwise pretraining only, false otherwise
     */
    boolean isPretrainParam(String paramName);

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName Parameter name
     * @return IUpdater for the parameter
     */
    IUpdater getUpdaterByParam(String paramName);

    GradientNormalization getGradientNormalization();

    double getGradientNormalizationThreshold();

    void setPretrain(boolean pretrain);

}
