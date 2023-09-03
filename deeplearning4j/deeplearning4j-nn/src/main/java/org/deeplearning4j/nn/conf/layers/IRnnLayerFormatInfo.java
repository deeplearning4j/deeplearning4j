package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.RNNFormat;

/**
 * Interface to implement by the layers that provide RNN layer format information.
 */
public interface IRnnLayerFormatInfo {

    RNNFormat getRnnDataFormat();
}
