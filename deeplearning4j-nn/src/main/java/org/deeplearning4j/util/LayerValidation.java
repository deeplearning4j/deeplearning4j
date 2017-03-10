package org.deeplearning4j.util;

import org.deeplearning4j.exception.DL4JInvalidConfigException;

/**
 * Created by Alex on 12/11/2016.
 */
public class LayerValidation {

    /**
     * Asserts that the layer nIn and nOut values are set for the layer
     *
     * @param layerType     Type of layer ("DenseLayer", etc)
     * @param layerName     Name of the layer (may be null if not set)
     * @param layerIndex    Index of the layer
     * @param nIn           nIn value
     * @param nOut          nOut value
     */
    public static void assertNInNOutSet(String layerType, String layerName, int layerIndex, int nIn, int nOut) {
        if (nIn <= 0 || nOut <= 0) {
            if (layerName == null)
                layerName = "(name not set)";
            throw new DL4JInvalidConfigException(layerType + " (index=" + layerIndex + ", name=" + layerName + ") nIn="
                            + nIn + ", nOut=" + nOut + "; nIn and nOut must be > 0");
        }
    }
}
