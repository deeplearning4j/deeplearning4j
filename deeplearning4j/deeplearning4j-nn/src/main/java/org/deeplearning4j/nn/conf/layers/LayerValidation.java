/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Utility methods for validating layer configurations
 *
 * @author Alex Black
 */
@Slf4j
public class LayerValidation {

    private LayerValidation() {}

    /**
     * Asserts that the layer nIn and nOut values are set for the layer
     *
     * @param layerType     Type of layer ("DenseLayer", etc)
     * @param layerName     Name of the layer (may be null if not set)
     * @param layerIndex    Index of the layer
     * @param nIn           nIn value
     * @param nOut          nOut value
     */
    public static void assertNInNOutSet(String layerType, String layerName, long layerIndex, long nIn, long nOut) {
        if (nIn <= 0 || nOut <= 0) {
            if (layerName == null)
                layerName = "(name not set)";
            throw new DL4JInvalidConfigException(layerType + " (index=" + layerIndex + ", name=" + layerName + ") nIn="
                            + nIn + ", nOut=" + nOut + "; nIn and nOut must be > 0");
        }
    }

    /**
     * Asserts that the layer nOut value is set for the layer
     *
     * @param layerType     Type of layer ("DenseLayer", etc)
     * @param layerName     Name of the layer (may be null if not set)
     * @param layerIndex    Index of the layer
     * @param nOut          nOut value
     */
    public static void assertNOutSet(String layerType, String layerName, long layerIndex, long nOut) {
        if (nOut <= 0) {
            if (layerName == null)
                layerName = "(name not set)";
            throw new DL4JInvalidConfigException(layerType + " (index=" + layerIndex + ", name=" + layerName + ") nOut="
                            + nOut + "; nOut must be > 0");
        }
    }

    public static void generalValidation(String layerName, Layer layer, IDropout iDropout, List<Regularization> regularization,
                                         List<Regularization> regularizationBias, List<LayerConstraint> allParamConstraints,
                    List<LayerConstraint> weightConstraints, List<LayerConstraint> biasConstraints) {

        if (layer != null) {
            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                configureBaseLayer(layerName, bLayer, iDropout, regularization, regularizationBias);
            } else if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
                configureBaseLayer(layerName, bLayer, iDropout, regularization, regularizationBias);
            } else if (layer instanceof Bidirectional) {
                Bidirectional l = (Bidirectional) layer;
                generalValidation(layerName, l.getFwd(), iDropout, regularization, regularizationBias, allParamConstraints,
                                weightConstraints, biasConstraints);
                generalValidation(layerName, l.getBwd(), iDropout, regularization, regularizationBias, allParamConstraints,
                                weightConstraints, biasConstraints);
            }

            if (layer.getConstraints() == null || layer.constraints.isEmpty()) {
                List<LayerConstraint> allConstraints = new ArrayList<>();
                if (allParamConstraints != null && !layer.initializer().paramKeys(layer).isEmpty()) {
                    for (LayerConstraint c : allConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().paramKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                if (weightConstraints != null && !layer.initializer().weightKeys(layer).isEmpty()) {
                    for (LayerConstraint c : weightConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().weightKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                if (biasConstraints != null && !layer.initializer().biasKeys(layer).isEmpty()) {
                    for (LayerConstraint c : biasConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().biasKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                if (!allConstraints.isEmpty()) {
                    layer.setConstraints(allConstraints);
                } else {
                    layer.setConstraints(null);
                }
            }
        }
    }

    private static void configureBaseLayer(String layerName, BaseLayer bLayer, IDropout iDropout, List<Regularization> regularization,
                                           List<Regularization> regularizationBias) {
        if (regularization != null && !regularization.isEmpty()) {
            bLayer.setRegularization(regularization);
        }
        if (regularizationBias != null && !regularizationBias.isEmpty()) {
            bLayer.setRegularizationBias(regularizationBias);
        }

        if (bLayer.getIDropout() == null) {
            bLayer.setIDropout(iDropout);
        }
    }
}
