/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.util.OneTimeLogger;

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

    private LayerValidation(){ }

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




    public static void generalValidation(String layerName, Layer layer, IDropout iDropOut,
                                         Double l2, Double l2Bias, Double l1, Double l1Bias,
                                         Distribution dist, List<LayerConstraint> allParamConstraints,
                                         List<LayerConstraint> weightConstraints, List<LayerConstraint> biasConstraints) {
        generalValidation(layerName, layer, iDropOut,
                        l2 == null ? Double.NaN : l2, l2Bias == null ? Double.NaN : l2Bias,
                        l1 == null ? Double.NaN : l1, l1Bias == null ? Double.NaN : l1Bias, dist, allParamConstraints, weightConstraints, biasConstraints);
    }

    public static void generalValidation(String layerName, Layer layer, IDropout iDropout,
                                         double l2, double l2Bias, double l1, double l1Bias,
                                         Distribution dist, List<LayerConstraint> allParamConstraints,
                                         List<LayerConstraint> weightConstraints, List<LayerConstraint> biasConstraints) {

        if (layer != null) {
            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                configureBaseLayer(layerName, bLayer, iDropout, l2, l2Bias, l1, l1Bias, dist);
            } else if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
                configureBaseLayer(layerName, bLayer, iDropout, l2, l2Bias, l1, l1Bias, dist);
            } else if (layer instanceof Bidirectional){
                Bidirectional l = (Bidirectional)layer;
                generalValidation(layerName, l.getFwd(), iDropout, l2, l2Bias, l1, l1Bias, dist, allParamConstraints,
                        weightConstraints, biasConstraints);
                generalValidation(layerName, l.getBwd(), iDropout, l2, l2Bias, l1, l1Bias, dist, allParamConstraints,
                        weightConstraints, biasConstraints);
            }

            if(layer.getConstraints() == null || layer.constraints.isEmpty()) {
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

                if(!allConstraints.isEmpty()){
                    layer.setConstraints(allConstraints);
                } else {
                    layer.setConstraints(null);
                }
            }
        }
    }

    private static void configureBaseLayer(String layerName, BaseLayer bLayer, IDropout iDropout, Double l2, Double l2Bias,
                                           Double l1, Double l1Bias,
                    Distribution dist) {

        if (!Double.isNaN(l1) && Double.isNaN(bLayer.getL1())) {
            bLayer.setL1(l1);
        }
        if (!Double.isNaN(l2) && Double.isNaN(bLayer.getL2())) {
            bLayer.setL2(l2);
        }
        if (!Double.isNaN(l1Bias) && Double.isNaN(bLayer.getL1Bias())) {
            bLayer.setL1Bias(l1Bias);
        }
        if (!Double.isNaN(l2Bias) && Double.isNaN(bLayer.getL2Bias())) {
            bLayer.setL2Bias(l2Bias);
        }

        if (Double.isNaN(l2) && Double.isNaN(bLayer.getL2())) {
            bLayer.setL2(0.0);
        }
        if (Double.isNaN(l1) && Double.isNaN(bLayer.getL1())) {
            bLayer.setL1(0.0);
        }
        if (Double.isNaN(l2Bias) && Double.isNaN(bLayer.getL2Bias())) {
            bLayer.setL2Bias(0.0);
        }
        if (Double.isNaN(l1Bias) && Double.isNaN(bLayer.getL1Bias())) {
            bLayer.setL1Bias(0.0);
        }

        if(bLayer.getIDropout() == null){
            bLayer.setIDropout(iDropout);
        }


        if (bLayer.getWeightInit() == WeightInit.DISTRIBUTION) {
            if (dist != null && bLayer.getDist() == null)
                bLayer.setDist(dist);
            else if (dist == null && bLayer.getDist() == null) {
                bLayer.setDist(new NormalDistribution(0, 1));
                OneTimeLogger.warn(log, "Layer \"" + layerName
                                + "\" distribution is automatically set to normalize distribution with mean 0 and variance 1.");
            }
        } else if ((dist != null || bLayer.getDist() != null)) {
            OneTimeLogger.warn(log, "Layer \"" + layerName
                            + "\" distribution is set but will not be applied unless weight init is set to WeighInit.DISTRIBUTION.");
        }
    }
}
