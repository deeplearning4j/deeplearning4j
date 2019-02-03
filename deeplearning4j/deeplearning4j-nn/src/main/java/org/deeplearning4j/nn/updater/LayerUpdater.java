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

package org.deeplearning4j.nn.updater;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * Updater for a single layer, excluding MultiLayerNetwork (which also implements the Layer interface)
 *
 * @author Alex Black
 */
@Slf4j
public class LayerUpdater extends BaseMultiLayerUpdater<Layer> {

    public LayerUpdater(Layer layer) {
        this(layer, null);
    }

    public LayerUpdater(Layer layer, INDArray updaterState) {
        super(layer, updaterState);
        if (layer instanceof MultiLayerNetwork) {
            throw new UnsupportedOperationException("Cannot use LayerUpdater for a MultiLayerNetwork");
        }

        layersByName = new HashMap<>();
        layersByName.put(layer.conf().getLayer().getLayerName(), layer);
    }

    @Override
    protected Trainable[] getOrderedLayers() {
        return new Trainable[] {(Trainable)network};
    }

    @Override
    protected INDArray getFlattenedGradientsView() {
        return network.getGradientsViewArray();
    }

    @Override
    protected INDArray getParams() {
        return network.params();
    }

    @Override
    protected boolean isMiniBatch() {
        return network.conf().isMiniBatch();
    }

    @Override
    protected boolean isSingleLayerUpdater() {
        return true;
    }
}
