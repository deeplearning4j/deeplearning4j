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

package org.deeplearning4j.nn.updater.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.HashMap;

/**
 * Gradient updater for ComputationGraph. Most of the functionality is shared with
 * {@link org.deeplearning4j.nn.updater.MultiLayerUpdater} via {@link BaseMultiLayerUpdater}
 *
 * @author Alex Black
 */
public class ComputationGraphUpdater extends BaseMultiLayerUpdater<ComputationGraph> {

    protected Trainable[] orderedLayers;

    public ComputationGraphUpdater(ComputationGraph graph) {
        this(graph, null);
    }

    public ComputationGraphUpdater(ComputationGraph graph, INDArray updaterState) {
        super(graph, updaterState);

        layersByName = new HashMap<>();
        Trainable[] layers = getOrderedLayers();
        for (Trainable l : layers) {
            layersByName.put(l.getConfig().getLayerName(), l);
        }
    }

    @Override
    protected Trainable[] getOrderedLayers() {
        if (orderedLayers != null) {
            return orderedLayers;
        }
        GraphVertex[] vertices = network.getVertices();

        //In CompGraph: we need to know topological ordering, so we know how parameters are laid out in the 1d view arrays
        int[] topologicalOrdering = network.topologicalSortOrder();

        Trainable[] out = new Trainable[network.getVertices().length];

        int j = 0;
        for (int i = 0; i < topologicalOrdering.length; i++) {
            GraphVertex currentVertex = vertices[topologicalOrdering[i]];
            if (currentVertex.numParams() == 0) {
                continue;
            }

            out[j++] = currentVertex;
        }
        if(j != out.length){
            out = Arrays.copyOfRange(out, 0, j);
        }

        orderedLayers = out;
        return orderedLayers;
    }

    @Override
    protected INDArray getFlattenedGradientsView() {
        if (network.getFlattenedGradients() == null) {
            network.initGradientsView();
        }
        return network.getFlattenedGradients();
    }

    @Override
    protected INDArray getParams() {
        return network.params();
    }

    @Override
    protected boolean isMiniBatch() {
        return network.conf().isMiniBatch();
    }
}
