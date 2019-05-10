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

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;

/**
 *
 *
 * @author Adam Gibson
 */
public class UpdaterCreator {

    private UpdaterCreator() {}

    public static org.deeplearning4j.nn.api.Updater getUpdater(Model layer) {
        if (layer instanceof MultiLayerNetwork) {
            return new MultiLayerUpdater((MultiLayerNetwork) layer);
        } else if (layer instanceof ComputationGraph) {
            return new ComputationGraphUpdater((ComputationGraph) layer);
        } else {
            return new LayerUpdater((Layer) layer);
        }
    }

}
