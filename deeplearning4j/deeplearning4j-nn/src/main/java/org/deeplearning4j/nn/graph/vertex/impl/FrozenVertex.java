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

package org.deeplearning4j.nn.graph.vertex.impl;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.misc.DummyConfig;
import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

/**
 * FrozenVertex is used for the purposes of transfer learning
 * A frozen layers wraps another DL4J GraphVertex within it.
 * During backprop, the FrozenVertex is skipped, and any parameters are not be updated.
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true, exclude = {"config"})
public class FrozenVertex extends BaseWrapperVertex {
    public FrozenVertex(GraphVertex underlying) {
        super(underlying);
    }

    private transient DummyConfig config;

    @Override
    public TrainingConfig getConfig(){
        if (config == null) {
            config = new DummyConfig(getVertexName());
        }
        return config;
    }
}
