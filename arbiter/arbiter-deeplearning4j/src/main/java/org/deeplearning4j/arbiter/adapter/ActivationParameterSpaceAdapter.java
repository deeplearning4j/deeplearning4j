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

package org.deeplearning4j.arbiter.adapter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.adapter.ParameterSpaceAdapter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple class to adapt a {@link Activation} parameter space to a {@link IActivation} parameter space
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class ActivationParameterSpaceAdapter extends ParameterSpaceAdapter<Activation, IActivation> {

    private ParameterSpace<Activation> activation;

    public ActivationParameterSpaceAdapter(@JsonProperty("activation") ParameterSpace<Activation> activation) {
        this.activation = activation;
    }

    @Override
    public IActivation convertValue(Activation from) {
        return from.getActivationFunction();
    }

    @Override
    protected ParameterSpace<Activation> underlying() {
        return activation;
    }
}
