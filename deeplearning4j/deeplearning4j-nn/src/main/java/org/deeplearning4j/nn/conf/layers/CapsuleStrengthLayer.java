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

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InputType.InputTypeRecurrent;
import org.deeplearning4j.nn.conf.inputs.InputType.Type;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * An layer to get the "strength" of each capsule, that is, the probability of it being in the input.
 * This is the vector length or L2 norm of each capsule's output.
 * The lengths will not exceed one because of the squash function.
 *
 * CapsNet is from <a href="http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf">Dynamic Routing Between Capsules</a>
 *
 * @author Ryan Nett
 */
public class CapsuleStrengthLayer extends SameDiffLambdaLayer {
    @Override
    public SDVariable defineLayer(SameDiff SD, SDVariable layerInput) {
        return SD.norm2("caps_strength", layerInput, 2);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {

        if(inputType == null || inputType.getType() != Type.RNN) {
            throw new IllegalStateException("Invalid input for Capsule Strength layer (layer name = \""
                    + layerName + "\"): expect RNN input.  Got: " + inputType);
        }

        InputTypeRecurrent ri = (InputTypeRecurrent) inputType;
        return InputType.feedForward(ri.getSize());
    }

    class Builder extends SameDiffLambdaLayer.Builder<Builder>{

        @Override
        public <E extends Layer> E build() {
            return (E) new CapsuleStrengthLayer();
        }
    }

}
