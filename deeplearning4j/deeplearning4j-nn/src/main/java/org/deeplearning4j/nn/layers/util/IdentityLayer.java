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

package org.deeplearning4j.nn.layers.util;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Identity layer, passes data through unaltered. This is a pure utility layer needed to support
 * Keras Masking layer import. To do so we wrap this identity layer into a MaskZeroLayer and apply
 * masks accordingly.
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class IdentityLayer extends SameDiffLambdaLayer {

    public IdentityLayer(String name) {
        this.layerName = name;
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
        return layerInput;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) { return inputType; }
}
