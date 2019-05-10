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

package org.deeplearning4j.nn.layers.samediff.testlayers;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class SameDiffMSELossLayer extends SameDiffOutputLayer {
    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable labels, Map<String, SDVariable> paramTable) {
        //MSE: 1/nOut * (input-labels)^2
        SDVariable diff = layerInput.sub(labels);
        return diff.mul(diff).mean(1).sum(0);
    }

    @Override
    public String activationsVertexName() {
        return "input";
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        //No op for loss layer (no params)
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        //No op for loss layer (no params)
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return null;
    }
}
