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

package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.Map;

public abstract class SameDiffLambdaLayer extends SameDiffLayer {


    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput);

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        return defineLayer(sameDiff, layerInput);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //TODO let's try to infer output shape from input shape, using SameDiff + DefineLayer
        throw new UnsupportedOperationException("Override SameDiffLamdaLayer.getOutputType to use OutputType functionality");
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        //No op: lambda layer doesn't have parameters
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        //No op: lambda layer doesn't have parameters
    }
}
