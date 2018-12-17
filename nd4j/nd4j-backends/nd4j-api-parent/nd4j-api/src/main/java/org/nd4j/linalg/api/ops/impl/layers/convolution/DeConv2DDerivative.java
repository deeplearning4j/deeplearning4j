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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 * DeConv2DDerivative operation
 */
@Slf4j
public class DeConv2DDerivative extends DeConv2D {

    public DeConv2DDerivative() {}

    @Builder(builderMethodName = "derivativeBuilder")
    public DeConv2DDerivative(SameDiff sameDiff, SDVariable[] inputs, INDArray[] inputArrays, INDArray[] outputs, DeConv2DConfig config) {
        super(sameDiff, inputs, inputArrays, outputs, config);
    }

    @Override
    public String opName() {
        return "deconv2d_bp";
    }



    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op name found for backwards.");
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found for backwards");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");

    }

    @Override
    public int getNumOutputs(){
        //Inputs: in, weights, optional bias, gradOut                      3 req, 1 optional
        //Outputs: gradAtInput, gradW, optional gradB                      2 req, 1 optional
        SDVariable[] args = args();
        return args.length - 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n-1, "Expected %s input data types, got %s", n-1, inputDataTypes);
        List<DataType> out = new ArrayList<>(n-1);
        for( int i=0; i<n-1; i++ ){
            out.add(inputDataTypes.get(i));
        }
        return out;
    }
}
