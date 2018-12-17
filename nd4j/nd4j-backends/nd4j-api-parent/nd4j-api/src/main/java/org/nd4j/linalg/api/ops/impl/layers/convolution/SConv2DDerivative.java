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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 * SConv2DDerivative operation
 */
@Slf4j
public class SConv2DDerivative extends SConv2D {

    @Builder(builderMethodName = "sDerviativeBuilder")
    public SConv2DDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
    }

    public SConv2DDerivative() {}

    @Override
    public String opName() {
        return "sconv2d_bp";
    }

    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("No op name found for backwards");
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
        //Inputs: in, gradAtOutput, weightsDepth, optional weightsPoint, optional weightsBias       3 req, 2 optional
        //Outputs: gradAtInput, gradWD, optional gradWP, optional gradB                             2 req, 2 optional
        SDVariable[] args = args();
        return args.length - 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;  //Original inputs + gradient at
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types, got %s", n, inputDataTypes);
        List<DataType> out = new ArrayList<>(n-1);
        for( int i=0; i<n-1; i++ ){
            out.add(inputDataTypes.get(i));
        }
        return out;
    }

}
