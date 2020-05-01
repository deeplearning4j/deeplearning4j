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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.*;


/**
 * Composed op: mmul (X, W) + b
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class XwPlusB extends DynamicCustomOp {


    public XwPlusB(SameDiff sameDiff, SDVariable input, SDVariable weights, SDVariable bias) {
        super(null, sameDiff, new SDVariable[] {input, weights, bias}, false);
    }

    public XwPlusB(INDArray input, INDArray weights, INDArray bias) {
        super(new INDArray[] {input, weights, bias}, null);
    }

    public XwPlusB(INDArray[] inputs, INDArray output){
        super(inputs, wrapOrNull(output));
    }

    @Override
    public String opName() {
        return "xw_plus_b";
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for shape " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        SDVariable in = arg(0);
        SDVariable w = arg(1);
        SDVariable dLdOut = gradient.get(0);

        SDVariable dLdb = dLdOut.sum(0);
        SDVariable dLdIn = sameDiff.mmul(dLdOut, w, false, true, false);
        SDVariable dLdW = sameDiff.mmul(in, dLdOut, true, false, false);

        return Arrays.asList(dLdIn, dLdW, dLdb);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        for( int i=0; i<3; i++ ) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0){
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }
        return Collections.singletonList(first);
    }

}
