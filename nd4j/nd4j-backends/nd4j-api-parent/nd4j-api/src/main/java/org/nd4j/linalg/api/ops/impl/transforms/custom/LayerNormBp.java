/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * Composed op: g*standarize(x) + b
 *
 * Bias is optional, and can be set as null
 *
 * @author Paul Dubs
 */
@NoArgsConstructor
public class LayerNormBp extends DynamicCustomOp {

    private boolean noBias = false;


    public LayerNormBp(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable bias, SDVariable gradient, int... dimensions) {
        super(null, sameDiff, new SDVariable[] {input, gain, bias, gradient}, false);
        Preconditions.checkArgument(bias != null, "LayerNormBp: Use constructor without bias argument if bias is null / not available.");

        setDimensions(dimensions);
    }

    public LayerNormBp(INDArray input, INDArray gain, INDArray bias, INDArray grad, INDArray dLdx, INDArray dLdg, INDArray dLdb, int... dimensions) {
        super("layer_norm_bp", new INDArray[]{input, gain, bias, grad}, new INDArray[]{dLdx, dLdg, dLdb});
        Preconditions.checkArgument(bias != null, "LayerNormBp: Use constructor without bias argument if bias is null / not available.");

        setDimensions(dimensions);
    }


    public LayerNormBp(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable gradient, int... dimensions) {
        super(null, sameDiff, new SDVariable[] {input, gain, gradient}, false);
        noBias = true;
        setDimensions(dimensions);
    }

    public LayerNormBp(INDArray input, INDArray gain, INDArray grad, INDArray dLdx, INDArray dLdg, int... dimensions) {
        super("layer_norm_bp", new INDArray[]{input, gain, grad}, new INDArray[]{dLdx, dLdg});
        noBias = true;
        setDimensions(dimensions);
    }

    @Override
    public void setDimensions(int[] dimensions) {
        Preconditions.checkArgument(dimensions != null, "LayerNormBp: You have to provide dimensions");
        Preconditions.checkArgument(dimensions.length > 0, "LayerNormBp: You have to provide dimensions");

        this.dimensions = dimensions;
        addIArgument(dimensions);
    }

    @Override
    public String opName() {
        return "layer_norm_bp";
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
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 3 && dataTypes.size() <= 4, "Expected exactly 3 or 4 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        for (DataType dataType : dataTypes) {
            Preconditions.checkState(dataType.isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            Preconditions.checkState(first == dataType, "All datatypes must be same type, got input datatypes %s", dataTypes);
        }
        return dataTypes.subList(0, dataTypes.size()-1);
    }

    @Override
    public int getNumOutputs(){
        return noBias ? 2 : 3;
    }

}
