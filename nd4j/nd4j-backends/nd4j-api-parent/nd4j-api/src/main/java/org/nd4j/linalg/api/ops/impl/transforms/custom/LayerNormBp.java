/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


@NoArgsConstructor
public class LayerNormBp extends DynamicCustomOp {

    private boolean noBias = false;
    private boolean channelsFirst;


    public LayerNormBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable gain, SDVariable bias, @NonNull SDVariable gradient, boolean channelsFirst, int... dimensions) {
        super(null, sameDiff, wrapFilterNull(input, gain, bias, gradient), false);
        this.noBias = bias == null;
        this.channelsFirst = channelsFirst;
        setDimensions(dimensions);
    }

    public LayerNormBp(@NonNull INDArray input, @NonNull INDArray gain, INDArray bias, @NonNull INDArray grad, @NonNull INDArray dLdx, @NonNull INDArray dLdg, INDArray dLdb, boolean channelsFirst, int... dimensions) {
        super("layer_norm_bp", wrapFilterNull(input, gain, bias, grad), wrapFilterNull(dLdx, dLdg, dLdb));
        this.noBias = bias == null;
        this.channelsFirst = channelsFirst;
        setDimensions(dimensions);
    }


    public LayerNormBp(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable gradient, boolean channelsFirst, int... dimensions) {
        this(sameDiff, input, gain, null, gradient, channelsFirst, dimensions);
    }

    public LayerNormBp(INDArray input, INDArray gain, INDArray grad, INDArray dLdx, INDArray dLdg, boolean channelsFirst, int... dimensions) {
        this(input, gain, null, grad, dLdx, dLdg, null, channelsFirst, dimensions);
    }

    @Override
    public void setDimensions(int[] dimensions) {
        Preconditions.checkArgument(dimensions != null, "LayerNormBp: You have to provide dimensions");
        Preconditions.checkArgument(dimensions.length > 0, "LayerNormBp: You have to provide dimensions");

        this.dimensions = dimensions;
        this.iArguments.clear();
        addIArgument(dimensions);
        this.bArguments.clear();
        addBArgument(channelsFirst);
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
