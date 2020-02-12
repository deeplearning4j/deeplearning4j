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

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;

import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * Max Pooling3D operation
 *
 * @author Alex Black
 */
@Slf4j
@Getter
public class MaxPooling3D extends Pooling3D {
    public MaxPooling3D() {
        super(null, null, input, null, false, pooling3DConfig, Pooling3DType.AVG);
    }

    public MaxPooling3D(SameDiff sameDiff, SDVariable input, Pooling3DConfig config) {
        super(sameDiff, new SDVariable[]{input}, null, null, false, config, Pooling3DType.MAX);
    }

    public MaxPooling3D(SameDiff sameDiff,@NonNull INDArray arrayInput, INDArray arrayOutput,@NonNull Pooling3DConfig config) {
        super(sameDiff, null, new INDArray[]{arrayInput}, wrapOrNull(arrayOutput), false, config, Pooling3DType.MAX);
    }

    public MaxPooling3D(@NonNull INDArray input,@NonNull Pooling3DConfig pooling3DConfig) {
        super(null, null, new INDArray[]{input},null, false, pooling3DConfig, Pooling3DType.MAX);

    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }


    public String getPoolingPrefix() {
        return "max";
    }

    @Override
    public String opName() {
        return "maxpool3dnew";
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public String tensorflowName() {
        return "MaxPool3D";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
