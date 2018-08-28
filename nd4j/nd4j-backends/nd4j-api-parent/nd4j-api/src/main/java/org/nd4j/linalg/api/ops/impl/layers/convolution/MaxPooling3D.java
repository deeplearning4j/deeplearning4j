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
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;

import java.lang.reflect.Field;
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
    }

    public MaxPooling3D(SameDiff sameDiff, SDVariable input, INDArray arrayInput, INDArray arrayOutput, Pooling3DConfig config) {
        super(sameDiff, new SDVariable[]{input}, new INDArray[]{arrayInput}, new INDArray[]{arrayOutput}, false, config, Pooling3DType.MAX);
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
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target, value);
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
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public String tensorflowName() {
        return "MaxPool3D";
    }
}
