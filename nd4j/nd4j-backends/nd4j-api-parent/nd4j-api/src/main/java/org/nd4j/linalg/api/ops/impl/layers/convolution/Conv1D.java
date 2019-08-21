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
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueIntIndexArrayAdapter;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueNDArrayShapeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.SizeThresholdIntArrayIntIndexAdpater;
import org.nd4j.imports.descriptors.properties.adapters.StringEqualsAdapter;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * Conv2D operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class Conv1D extends DynamicCustomOp {

    protected Conv1DConfig config;
    private static final String INVALID_CONFIGURATION = "Invalid Conv1D configuration : s = %s p = %s ";

    @Builder(builderMethodName = "builder")
    public Conv1D(SameDiff sameDiff,
                  SDVariable[] inputFunctions,
                  INDArray[] inputArrays, INDArray[] outputs,
                  Conv1DConfig config) {
        super(null, inputArrays, outputs);
        this.sameDiff = sameDiff;
        this.config = config;
        Preconditions.checkState(config.getS() >= 1 && config.getP() >= 0, INVALID_CONFIGURATION, config.getS(), config.getP());
        addArgs();
        sameDiff.putOpForId(this.getOwnName(), this);
        sameDiff.addArgsFor(inputFunctions, this);
    }

    protected void addArgs() {
        if (config == null)
            config = Conv1DConfig.builder().build();

        addIArgument(config.getK(),
                config.getS(),
                config.getP(),
                ArrayUtil.fromBoolean(config.isSameMode()),
                ArrayUtil.fromBoolean(config.isNWC()));
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Object getValue(Field property) {
        if (config == null && !iArguments.isEmpty()) {
            config = Conv1DConfig.builder()
                    .s(iArguments.get(0))
                    .p(iArguments.get(1))
                    .isSameMode(iArguments.get(2) == 1)
                    .dataFormat(iArguments.get(3) == 1 ? Conv1DConfig.NCW : Conv1DConfig.NWC)
                    .build();
        }

        return config.getValue(property);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
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
    public String opName() {
        return "conv1d";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No ONNX op name found for: " + getClass().getName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
