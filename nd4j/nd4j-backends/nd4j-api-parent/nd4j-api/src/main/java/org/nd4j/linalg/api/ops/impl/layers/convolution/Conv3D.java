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
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.*;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * Conv3D operation
 */
@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {

    protected Conv3DConfig config;

    public Conv3D() {
    }

    @Builder(builderMethodName = "builder")
    public Conv3D(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs,
                  Conv3DConfig conv3DConfig) {
        super(null, sameDiff, inputFunctions, false);
        setSameDiff(sameDiff);

        if (inputs != null)
            addInputArgument(inputs);
        if (outputs != null)
            addOutputArgument(outputs);
        this.config = conv3DConfig;
        addArgs();


        //for (val arg: iArgs())
        //  System.out.println(getIArgument(arg));
    }


    private void addArgs() {
        addIArgument(
                // TODO: support bias terms
//                ArrayUtil.fromBoolean(getConfig().isBiasUsed()),
                getConfig().getKD(),
                getConfig().getKH(),
                getConfig().getKW(),

                getConfig().getSD(),
                getConfig().getSH(),
                getConfig().getSW(),

                getConfig().getPD(),
                getConfig().getPH(),
                getConfig().getPW(),

                getConfig().getDD(),
                getConfig().getDH(),
                getConfig().getDW(),

                getConfig().isSameMode() ? 1 : 0,
                getConfig().isNCDHW() ? 0 : 1
        );
    }


    @Override
    public Object getValue(Field property) {
        if (config == null && !iArguments.isEmpty()) {
            config = Conv3DConfig.builder()
                    .kD(iArguments.get(0))
                    .kH(iArguments.get(1))
                    .kW(iArguments.get(2))
                    .sD(iArguments.get(3))
                    .sH(iArguments.get(4))
                    .sW(iArguments.get(5))
                    .pD(iArguments.get(6))
                    .pH(iArguments.get(7))
                    .pW(iArguments.get(8))
                    .dD(iArguments.get(9))
                    .dH(iArguments.get(10))
                    .dW(iArguments.get(11))
                    .isSameMode(iArguments.get(12) == 1)
                    .dataFormat(iArguments.get(13) == 1 ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                    .build();
        }

        return config.getValue(property);
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String, AttributeAdapter> tfAdapters = new LinkedHashMap<>();
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);

        //TF uses [kD, kH, kW, iC, oC] for weights
        tfAdapters.put("kD", new NDArrayShapeAdapter(0));
        tfAdapters.put("kH", new NDArrayShapeAdapter(1));
        tfAdapters.put("kW", new NDArrayShapeAdapter(2));

        tfAdapters.put("sD", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("sH", new IntArrayIntIndexAdpater(2));
        tfAdapters.put("sW", new IntArrayIntIndexAdpater(3));

        tfAdapters.put("pD", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("pH", new IntArrayIntIndexAdpater(2));
        tfAdapters.put("pW", new IntArrayIntIndexAdpater(3));


        tfAdapters.put("isSameMode", new StringNotEqualsAdapter("VALID"));

        ret.put(tensorflowName(), tfAdapters);

        return ret;
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if (config == null) {
            return Collections.emptyMap();
        }
        return config.toProperties();
    }

    @Override
    public String opName() {
        return "conv3dnew";
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();


        val kernelMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"kD", "kW", "kH"})
                .tfInputPosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"sD", "sW", "sH"})
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dD", "dH", "dW"})
                .tfAttrName("rates")
                .build();

        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pD", "pW", "pH"})
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();


        val outputPadding = PropertyMapping.builder()
                .propertyNames(new String[]{"aD", "aH", "aW"})
                .build();


        val biasUsed = PropertyMapping.builder()
                .propertyNames(new String[]{"biasUsed"})
                .build();


        for (val propertyMapping : new PropertyMapping[]{
                kernelMapping,
                strideMapping,
                dilationMapping,
                sameMode,
                paddingWidthHeight,
                dataFormat,
                outputPadding, biasUsed}) {
            for (val keys : propertyMapping.getPropertyNames())
                map.put(keys, propertyMapping);
        }

        ret.put(onnxName(), map);
        ret.put(tensorflowName(), map);
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv3DDerivative conv3DDerivative = Conv3DDerivative.derivativeBuilder()
                .conv3DConfig(config)
                .inputFunctions(args())
                .outputs(outputArguments())
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        ret.addAll(Arrays.asList(conv3DDerivative.outputVariables()));
        return ret;
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        if (numIArguments() < 1) {
            addArgs();
        }

        if (numInputArguments() < getDescriptor().getNumIArgs()) {
            populateInputsAndOutputsFromSameDiff();
        }


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
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Conv3D";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types, got %s", n, inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
