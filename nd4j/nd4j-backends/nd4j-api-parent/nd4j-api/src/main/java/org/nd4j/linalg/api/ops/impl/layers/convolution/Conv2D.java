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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueIntIndexArrayAdapter;
import org.nd4j.imports.descriptors.properties.adapters.NDArrayShapeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.SizeThresholdIntArrayIntIndexAdapter;
import org.nd4j.imports.descriptors.properties.adapters.StringEqualsAdapter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

import static org.nd4j.enums.WeightsFormat.YXIO;


@Slf4j
@Getter
@NoArgsConstructor
public class Conv2D extends DynamicCustomOp {

    protected Conv2DConfig config;
    private static final String INVALID_CONFIGURATION = "Invalid Conv2D configuration : sW = %s pH = %s dW = %s ";

    public Conv2D(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights,
                  SDVariable bias, @NonNull Conv2DConfig conv2DConfig) {
        this(sameDiff, wrapFilterNull(input, weights, bias), conv2DConfig);
    }
    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv2D(SameDiff sameDiff,
                  SDVariable[] inputFunctions,
                  Conv2DConfig config) {
        super(sameDiff, inputFunctions);

        initConfig(config);
    }

    public Conv2D(INDArray[] inputs, INDArray[] outputs, Conv2DConfig config){
        super(inputs, outputs);

        initConfig(config);
    }

    public Conv2D(@NonNull INDArray input, @NonNull INDArray weights, INDArray bias, INDArray output, @NonNull Conv2DConfig config){
        this(wrapFilterNull(input, weights, bias), wrapOrNull(output), config);
    }

    public Conv2D(INDArray layerInput, INDArray weights, INDArray bias, Conv2DConfig config) {
        this(layerInput, weights, bias, null, config);
    }

    protected void initConfig(Conv2DConfig config) {
        this.config = config;

        Preconditions.checkState(config.getSW() >= 1 && config.getPH() >= 0 && config.getDW() >= 1,
                INVALID_CONFIGURATION,
                config.getSH(), config.getPH(), config.getDW());
        addArgs();
    }

    protected void addArgs() {
        if(config != null)
            addIArgument(config.getKH(),
                    config.getKW(),
                    config.getSH(),
                    config.getSW(),
                    config.getPH(),
                    config.getPW(),
                    config.getDH(),
                    config.getDW(),
                    config.getPaddingMode().index,
                    config.getDataFormat().equalsIgnoreCase("NCHW") ? 0 : 1,
                    config.getWeightsFormat().ordinal());
    }



    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(config == null) {
            Conv2DConfig.Conv2DConfigBuilder builder =  Conv2DConfig.builder();
            Long dH = getLongValueFromProperty("dH",properties);
            if(dH != null)
                builder.dH(dH);
            Long sW = getLongValueFromProperty("sW",properties);
            if(sW != null)
                builder.sW(sW);
            Long pW = getLongValueFromProperty("pW",properties);
            if(pW != null)
                builder.pW(pW);


            Long dW = getLongValueFromProperty("dW",properties);
            if(dW != null)
                builder.dW(dW);


            Long sH = getLongValueFromProperty("sH",properties);
            if(sH != null)
                builder.sH(sH);

            Long pH = getLongValueFromProperty("pH",properties);
            if(pH != null)
                builder.pH(pH);


            Long kW = getLongValueFromProperty("kW",properties);
            if(kW != null)
                builder.kW(kW);

            Long kH = getLongValueFromProperty("kH",properties);
            if(kH != null)
                builder.kH(kH);

            String paddingMode = getStringFromProperty("paddingMode",properties);
            if(paddingMode != null)
                builder.paddingMode(PaddingMode.valueOf(paddingMode));

            if(properties.containsKey("dataFormat")) {
                builder.dataFormat(properties.get("dataFormat").toString());
            }


            this.config = builder.build();

        }
    }

    @Override
    public void configureFromArguments() {
        if(config == null && iArguments.size() >= 10) {
            config = Conv2DConfig.builder()
                    .kH(iArguments.get(0))
                    .kW(iArguments.get(1))
                    .sH(iArguments.get(2))
                    .sW(iArguments.get(3))
                    .pH(iArguments.get(4))
                    .pW(iArguments.get(5))
                    .dH(iArguments.get(6))
                    .dW(iArguments.get(7))
                    .paddingMode(PaddingMode.fromNumber(iArguments.get(8).intValue()))
                    .build();
        }
    }


    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Object getValue(Field property) {
        if (config == null) {
            config = Conv2DConfig.builder().build();
        }

        return config.getValue(property);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(config != null)
            return config.toProperties();
        return Collections.emptyMap();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

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
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }


    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        throw new RuntimeException();

    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"sW", "sH"})
                .build();


        val kernelMappingH = PropertyMapping.builder()
                .propertyNames(new String[]{"kH"})
                .tfInputPosition(1)
                .shapePosition(0)
                .onnxAttrName("kernel_shape")
                .build();

        val kernelMappingW = PropertyMapping.builder()
                .propertyNames(new String[]{"kW"})
                .tfInputPosition(1)
                .shapePosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dW", "dH"})
                .tfAttrName("dilations")
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();

        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pH", "pW"})
                .build();


        map.put("sW", strideMapping);
        map.put("sH", strideMapping);
        map.put("kH", kernelMappingH);
        map.put("kW", kernelMappingW);
        map.put("dW", dilationMapping);
        map.put("dH", dilationMapping);
        map.put("isSameMode", sameMode);
        map.put("pH", paddingWidthHeight);
        map.put("pW", paddingWidthHeight);
        map.put("dataFormat", dataFormat);

        try {
            ret.put(onnxName(), map);
        } catch (NoOpNameFoundException e) {
            //ignore
        }


        try {
            ret.put(tensorflowName(), map);
        } catch (NoOpNameFoundException e) {
            //ignore
        }

        return ret;
    }


    @Override
    public String opName() {
        return "conv2d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> inputs = new ArrayList<>(Arrays.asList(args()));
        inputs.add(f1.get(0));
        if(config == null) {
            if(!iArguments.isEmpty()) {
                createConfigFromArguments();
            }
        }

        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .sameDiff(sameDiff)
                .config(config)
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        List<SDVariable> ret = Arrays.asList(conv2DDerivative.outputVariables());
        return ret;
    }


    private void createConfigFromArguments() {
        LinAlgExceptions.assertAllConfigured(this,9);
        config = Conv2DConfig.builder()
                .kH(iArguments.get(0))
                .kW(iArguments.get(1))
                .sH(iArguments.get(2))
                .sW(iArguments.get(3))
                .pH(iArguments.get(4))
                .pW(iArguments.get(5))
                .dH(iArguments.get(6))
                .dW(iArguments.get(7))
                .paddingMode(iArguments.size() < 9 ? PaddingMode.VALID : PaddingMode.fromNumber(iArguments.get(8).intValue()))
                .dataFormat(iArguments.size() < 10 ? "NCHW" : iArguments.get(9) > 0 ? "NHWC" : "NCHW")
                .weightsFormat(iArguments.size() < 11 ? YXIO : WeightsFormat.values()[iArguments.get(10).intValue()])
                .build();
    }


    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Conv2D";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Conv2D"};
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
