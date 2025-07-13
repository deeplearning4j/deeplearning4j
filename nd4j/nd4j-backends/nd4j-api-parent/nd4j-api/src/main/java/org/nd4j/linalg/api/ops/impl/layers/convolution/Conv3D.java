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

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {

    protected Conv3DConfig config;
    private static final String INVALID_CONFIGURATION = "Invalid Conv3D configuration : sW = %s pH = %s dW = %s ";

    public Conv3D() {
    }

    public Conv3D(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights,
                  SDVariable bias, @NonNull Conv3DConfig config) {
        this(sameDiff, wrapFilterNull(input, weights, bias), config);
    }

    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv3D(SameDiff sameDiff, SDVariable[] inputFunctions, Conv3DConfig config) {
        super(sameDiff, inputFunctions);
        initConfig(config);
    }

    public Conv3D(INDArray[] inputs, INDArray[] outputs, Conv3DConfig config){
        super(inputs, outputs);
        initConfig(config);
    }

    public Conv3D(@NonNull INDArray input, @NonNull INDArray weights, INDArray bias, INDArray output, @NonNull Conv3DConfig config){
        this(wrapFilterNull(input, weights, bias), wrapOrNull(output), config);
    }

    public Conv3D(INDArray input, INDArray weights, INDArray bias, Conv3DConfig config) {
        this(wrapFilterNull(input, weights, bias), null, config);
    }

    public Conv3D(INDArray input, INDArray weights, Conv3DConfig config) {
        this(wrapFilterNull(input, weights), null, config);
    }

    private void initConfig(Conv3DConfig config){
        this.config = config;
        Preconditions.checkState(config.getSW() >= 1 && config.getPH() >= 0 && config.getDW() >= 1,
                INVALID_CONFIGURATION,
                config.getSW(), config.getPH(), config.getDW());
        addArgs();
    }



    @Override
    public void configureFromArguments() {
        if(config == null  && iArguments.size() >= 14) {
            Conv3DConfig.Conv3DConfigBuilder builder = Conv3DConfig.builder();
            builder.kD(getIArgument(0));
            builder.kH(getIArgument(1));
            builder.kW(getIArgument(2));
            builder.sD(getIArgument(3));
            builder.sH(getIArgument(4));
            builder.sW(getIArgument(5));
            builder.pD(getIArgument(6));
            builder.pH(getIArgument(7));
            builder.pW(getIArgument(8));
            builder.dD(getIArgument(9));
            builder.dH(getIArgument(10));
            builder.dW(getIArgument(11));
            builder.paddingMode(PaddingMode.fromNumber(getIArgument(12).intValue()));
            builder.dataFormat(getIArgument(13) > 0 ? "NCDHW" : "NCHDWC");
            this.config = builder.build();
        }

    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(config == null) {
            Conv3DConfig.Conv3DConfigBuilder builder = Conv3DConfig.builder();
            Long dD = getLongValueFromProperty("dD",properties);
            if(dD != null)
                builder.dD(dD);
            Long dH = getLongValueFromProperty("dH",properties);
            if(dH != null)
                builder.dH(dH);

            Long dW = getLongValueFromProperty("dW",properties);
            if(dW != null)
                builder.dW(dW);

            Long sW = getLongValueFromProperty("sW",properties);
            if(sW != null)
                builder.sW(sW);
            Long sD = getLongValueFromProperty("sD",properties);
            if(sD != null)
                builder.sD(sD);
            Long sH = getLongValueFromProperty("sH",properties);
            if(sH != null)
                builder.sH(sH);

            Long pW = getLongValueFromProperty("pW",properties);
            if(pW != null)
                builder.pW(pW);

            Long pD = getLongValueFromProperty("pD",properties);
            if(pD != null)
                builder.pD(pD);

            Long pH = getLongValueFromProperty("pH",properties);
            if(pH != null)
                builder.pH(pH);

            Long kD = getLongValueFromProperty("kD",properties);
            if(kD != null)
                builder.kD(kD);

            Long kW = getLongValueFromProperty("kW",properties);
            if(kW != null)
                builder.kW(kW);

            Long kH = getLongValueFromProperty("kH",properties);
            if(kH != null)
                builder.kH(kH);


            Boolean biasUsed = getBooleanFromProperty("biasUsed",properties);
            if(biasUsed != null)
                builder.biasUsed(biasUsed);

            if(properties.containsKey("dataFormat")) {
                builder.dataFormat(properties.get("dataFormat").toString());
            }

            if(properties.containsKey("paddingMode")) {
                builder.paddingMode(PaddingMode.VALID.valueOf(properties.get("paddingMode").toString()));
            }

            this.config = builder.build();

        }

    }

    private void addArgs() {
        if(getConfig().getPaddingMode() == null)
            getConfig().setPaddingMode(PaddingMode.VALID);
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
                getConfig().getPaddingMode().index,
                getConfig().isNCDHW() ? 0 : 1
        );
    }


    @Override
    public Object getValue(Field property) {
        if (config == null && !iArguments.isEmpty()) {
            LinAlgExceptions.assertAllConfigured(this,12);
            createConfigFromArgs();
        }

        return config.getValue(property);
    }

    public void createConfigFromArgs() {
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
                .paddingMode(PaddingMode.fromNumber(iArguments.get(12).intValue()))
                .dataFormat(iArguments.get(13) == 1 ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                .build();
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        throw new RuntimeException();

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

        ret.put(tensorflowName(), map);
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));

        if (config == null && !iArguments.isEmpty()) {
            LinAlgExceptions.assertAllConfigured(this,12);
            createConfigFromArgs();
        }


        Conv3DDerivative conv3DDerivative = Conv3DDerivative.derivativeBuilder()
                .conv3DConfig(config)
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        ret.addAll(Arrays.asList(conv3DDerivative.outputVariables()));
        return ret;
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
        throw new NoOpNameFoundException("No ONNX op name found for: " + getClass().getName());
    }

    @Override
    public String tensorflowName() {
        return "Conv3D";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
