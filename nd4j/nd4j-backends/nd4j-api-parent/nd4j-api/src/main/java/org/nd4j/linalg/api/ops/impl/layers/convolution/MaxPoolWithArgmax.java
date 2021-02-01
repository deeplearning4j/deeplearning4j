/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

@Slf4j
@Getter
public class MaxPoolWithArgmax extends DynamicCustomOp {

    protected Pooling2DConfig config;
    protected DataType outputType;

    public MaxPoolWithArgmax() {
    }

    @Builder(builderMethodName = "sameDiffBuilder")
    @SuppressWarnings("Used in lombok")
    public MaxPoolWithArgmax(SameDiff sameDiff, SDVariable input, Pooling2DConfig config) {
        super(null, sameDiff, new SDVariable[]{input}, false);

        config.setType(Pooling2D.Pooling2DType.MAX);
        this.config = config;
        addArgs();
    }

    public MaxPoolWithArgmax(@NonNull INDArray input, @NonNull Pooling2DConfig config){
        this(input, null, null, config);
    }

    public MaxPoolWithArgmax(@NonNull INDArray input, INDArray output,INDArray outArgMax, @NonNull Pooling2DConfig config){
        super(null, new INDArray[]{input}, wrapFilterNull(output, outArgMax));
        config.setType(Pooling2D.Pooling2DType.MAX);

        this.config = config;
        addArgs();
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
        if(config == null && !iArguments.isEmpty()) {
            //Perhaps loaded from FlatBuffers - hence we have IArgs but not Config object
            config = Pooling2DConfig.builder()
                    .kH(iArguments.get(0))
                    .kW(iArguments.get(1))
                    .sH(iArguments.get(2))
                    .sW(iArguments.get(3))
                    .pH(iArguments.get(4))
                    .pW(iArguments.get(5))
                    .dH(iArguments.get(6))
                    .dW(iArguments.get(7))
                    .isSameMode(iArguments.get(8) == 1)
                    .extra(iArguments.get(9))
                    .isNHWC(iArguments.get(10) == 1)
                    .type(Pooling2D.Pooling2DType.MAX)
                    .build();
        }
        return config.toProperties();
    }

    private void addArgs() {
        addIArgument(config.getKH(),
                config.getKW(),
                config.getSH(),
                config.getSW(),
                config.getPH(),
                config.getPW(),
                config.getDH(),
                config.getDW(),
                ArrayUtil.fromBoolean(config.isSameMode()),
                (int) config.getExtra(),
                ArrayUtil.fromBoolean(config.isNHWC())
        );

    }


    public String getPoolingPrefix() {
        return "max";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.derivativeBuilder()
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .config(config)
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.outputVariables()));
        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();

        val aKernels = nodeDef.getAttrOrThrow("ksize");
        val tfKernels = aKernels.getList().getIList();

        int sH = 0;
        int sW = 0;

        int pH = 0;
        int pW = 0;

        int kH = 0;
        int kW = 0;

        val aPadding = nodeDef.getAttrOrThrow("padding");
        val padding = aPadding.getList().getIList();

        val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"", "");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        String data_format = "nhwc";
        if (nodeDef.containsAttr("data_format")) {
            val attr = nodeDef.getAttrOrThrow("data_format");

            data_format = attr.getS().toStringUtf8().toLowerCase();
        }

        if (data_format.equalsIgnoreCase("nhwc")) {
            sH = tfStrides.get(1).intValue();
            sW = tfStrides.get(2).intValue();

            kH = tfKernels.get(1).intValue();
            kW = tfKernels.get(2).intValue();

            pH = padding.size() > 0 ? padding.get(1).intValue() : 0;
            pW = padding.size() > 0 ? padding.get(2).intValue() : 0;
        } else {
            sH = tfStrides.get(2).intValue();
            sW = tfStrides.get(3).intValue();

            kH = tfKernels.get(2).intValue();
            kW = tfKernels.get(3).intValue();

            pH = padding.size() > 0 ? padding.get(2).intValue() : 0;
            pW = padding.size() > 0 ? padding.get(3).intValue() : 0;
        }

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sH(sH)
                .sW(sW)
                .type(Pooling2D.Pooling2DType.MAX)
                .isSameMode(isSameMode)
                .kH(kH)
                .kW(kW)
                .pH(pH)
                .pW(pW)
                .isNHWC(data_format.equalsIgnoreCase("nhwc"))
                .extra(1.0) // averaging only for non-padded values
                .build();
        this.config = pooling2DConfig;
        addArgs();
        if(attributesForNode.containsKey("argmax")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("argmax").getType());
        } else {
            outputType = DataType.LONG;
        }
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

        val paddingMapping = PropertyMapping.builder()
                .onnxAttrName("padding")
                .tfAttrName("padding")
                .propertyNames(new String[]{"pH", "pW"})
                .build();

        val kernelMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"kH", "kW"})
                .tfInputPosition(1)
                .onnxAttrName("ksize")
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dW", "dH"})
                .tfAttrName("rates")
                .build();


        //data_format
        val dataFormatMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"isNHWC"})
                .tfAttrName("data_format")
                .build();

        map.put("sW", strideMapping);
        map.put("sH", strideMapping);
        map.put("kH", kernelMapping);
        map.put("kW", kernelMapping);
        map.put("dW", dilationMapping);
        map.put("dH", dilationMapping);
        map.put("pH", paddingMapping);
        map.put("pW", paddingMapping);
        map.put("isNHWC", dataFormatMapping);

        ret.put(onnxName(), map);
        ret.put(tensorflowName(), map);
        return ret;
    }

    @Override
    public String opName() {
        return "max_pool_with_argmax";
    }

    @Override
    public String onnxName() {
        return "MaxPoolWithArgmax";
    }

    @Override
    public String tensorflowName() {
        return "MaxPoolWithArgmax";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        List<DataType> result = new ArrayList<>();
        result.add(inputDataTypes.get(0));
        result.add(outputType == null ? DataType.INT : outputType);
        return result;
    }
}
