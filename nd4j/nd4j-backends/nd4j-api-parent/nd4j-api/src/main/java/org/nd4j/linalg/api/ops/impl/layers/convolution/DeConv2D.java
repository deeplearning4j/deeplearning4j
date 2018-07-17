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
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * DeConv2D operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class DeConv2D extends DynamicCustomOp {

    protected DeConv2DConfig config;

    @Builder(builderMethodName = "builder")
    public DeConv2D(SameDiff sameDiff,
                    SDVariable[] inputs,
                    INDArray[] inputArrays, INDArray[] outputs,
                    DeConv2DConfig config) {
        super(null, inputArrays, outputs);
        this.sameDiff = sameDiff;
        this.config = config;

        if (inputArrays != null) {
            addInputArgument(inputArrays);
        }
        if (outputs != null) {
            addOutputArgument(outputs);
        }

        addArgs();
        sameDiff.putFunctionForId(this.getOwnName(), this);
        sameDiff.addArgsFor(inputs, this);
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    private void addArgs() {
        addIArgument(config.getKH());
        addIArgument(config.getKW());
        addIArgument(config.getSH());
        addIArgument(config.getSW());
        addIArgument(config.getPH());
        addIArgument(config.getPW());
        addIArgument(config.getDH());
        addIArgument(config.getDW());
        addIArgument(ArrayUtil.fromBoolean(config.isSameMode()));

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
    public Object getValue(Field property) {
        if (config == null) {
            config = DeConv2DConfig.builder().build();
        }

        return config.getValue(property);
    }

    @Override
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target, value);
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .build();

        val kernelMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"kH", "kW"})
                .tfInputPosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dW", "dH"})
                .tfAttrName("rates")
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
        map.put("kH", kernelMapping);
        map.put("kW", kernelMapping);
        map.put("dW", dilationMapping);
        map.put("dH", dilationMapping);
        map.put("isSameMode", sameMode);
        map.put("pH", paddingWidthHeight);
        map.put("pW", paddingWidthHeight);

        ret.put(onnxName(), map);
        ret.put(tensorflowName(), map);
        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        int sH = 1;
        int sW = 1;
        int kH = 1;
        int kW = 1;

        val aPadding = nodeDef.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();

        val args = args();
        INDArray arr = sameDiff.getVariable(args[1].getVarName()).getArr();
        if (arr == null) {
            arr = TFGraphMapper.getInstance().getNDArrayFromTensor(nodeDef.getInput(0), nodeDef, graph);
            // TODO: arguable. it might be easier to permute weights once
            //arr = (arr.permute(3, 2, 0, 1).dup('c'));
            val varForOp = initWith.getVariable(args[1].getVarName());
            if (arr != null)
                initWith.associateArrayWithVariable(arr, varForOp);


        }

        String dataFormat = "nhwc";
        if (nodeDef.containsAttr("data_format")) {
            val attr = nodeDef.getAttrOrThrow("data_format");
            dataFormat = attr.getS().toStringUtf8().toLowerCase();
        }

        // FIXME: int cast


        if (dataFormat.equalsIgnoreCase("nchw")) {
            sH = tfStrides.get(2).intValue();
            sW = tfStrides.get(3).intValue();

            kH = (int) arr.size(2);
            kW = (int) arr.size(3);
        } else {
            sH = tfStrides.get(1).intValue();
            sW = tfStrides.get(2).intValue();

            kH = (int) arr.size(0);
            kW = (int) arr.size(1);
        }


        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        DeConv2DConfig conv2DConfig = DeConv2DConfig.builder()
                .kH(kH)
                .kW(kW)
                .sH(sW)
                .sW(sH)
                .isSameMode(isSameMode)
                //c++ check checks for nchw
                .isNHWC(dataFormat.equalsIgnoreCase("nhwc"))
                .build();
        this.config = conv2DConfig;

        addArgs();


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val autoPad = !attributesForNode.containsKey("auto_pad") ? "VALID" : attributesForNode.get("auto_pad").getS().toStringUtf8();
        val dilations = attributesForNode.get("dilations");
        val dilationY = dilations == null ? 1 : dilations.getIntsList().get(0).intValue();
        val dilationX = dilations == null ? 1 : dilations.getIntsList().get(1).intValue();
        val group = attributesForNode.get("group");

        val kernelShape = attributesForNode.get("kernel_shape");
        int kH = kernelShape.getIntsList().get(0).intValue();
        int kW = kernelShape.getIntsList().size() < 2 ? kH : kernelShape.getIntsList().get(1).intValue();

        val vertexId = args()[0];

        INDArray arr = vertexId.getArr();
        arr = (arr.permute(3, 2, 0, 1).dup('c'));
        initWith.associateArrayWithVariable(arr, vertexId);

        String dataFormat = "nhwc";

        val strides = attributesForNode.get("strides");
        val sH = strides.getIntsList().get(0);
        val sW = strides.getIntsList().size() < 2 ? sH : strides.getIntsList().get(1);
        boolean isSameMode = autoPad
                .equalsIgnoreCase("SAME");


        DeConv2DConfig conv2DConfig = DeConv2DConfig.builder()
                .kH(kH)
                .kW(kW)
                .sH(sH.intValue())
                .sW(sW.intValue())
                .isSameMode(isSameMode)
                //c++ check checks for nchw
                .isNHWC(dataFormat.equalsIgnoreCase("nhwc"))
                .build();
        this.config = conv2DConfig;

        addArgs();

        addOutputArgument(arr);
    }


    @Override
    public String opName() {
        return "deconv2d";
    }

    @Override
    public String onnxName() {
        return "ConvTranspose";
    }

    @Override
    public String tensorflowName() {
        return "Conv2DTranspose";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.addAll(f1);
        DeConv2DDerivative deConv2DDerivative = DeConv2DDerivative.derivativeBuilder()
                .sameDiff(sameDiff)
                .config(config)
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(deConv2DDerivative.outputVariables()));
        return ret;
    }

}
