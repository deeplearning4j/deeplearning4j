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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * LocalResponseNormalization operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class LocalResponseNormalization extends DynamicCustomOp {



    protected LocalResponseNormalizationConfig config;


    @Builder(builderMethodName = "builder")
    public LocalResponseNormalization(SameDiff sameDiff, SDVariable[] inputFunctions,
                                      INDArray[] inputs, INDArray[] outputs,boolean inPlace,
                                      LocalResponseNormalizationConfig config) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.config = config;
        if(inputs != null) {
            addInputArgument(inputs);
        }
        if(outputs!= null) {
            addOutputArgument(outputs);
        }
        addArgs();
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    private void addArgs() {
        addTArgument(config.getBias());
        addTArgument(config.getAlpha());
        addTArgument(config.getBeta());
        addIArgument(config.getDepth());
    }

    @Override
    public String opName() {
        return "lrn";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        val aAlpha = nodeDef.getAttrOrThrow("alpha");
        val aBeta = nodeDef.getAttrOrThrow("beta");
        val aBias = nodeDef.getAttrOrThrow("bias");
        val aDepth = nodeDef.getAttrOrThrow("depth_radius");

        val alpha = aAlpha.getF();
        val beta = aBeta.getF();
        val bias = aBias.getF();
        val depth = aDepth.getF();

        LocalResponseNormalizationConfig localResponseNormalizationConfig = LocalResponseNormalizationConfig.builder()
                .alpha(alpha)
                .beta(beta)
                .bias(bias)
                .depth((int) depth)
                .build();
        this.config = localResponseNormalizationConfig;
        addArgs();
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val aAlpha = attributesForNode.get("alpha");
        val aBeta = attributesForNode.get("beta");
        val aBias = attributesForNode.get("bias");
        val aDepth = attributesForNode.get("size");

        val alpha = aAlpha.getF();
        val beta = aBeta.getF();
        val bias = aBias.getF();
        val depth = aDepth.getF();

        LocalResponseNormalizationConfig localResponseNormalizationConfig = LocalResponseNormalizationConfig.builder()
                .alpha(alpha)
                .beta(beta)
                .bias(bias)
                .depth((int) depth)
                .build();
        this.config = localResponseNormalizationConfig;
        addArgs();
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        val depthMapping = PropertyMapping.builder()
                .tfAttrName("depth_radius")
                .propertyNames(new String[]{"depth"})
                .onnxAttrName("size")
                .build();

        val alphaMapping = PropertyMapping.builder()
                .tfAttrName("alpha")
                .onnxAttrName("alpha")
                .propertyNames(new String[]{"alpha"})
                .build();

        val betaMapping = PropertyMapping.builder()
                .tfAttrName("beta")
                .onnxAttrName("beta")
                .propertyNames(new String[]{"beta"})
                .build();

        val biasMapping = PropertyMapping.builder()
                .tfAttrName("bias")
                .onnxAttrName("bias")
                .propertyNames(new String[]{"bias"})
                .build();




        Map<String,PropertyMapping> map = new HashMap<>();
        map.put("depth",depthMapping);
        map.put("alpha",alphaMapping);
        map.put("beta",betaMapping);
        map.put("bias",biasMapping);


        ret.put(tensorflowName(),map);
        ret.put(onnxName(),map);
        return ret;
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable[] gradFnInputs = new SDVariable[]{arg(), f1.get(0)};
        LocalResponseNormalizationDerivative lrnGrad = LocalResponseNormalizationDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputFunctions(gradFnInputs)
                .config(config)
                .build();
        return Collections.singletonList(lrnGrad.outputVariable());
    }

    @Override
    public String onnxName() {
        return "LRN";
    }

    @Override
    public String tensorflowName() {
        return "LRN";
    }

}
