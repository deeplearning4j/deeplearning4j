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
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * BatchNorm operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class BatchNorm extends DynamicCustomOp {

    private boolean applyGamma;
    private boolean applyBeta;
    private double epsilon;

    @Builder(builderMethodName = "builder")
    public BatchNorm(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[]
            outputArrays, boolean inPlace, boolean applyGamma, boolean applyBeta, double epsilon) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.sameDiff = sameDiff;

        this.applyGamma = applyGamma;
        this.applyBeta = applyBeta;
        this.epsilon = epsilon;
        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }
        if(outputArrays != null) {
            addOutputArgument(outputArrays);
        }
        addArgs();
    }

    public void addArgs() {
        addIArgument(ArrayUtil.fromBoolean(applyGamma));
        addIArgument(ArrayUtil.fromBoolean(applyBeta));
        addTArgument(epsilon);
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("applyGamma", applyGamma);
        ret.put("applyBeta", applyBeta);
        ret.put("epsilon", epsilon);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        //Switch order: TF uses [input, gamma, beta, mean, variance]; libnd4j expects [input, mean, variance, gamma, beta]
        String[] inputs = initWith.getInputsForFunction(this);
        String[] orig = inputs.clone();
        inputs[1] = orig[3];    //Mean
        inputs[2] = orig[4];    //Variance
        inputs[3] = orig[1];    //gamma
        inputs[4] = orig[2];    //beta
        addArgs();
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode, node, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "batchnorm";
    }

    @Override
    public String onnxName() {
        return "BatchNormalization";
    }

    @Override
    public String tensorflowName() {
        return "FusedBatchNorm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        BatchNormDerivative batchNormDerivative = BatchNormDerivative.derivativeBuilder()
                .applyGamma(applyGamma)
                .applyBeta(applyBeta)
                .epsilon(epsilon)
                .build();
        ret.addAll(Arrays.asList(batchNormDerivative.outputVariables()));
        return ret;
    }

}
