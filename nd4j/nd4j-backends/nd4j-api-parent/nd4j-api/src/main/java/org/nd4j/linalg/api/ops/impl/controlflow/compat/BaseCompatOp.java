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

package org.nd4j.linalg.api.ops.impl.controlflow.compat;

import java.util.Arrays;
import java.util.List;
import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public abstract class BaseCompatOp extends DynamicCustomOp {
    protected String frameName;

    public BaseCompatOp(SameDiff sameDiff, SDVariable[] inputs){
        super(null, sameDiff, inputs);
    }

    public BaseCompatOp(INDArray... inputs) {
        addInputArgument(inputs);
    }

    public BaseCompatOp() {

    }

    public String getFrameName() {
        if(numSArguments() > 0)
            return getSArgument(0);
        return frameName;
    }

    public void setFrameName(@NonNull String frameName) {
        this.frameName = frameName;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.stream(args()).sequential().map(input -> sameDiff.zerosLike(input))
                .collect(Collectors.toList());
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        if(frameName != null)
            ret.put("frameName",frameName);
        return ret;
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        super.setPropertiesForFunction(properties);
        if(properties.containsKey("frameName")) {
            String frameName = getStringFromProperty("frameName",properties);
            this.frameName = frameName;
        }
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val frameNameMapping = PropertyMapping.builder()
                .tfAttrName("frame_name")
                .onnxAttrName("frame_name") // not sure if it exists in onnx
                .propertyNames(new String[]{"frameName"})
                .build();

        map.put("frameName", frameNameMapping);

        try {
            ret.put(onnxName(), map);
        }catch(NoOpNameFoundException e) {
            //ignore, we dont care about onnx for this set of ops
        }


        try {
            ret.put(tensorflowName(),map);
        }catch(NoOpNameFoundException e) {
            //ignore
        }

        return ret;
    }

    @Override
    public void computeArrays() {
        if(sameDiff.isEagerMode()) {
            SDVariable[] args = args();
            //special work around for non existing arrays like nextiteration that aren't computed till last
            //note we do this in case shape related ops are impacted by the stub arrays during calculation
            //usually in this situation shapes can be disregarded and won't impact normal compute
            long[] shape = new long[6];
            for(int i = 0; i < shape.length; i++)
                shape[i] = 1;
            INDArray arr = Nd4j.scalar(1.0f).reshape(shape);
            outputVariables[0].setShape(arr.shape());
            sameDiff.setEagerArrForVarName(outputVariables[0].name(),arr);
        }
    }


    @Override
    public void addSArgument(String... args) {
        super.addSArgument(args);
        if(args != null && args.length >= 1) {
            setFrameName(args[0]);
        }
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        return super.attributeAdaptersForFunction();
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        throw new UnsupportedOperationException("calculateOutputShape() is not supported for control flow ops.");
    }
}
