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

package org.nd4j.linalg.api.ops.impl.controlflow.compat;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.Map;

public abstract class BaseCompatOp extends DynamicCustomOp {
    protected String frameName;

    public BaseCompatOp(SameDiff sameDiff, SDVariable[] inputs){
        super(null, sameDiff, inputs);
    }

    public BaseCompatOp(){

    }

    public String getFrameName() {
        return frameName;
    }

    public void setFrameName(@NonNull String frameName) {
        this.frameName = frameName;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode,nodeDef, graph);
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
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        return super.attributeAdaptersForFunction();
    }
}
