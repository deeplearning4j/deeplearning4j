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

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.BooleanAdapter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class CumProd extends DynamicCustomOp {
    protected boolean exclusive = false;
    protected boolean reverse = false;
    protected int[] axis = new int[0];

    public CumProd() {
    }

    public CumProd(SameDiff sameDiff, SDVariable x, int... axis) {
        this(sameDiff, x, false, false, axis);
    }

    public CumProd(SameDiff sameDiff, SDVariable x, boolean exclusive, boolean reverse, int... axis) {
        super(null, sameDiff, new SDVariable[]{x, });
        this.sameDiff = sameDiff;
        this.exclusive = exclusive;
        this.reverse = reverse;
        this.axis = axis;

        tArguments.clear();
        iArguments.clear();
        addArgs();
    }

    public CumProd(INDArray in, INDArray result, boolean exclusive, boolean reverse, int... axis) {
        super(null, new INDArray[]{in}, new INDArray[]{result}, null, (List<Integer>)null);
        this.exclusive = exclusive;
        this.reverse = reverse;
        this.axis = axis;

        tArguments.clear();
        iArguments.clear();
        addArgs();
    }

    @Override
    public String opName() {
        return "cumprod";
    }


    @Override
    public String tensorflowName() {
        return "Cumprod";
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String, AttributeAdapter> tfMappings = new LinkedHashMap<>();

        tfMappings.put("exclusive", new BooleanAdapter());
        tfMappings.put("reverse", new BooleanAdapter());


        ret.put(tensorflowName(), tfMappings);

        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val exclusiveMapper = PropertyMapping.builder()
                .tfAttrName("exclusive")
                .propertyNames(new String[]{"exclusive"})
                .build();

        val reverseMapper = PropertyMapping.builder()
                .tfAttrName("reverse")
                .propertyNames(new String[]{"reverse"})
                .build();


        map.put("exclusive", exclusiveMapper);
        map.put("reverse", reverseMapper);

        ret.put(tensorflowName(), map);

        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    protected void addArgs() {
        addIArgument(exclusive ? 1 : 0, reverse ? 1 : 0);
        if (axis != null)
            for (val a: axis)
                addIArgument(a);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Collections.singletonList(f().cumprodBp(arg(0), grad.get(0), exclusive, reverse, axis));
    }
}
