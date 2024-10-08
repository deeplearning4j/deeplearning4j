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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.BooleanAdapter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.CumSumBp;
import org.nd4j.shade.guava.primitives.Longs;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class CumSum extends DynamicCustomOp {

    protected boolean exclusive = false;
    protected boolean reverse = false;
    protected long[] jaxis = new long[0];

    public CumSum() {
    }


    public CumSum(SameDiff sameDiff, SDVariable x, long... axis) {
        this(sameDiff, x, false, false, axis);
    }

    public CumSum(SameDiff sameDiff, SDVariable x,  boolean exclusive, boolean reverse, long... axis) {
        super(null, sameDiff, new SDVariable[]{x});
        this.sameDiff = sameDiff;
        this.exclusive = exclusive;
        this.reverse = reverse;
        this.jaxis = axis;
        addArgs();
    }

    public CumSum(INDArray in, INDArray result, boolean exclusive, boolean reverse, long... axis) {
        super(null, new INDArray[]{in}, wrapOrNull(result), null, (List<Long>)null);
        this.exclusive = exclusive;
        this.reverse = reverse;
        this.jaxis = axis;
        addArgs();
    }

    public CumSum(INDArray in, boolean exclusive, boolean reverse, long... axis) {
        this(in, null, exclusive, reverse, axis);
    }

    @Override
    public String opName() {
        return "cumsum";
    }

    @Override
    public String tensorflowName() {
        return "Cumsum";
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
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    protected void addArgs() {
        addIArgument(exclusive ? 1 : 0, reverse ? 1 : 0);
        for (val a: jaxis)
            addIArgument(jaxis);
    }

    @Override
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            this.jaxis = Longs.toArray(iArguments.subList(1,iArguments.size()));
            this.exclusive = iArguments.get(0) > 0;
        }


    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("jaxis")) {
            Long dimensions = getLongValueFromProperty("jaxis",properties);
            this.jaxis = new long[] {dimensions.longValue()};
        }

        if(properties.containsKey("exclusive")) {
            Long exclusive = getLongValueFromProperty("exclusive",properties);
            this.exclusive = exclusive > 0;
        }
    }



    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return new CumSumBp(sameDiff, arg(0), grad.get(0), exclusive, reverse, jaxis).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or 2 input datatype for %s, got %s", getClass(), dataTypes);    //2nd optional input - axis
        return Collections.singletonList(dataTypes.get(0));
    }

}
