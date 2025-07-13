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

package org.nd4j.linalg.api.ops.impl.transforms.dtype;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.DataTypeAdapter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

public class Cast extends BaseDynamicTransformOp {

    private DataType typeDst;

    public Cast() {
        //
    }

    public Cast(SameDiff sameDiff, SDVariable arg, @NonNull DataType dst) {
        super(sameDiff, new SDVariable[] {arg}, false);

        this.typeDst = dst;
        addArgs();
    }

    public Cast(@NonNull INDArray arg, @NonNull DataType dataType) {
        super(new INDArray[]{arg}, null);
        this.typeDst = dataType;
        addArgs();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    protected void addArgs() {
        addIArgument(FlatBuffersMapper.getDataTypeAsByte(typeDst));
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
       throw new RuntimeException();
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val dstMapping = PropertyMapping.builder()
                .tfAttrName("DstT")
                .propertyNames(new String[]{"typeDst"})
                .build();

        for(val propertyMapping : new PropertyMapping[] {dstMapping}) {
            for (val keys : propertyMapping.getPropertyNames())
                map.put(keys, propertyMapping);
        }

        ret.put(tensorflowName(),map);

        return ret;
    }

    @Override
    public void setValueFor(Field target, Object value) {
        //This is a hack around a property mapping issue - TF datatype DT_DOUBLE return attribute.getType() of DT_DOUBLE which doesn't make sense
        if(value == null || value instanceof String || value instanceof DataType) {
            super.setValueFor(target, value);
        }
    }

    @Override
    public String opName() {
        return "cast";
    }

    @Override
    public String tensorflowName() {
        return "Cast";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //If input is numerical: reverse cast. Otherwise 0
        if(arg().dataType().isFPType()){
            return Collections.singletonList(i_v.get(0).castTo(arg().dataType()));
        } else {
            return Collections.singletonList(sameDiff.zerosLike(arg()));
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //All scalar ops: output type is same as input type
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        return Collections.singletonList(typeDst);
    }
}
