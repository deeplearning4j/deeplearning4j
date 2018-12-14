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

package org.nd4j.linalg.api.ops.impl.transforms.dtype;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.DataTypeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Cast op wrapper. This op changes data type of input array.
 *
 * @author raver119@gmail.com
 */
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

/*
    @Override
    public void setValueFor(Field target, Object value) {
        if(value == null) {
            throw new ND4JIllegalStateException("Unable to set field " + target + " using null value!");
        }

        // FIXME!
        if (!(value instanceof DataType))
            return;

        try {
            target.set(this, (DataType) value);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
    }
    */

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    protected void addArgs() {
        addIArgument(FlatBuffersMapper.getDataTypeAsByte(typeDst));
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String,Map<String,AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String,AttributeAdapter> tfAdapters = new LinkedHashMap<>();

        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);

        tfAdapters.put("typeDst", new DataTypeAdapter());

        ret.put(tensorflowName(),tfAdapters);
        return ret;
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
    public List<LongShapeDescriptor> calculateOutputShape() {
        if (arg() != null && (arg().getArr() != null || arg().getShape() != null)) {
            if (arg().getArr() != null) {
                return Collections.singletonList(LongShapeDescriptor.fromShape(arg().getArr().shape(), DataType.fromInt(iArguments.get(0).intValue())));
            } else {
                return Collections.singletonList(LongShapeDescriptor.fromShape(arg().getShape(), DataType.fromInt(iArguments.get(0).intValue())));
            }
        }

        return Collections.emptyList();
    }

    @Override
    public void setValueFor(Field target, Object value) {
        //This is a hack around a property mapping issue - TF datatype DT_DOUBLE return attribute.getType() of DT_DOUBLE which doesn't make sense
        if(value == null || value instanceof String || value instanceof DataType){
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
            return Collections.singletonList(f().zerosLike(arg()));
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //All scalar ops: output type is same as input type
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype, got input %s", dataTypes);
        return Collections.singletonList(typeDst);
    }
}
