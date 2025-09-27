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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class Dilation2D extends DynamicCustomOp {
    protected boolean isSameMode;

    // rates
    protected int r0, r1, r2, r3;

    // strides
    protected int s0, s1, s2, s3;


    public Dilation2D() {
    }

    public Dilation2D(SameDiff sameDiff, SDVariable df, SDVariable weights, int[] strides, int[] rates, boolean isSameMode) {
        this(sameDiff, new SDVariable[]{df, weights}, strides, rates, isSameMode, false);
    }

    public Dilation2D(SameDiff sameDiff, SDVariable[] inputAndWeights, int[] strides,
                      int[] rates, boolean isSameMode, boolean inPlace ) {
        super(null, sameDiff, inputAndWeights, inPlace);
        Preconditions.checkArgument(rates.length == 4,
                "Dilation rate length must be 4, got an array with length %s with values %s", rates.length, rates);
        Preconditions.checkArgument(strides.length == 4,
                "Dilation strides length must be 4, got an array with length %s with values %s", strides.length, strides);

        r0 = rates[0];
        r1 = rates[1];
        r2 = rates[2];
        r3 = rates[3];
        s0 = strides[0];
        s1 = strides[1];
        s2 = strides[2];
        s3 = strides[3];
        this.isSameMode = isSameMode;

        addArgs();

    }

    public Dilation2D(INDArray[] inputArrays, INDArray[] outputs) {
        super(null, inputArrays, outputs);

    }

    public Dilation2D(INDArray df, INDArray weights, int[] strides, int[] rates,  boolean isSameMode) {
        addInputArgument(df, weights);

        if (rates.length < 4)
            throw new IllegalArgumentException("Dilation rate length must be 4.");
        if (strides.length < 4)
            throw new IllegalArgumentException("Strides length must be 4.");

        r0 = rates[0];
        r1 = rates[1];
        r2 = rates[2];
        r3 = rates[3];
        s0 = strides[0];
        s1 = strides[1];
        s2 = strides[2];
        s3 = strides[3];
        this.isSameMode = isSameMode;

        addArgs();
    }

    protected void addArgs() {
        addIArgument(isSameMode ? 1 : 0,
                r0, r1, r2, r3,
                s0, s1, s2, s3);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val sameMode = PropertyMapping.builder()
                .tfAttrName("padding")
                .propertyNames(new String[]{"isSameMode"})
                .build();

        val ratesMapping = PropertyMapping.builder()
                .tfAttrName("rates")
                .propertyNames(new String[]{"r0", "r1", "r2", "r3"})
                .build();

        val stridesMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .propertyNames(new String[]{"s0", "s1", "s2", "s3"})
                .build();

        map.put("isSameMode", sameMode);

        map.put("r0", ratesMapping);
        map.put("r1", ratesMapping);
        map.put("r2", ratesMapping);
        map.put("r3", ratesMapping);

        map.put("s0", stridesMapping);
        map.put("s1", stridesMapping);
        map.put("s2", stridesMapping);
        map.put("s3", stridesMapping);

        try {
            ret.put(onnxName(), map);
        }catch(NoOpNameFoundException e) {
            //ignore, we dont care about onnx for this set of ops
        }


        try {
            ret.put(tensorflowName(),map);
        }catch(NoOpNameFoundException e) {
            throw new RuntimeException(e);
        }

        return ret;
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        throw new RuntimeException();

    }


    @Override
    public String opName() {
        return "dilation2d";
    }

    @Override
    public String onnxName() {
        return "Dilation_2D";
    }

    @Override
    public String tensorflowName() {
        return "Dilation2D";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){  //Input and weights, optional rates/strides
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 2 && inputDataTypes.size() <= 4,
                "Expected 2 to 4 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
