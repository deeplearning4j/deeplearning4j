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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Split op
 */
public class Split extends DynamicCustomOp {

    private int numSplit;
    private int splitDim;

    public Split() {
    }

    public Split(SameDiff sameDiff, SDVariable input, int numSplit, int splitDim) {
        super(null,sameDiff,new SDVariable[]{input});
        this.numSplit = numSplit;
        this.splitDim = splitDim;
        addIArgument(numSplit,splitDim);
    }

    public Split(@NonNull INDArray in, INDArray out) {
        super(null, new INDArray[]{in}, wrapOrNull(out), null, (List<Long>)null);
    }

    public Split(INDArray input, int numSplit, int splitDim) {
        super(null,input,null,Collections.emptyList(),new long[0]);
        addIArgument(numSplit,splitDim);
        this.numSplit = numSplit;
        this.splitDim = splitDim;
    }

    public Split(SameDiff sd, SDVariable input, SDVariable numSplit, int splitDim) {
        super(sd,new SDVariable[]{input,numSplit});
        addIArgument(splitDim);
    }

    public Split(INDArray input, INDArray numSplit, int splitDim) {
        super(new INDArray[]{input,numSplit},null);
        addIArgument(splitDim);
    }


    @Override
    public String opName() {
        return "split";
    }

    @Override
    public String tensorflowName() {
        return "Split";
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        ret.put("numSplit",numSplit);
        ret.put("splitDim",splitDim);
        return ret;
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("splitDim")) {
            Integer splitDim = getIntValueFromProperty("splitDim",properties);
            this.splitDim = splitDim;
        }

        if(properties.containsKey("numSplit")) {
            Integer numSplit = getIntValueFromProperty("numSplit",properties);
            this.numSplit = numSplit;
        }
    }
    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val splitDim = PropertyMapping.builder()
                .tfInputPosition(0)
                .propertyNames(new String[]{"splitDim"})
                .build();

        val numSplit = PropertyMapping.builder()
                .tfAttrName("num_split")
                .propertyNames(new String[]{"numSplit"})
                .build();

        map.put("numSplit",numSplit);
        map.put("splitDim",splitDim);

        ret.put(tensorflowName(),map);

        return ret;
    }

    @Override
    public int getNumOutputs(){
        return numSplit;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && !dataTypes.isEmpty(), "No datatypes were provided for %s: %s", getClass(), dataTypes);
        DataType dt;
        if(dataTypes.size() == 1) {
            dt = dataTypes.get(0);
        } else {
            //Order seems to usually be axis first for TF import? libnd4j supports both...
            if(dataTypes.get(0).isIntType()){
                dt = dataTypes.get(1);
            } else {
                dt = dataTypes.get(0);
            }
        }
        //Output types are same as first input type - just numSplits of them...
        List<DataType> out = new ArrayList<>(numSplit);
        for( int i = 0; i < numSplit; i++) {
            out.add(dt);
        }
        return out;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(new Concat(sameDiff,splitDim,f1.toArray(new SDVariable[f1.size()])).outputVariables());
    }

    /**
     * Split output shapes: each output has the same shape as input, except the split dimension
     * is divided by numSplit.
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        // Get numSplit and splitDim from iArgs
        List<Long> iArgs = oc.getIArguments();
        int numSplits;
        int splitAxis;

        if (iArgs != null && iArgs.size() >= 2) {
            numSplits = iArgs.get(0).intValue();
            splitAxis = iArgs.get(1).intValue();
        } else if (iArgs != null && iArgs.size() == 1) {
            // Single iArg might be just the axis with numSplit from field
            numSplits = this.numSplit;
            splitAxis = iArgs.get(0).intValue();
        } else {
            numSplits = this.numSplit;
            splitAxis = this.splitDim;
        }

        if (numSplits <= 0) {
            return null; // Invalid - fall back to C++
        }

        INDArray input = oc.getInputArray(0);
        if (input == null) {
            return null;
        }

        long[] inputShape = input.shape();
        int rank = inputShape.length;

        // Normalize negative axis
        if (splitAxis < 0) {
            splitAxis += rank;
        }

        if (splitAxis < 0 || splitAxis >= rank) {
            return null; // Invalid axis - fall back to C++
        }

        // Check if split dimension is evenly divisible
        long splitDimSize = inputShape[splitAxis];
        if (splitDimSize % numSplits != 0) {
            return null; // Not evenly divisible - fall back to C++
        }

        long splitSize = splitDimSize / numSplits;

        // Build output shape (same as input except split dimension)
        long[] outputShape = inputShape.clone();
        outputShape[splitAxis] = splitSize;

        DataType dtype = input.dataType();
        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);

        // All outputs have the same shape
        List<DataBuffer> results = new ArrayList<>(numSplits);
        for (int i = 0; i < numSplits; i++) {
            results.add(shapeInfo);
        }
        return results;
    }
}
