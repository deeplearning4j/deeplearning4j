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

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.DynamicPartitionBp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


public class DynamicPartition extends DynamicCustomOp {

    private int numPartitions;
    private SDVariable partitions;

    public DynamicPartition() {
    }

    public DynamicPartition(SameDiff sameDiff, SDVariable input,  SDVariable[] partitions, int numPartitions) {
        this(sameDiff, input, partitions[0], numPartitions);
    }

    public DynamicPartition(SameDiff sameDiff, SDVariable input,  SDVariable partitions, int numPartitions) {
        super(null, sameDiff, new SDVariable[] {input, partitions}, false);
        // Set numPartitions BEFORE any method that might call getNumOutputs()
        this.partitions = partitions;
        this.numPartitions = numPartitions;
        addArgs();
    }

    public DynamicPartition(@NonNull INDArray input, @NonNull INDArray partitions, int numPartitions) {
        super(new INDArray[]{input, partitions}, null);
        this.numPartitions = numPartitions;
        addArgs();
    }

    public DynamicPartition(INDArray x, INDArray [] partitions, int numPartitions){
        super(new INDArray[]{x}, null);
        this.numPartitions = numPartitions;
        addArgs();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable inputGrad = new DynamicPartitionBp(sameDiff, arg(0), arg(1), i_v.toArray(new SDVariable[0]), numPartitions)
                .outputVariables()[0];
        SDVariable partitionsGrad = sameDiff.zerosLike(arg(1));
        return Arrays.asList(inputGrad, partitionsGrad);
    }

    protected void addArgs() {
        addIArgument(numPartitions);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.numPartitions = iArguments.get(0).intValue();
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        super.setPropertiesForFunction(properties);
        if (properties.containsKey("numPartitions")) {
            Object val = properties.get("numPartitions");
            if (val instanceof Number) {
                this.numPartitions = ((Number) val).intValue();
            }
        }
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("numPartitions", numPartitions);
        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> attrs = new LinkedHashMap<>();

        val numPartitions = PropertyMapping.builder()
                .tfAttrName("num_partitions")
                .propertyNames(new String[]{"numPartitions"})
                .build();
        attrs.put("numPartitions", numPartitions);

        ret.put(tensorflowName(),attrs);
        return ret;
    }


    @Override
    public String opName() {
        return "dynamic_partition";
    }


    @Override
    public String tensorflowName() {
        return "DynamicPartition";
    }

    @Override
    public String onnxName() {
        return "Dynamic partitioning currently not supported by ONNX";
    }

    @Override
    public int getNumOutputs(){
        // Ensure numPartitions is loaded from iArguments if not set
        if (numPartitions <= 0 && !iArguments.isEmpty()) {
            numPartitions = iArguments.get(0).intValue();
        }
        return numPartitions > 0 ? numPartitions : 1;  // Default to 1 if still not set
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type: same as (data) input type
        int numOutputs = getNumOutputs();  // Use getNumOutputs() to ensure numPartitions is properly loaded
        List<DataType> out = new ArrayList<>(numOutputs);
        for( int i=0; i<numOutputs; i++ ){
            out.add(dataTypes.get(0));
        }
        return out;
    }
}
