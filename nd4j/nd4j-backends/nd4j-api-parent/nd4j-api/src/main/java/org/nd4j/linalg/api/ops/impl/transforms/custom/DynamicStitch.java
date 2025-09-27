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
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


public class DynamicStitch extends DynamicCustomOp {

    private int numPartitions;
    private SDVariable[] indices;
    private String[] indexNames;
    public DynamicStitch() {
    }

    public DynamicStitch(SameDiff sameDiff, SDVariable[] indices, SDVariable[] inputs) {
        super(null, sameDiff, ArrayUtils.addAll(indices, inputs), false);

        this.indices = indices;
        this.numPartitions = inputs.length;
    }

    public DynamicStitch(@NonNull INDArray[] indices, @NonNull INDArray[] inputs) {
        super(ArrayUtils.addAll(indices, inputs), null);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // DynamicPartition and DynamicStitch are mutually inverse
        SDVariable gradient = i_v.get(0);
        SDVariable[] partitionData = new SDVariable[indices.length];
        for (int i = 0; i < indices.length; i++)
            partitionData[i] = sameDiff.onesLike(indices[i]).mul(i);
        SDVariable partitions = sameDiff.dynamicStitch(indices, partitionData);

        SDVariable[] partition = sameDiff.dynamicPartition(gradient, partitions, numPartitions);
        List<SDVariable> ret = new ArrayList<>();
        for (SDVariable i : indices)
            ret.add(sameDiff.zerosLike(i));
        Collections.addAll(ret, partition);
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        this.numPartitions = (int)attributesForNode.get("N").getI();
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if (properties.containsKey("indices")) {
            //note we don't use the references here directly just the names
            //we will use the names separately in configureWithSamediff to
            //ensure the variables are from the proper samediff instance
            if(properties.get("indices") instanceof String) {
                indexNames = new String[1];
                indexNames[0] = properties.get("indices").toString();
            } else if(properties.get("indices") instanceof String[]) {
                String[] indicesGet = (String[]) properties.get("indices");
                indexNames = indicesGet;
            }

        }

        if(properties.containsKey("numPartitions")) {
            Integer numPartitions = (Integer) properties.get("numPartitions");
            this.numPartitions = numPartitions;
        }

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        throw new RuntimeException();

    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
       if(indexNames != null && indices == null) {
           indices = new SDVariable[indexNames.length];
           for(int i = 0; i < indices.length; i++) {
               indices[i] = sameDiff.getVariable(indexNames[i]);
           }
       }

    }

    @Override
    public String opName() {
        return "dynamic_stitch";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"DynamicStitch", "ParallelDynamicStitch"};
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2*numPartitions, "Expected %s input datatypes for %s partitions for %s, got %s",
                2 * numPartitions, numPartitions, getClass(), dataTypes);
        //Output type: same as (data) input type... only 1 output, however
        DataType inputType = dataTypes.get(dataTypes.size()-1);
        return Collections.singletonList(inputType);
    }
}
