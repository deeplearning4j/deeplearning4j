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

package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.Getter;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class TensorArray extends  BaseTensorOp {

    @Getter
    protected DataType tensorArrayDataType;
    @Override
    public String tensorflowName() {
        return "TensorArrayV3";
    }

    public TensorArray(String name, SameDiff sameDiff, DataType dataType){
        super(name, sameDiff, new SDVariable[]{});
        this.tensorArrayDataType = dataType;
    }
    public TensorArray(SameDiff sameDiff, DataType dataType){
        super(sameDiff, new SDVariable[]{});
        this.tensorArrayDataType = dataType;
    }

    public TensorArray(TensorArray ta){
        super(ta.sameDiff, new SDVariable[]{});
        this.tensorArrayDataType = ta.tensorArrayDataType;
    }
    public TensorArray(TensorArray ta, SDVariable[] inputs){
        super(ta.sameDiff, inputs);
        this.tensorArrayDataType = ta.tensorArrayDataType;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val idd = nodeDef.getInput(nodeDef.getInputCount() - 1);
        NodeDef iddNode = null;
        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(idd)) {
                iddNode = graph.getNode(i);
            }
        }

        val arr = TFGraphMapper.getNDArrayFromTensor(iddNode);

        if (arr != null) {
            int idx = arr.getInt(0);
            addIArgument(idx);
        }

        this.tensorArrayDataType = TFGraphMapper.convertType(attributesForNode.get("dtype").getType());
    }


    public TensorArray(){
        this(DataType.FLOAT);
    }

    public TensorArray(DataType dataType){
        this.tensorArrayDataType = dataType;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "tensorarrayv3";
    }


    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }


    private SDVariable getVar(){
        return outputVariable();
    }

    @Override
    public SameDiff getSameDiff(){
        val sd = this.sameDiff;
        if (sd.getChild() != null){
            return sd.getChild();
        }
        return sd;
    }

    private SDVariable intToVar(int... index){
        return this.sameDiff.constant(Nd4j.createFromArray(index));
    }


    //----------- read ops-----------------\\
    public SDVariable read(int index){
        return new TensorArrayRead(getSameDiff(), new SDVariable[]{getVar(), intToVar(index)}).outputVariable();
    }
    public SDVariable read(SDVariable index){
        return new TensorArrayRead(getSameDiff(), new SDVariable[]{getVar(), index}).outputVariable();
    }
    public SDVariable gather(SDVariable flow, int... indices){
        return new TensorArrayGather(getSameDiff(), new SDVariable[]{getVar(), sameDiff.constant(Nd4j.createFromArray(indices)), flow}).outputVariable();
    }
    public SDVariable gather(SDVariable flow, SDVariable indices){
        return new TensorArrayGather(getSameDiff(), new SDVariable[]{getVar(), indices, flow}).outputVariable();
    }
    public SDVariable stack(SDVariable flow){
        return new TensorArrayGather(getSameDiff(), new SDVariable[]{getVar(), intToVar(-1), flow}).outputVariable();
    }

    public SDVariable concat(SDVariable flow){
        return new TensorArrayConcat(getSameDiff(), new SDVariable[]{getVar()}).outputVariable();
    }

    //----------- write ops-----------------\\
    public SDVariable write(SDVariable flow, int index, SDVariable value){
        return write(flow, intToVar(index), value);
    }

    public SDVariable write(SDVariable flow, SDVariable index, SDVariable value){
        return new TensorArrayWrite(getSameDiff(),
                new SDVariable[]{getVar(),
                        index, value, flow}).outputVariable();
    }

    public SDVariable scatter(SDVariable flow, SDVariable value, int... indices){
        return new TensorArrayScatter(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(indices),
                        value, flow}).outputVariable();
    }

    public SDVariable scatter(SDVariable flow, SDVariable value, SDVariable indices){
        return new TensorArrayScatter(getSameDiff(),
                new SDVariable[]{getVar(),
                        indices,
                        value, flow}).outputVariable();
    }

    public SDVariable unstack(SDVariable flow, SDVariable value){
        return new TensorArrayScatter(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(-1),
                        value, flow}).outputVariable();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataType){
        //The SDVariable that is the output of this "function" is just a dummy variable anyway...
        //Usually 2 outputs... seems like first one is dummy, second one is a float??
        //TODO work out exactly what this second output is for (it's used in TensorArrayWrite for example...
        return Arrays.asList(DataType.BOOL, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs(){
        return 2;
    }
}
