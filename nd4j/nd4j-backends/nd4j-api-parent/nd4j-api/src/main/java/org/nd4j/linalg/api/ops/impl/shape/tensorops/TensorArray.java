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
import onnx.Onnx;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.AbstractSession;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    public TensorArray(TensorArray ta) {
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
        return "create_list";
    }


    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }


    public SDVariable getVar(){
        return outputVariable();
    }

    @Override
    public SameDiff getSameDiff() {
        val sd = this.sameDiff;
        if (sd.getChild() != null) {
            return sd.getChild();
        }
        return sd;
    }

    private SDVariable intToVar(int... index){
        return this.sameDiff.constant(Nd4j.createFromArray(index));
    }


    //----------- read ops-----------------\\
    public SDVariable read(int index) {
        return new TensorArrayRead(getSameDiff(), new SDVariable[]{getVar(), intToVar(index)}).outputVariable();
    }

    public SDVariable read(SDVariable from,SDVariable index) {
        return new TensorArrayRead(getSameDiff(), new SDVariable[]{from, index}).outputVariable();
    }

    public SDVariable read(SDVariable index) {
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

    public SDVariable concat(SDVariable flow) {
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

    public SDVariable unstack(SDVariable flow, SDVariable value) {
        return new TensorArrayScatter(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(-1),
                        value, flow}).outputVariable();
    }

    public SDVariable size( SDVariable value) {
        return new TensorArraySize(getSameDiff(),value).outputVariable();
    }

    public SDVariable remove( SDVariable value,SDVariable idx) {
        return new TensorArrayRemove(getSameDiff(),value,idx).outputVariable();
    }

    public SDVariable remove( SDVariable value,int idx) {
        return new TensorArrayRemove(getSameDiff(),value,idx).outputVariable();
    }
    public SDVariable remove( SDVariable value) {
        return remove(value,-1);
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataType) {
        //The SDVariable that is the output of this "function" is just a dummy variable anyway...
        //Usually 2 outputs... seems like first one is dummy, second one is a float??
        //TODO work out exactly what this second output is for (it's used in TensorArrayWrite for example...
        return Arrays.asList(DataType.BOOL, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs(){
        return 2;
    }


    /**
     * Returns the item at the specified index
     * in the specified list.
     * @param sd the same diff instance to use
     * @param inputs the inputs including the relevant tensor array variable and position
     * @return
     */
    public static SDVariable itemAtIndex(SameDiff sd,SDVariable[] inputs) {
        return itemAtIndex(sd,inputs,null);
    }

    /**
     * Returns the item at the specified index
     * in the specified list. The output variable
     * name to specify for the final output.
     * @param sd the same diff instance to use
     * @param inputs the inputs including the relevant tensor array variable and position
     * @param outputVarName the name of the output variable for the read
     * @return
     */
    public static SDVariable itemAtIndex(SameDiff sd,SDVariable[] inputs,String outputVarName) {
        SDVariable sequenceVar = inputs[0];
        SDVariable position = inputs.length < 2 ? sd.constant(-1) : inputs[1];
        TensorArray ta = getTensorArray(sd, sequenceVar);

        SDVariable read = ta.read(sequenceVar,position);
        for(int i = 0; i < inputs.length; i++)
            read.addControlDependency(inputs[i]);

        if(outputVarName != null) {
            read = read.rename(outputVarName);
        }

        for(int i = 0; i < inputs.length; i++)
            read.addControlDependency(inputs[i]);

        return read;
    }

    public static TensorArray getTensorArray(SameDiff sd, SDVariable sequenceVar) {
        BaseTensorOp baseTensorOp = (BaseTensorOp) sd.getVariableOutputOp(sequenceVar.name());
        TensorArray ta =  null;
        if(baseTensorOp instanceof TensorArray) {
            ta = (TensorArray)  baseTensorOp;
        } else {
            SDVariable var2 = baseTensorOp.arg(0);
            ta = (TensorArray)  sd.getVariableOutputOp(var2.name());
        }
        return ta;
    }

    /**
     * Create an {@link TensorArray} op from the given inputs,
     * note this is the same as calling {@link #createTensorArrayFrom(SameDiff, SDVariable[],String)}
     * with null. The null value will avoid renaming the output
     * @param sd the {@link SameDiff} instance to use
     * @param inputs the input variables to create a {@link TensorArray} for
     * @return the output variable for the tensor array
     */
    public static SDVariable createTensorArrayFrom(SameDiff sd,SDVariable[] inputs) {
        return createTensorArrayFrom(sd,inputs,null);
    }

    /**
     * Create an {@link TensorArray} op from the given inputs
     * @param sd the {@link SameDiff} instance to use
     * @param inputs the input variables to create a {@link TensorArray} for
     * @param outputVarName the name of the output variable to use for the final output of the loop
     * @return the output variable for the tensor array
     */
    public static SDVariable createTensorArrayFrom(SameDiff sd,SDVariable[] inputs,String outputVarName) {
        TensorArray outputVar = sd.tensorArray(inputs[0].dataType());
        SDVariable outTmp = outputVar.getVar();
        for(int i = 0; i < inputs.length; i++) {
            val write =  outputVar.write(outTmp,i,inputs[i]);
            if(outTmp != null) {
                write.addControlDependency(outTmp);
            }

            outTmp = write;
        }

        if(outputVarName != null) {
            outTmp = outTmp.rename(outputVarName);
        }

        return outTmp;
    }


}
