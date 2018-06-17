package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.list.compat.TensorList;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

public class TensorArrayV3 extends  BaseTensorOp {

    TensorList list;
    @Override
    public String tensorflowName() {
        return "TensorArrayV3";
    }

    public TensorArrayV3(String name, SameDiff sameDiff){
        super(name, sameDiff, new SDVariable[]{});
        this.list = new TensorList(this.getOwnName());
    }
    public TensorArrayV3(SameDiff sameDiff){
        super(sameDiff, new SDVariable[]{});
        this.list = new TensorList(this.getOwnName());
    }

    public TensorArrayV3(TensorArrayV3 ta){
        super(ta.sameDiff, new SDVariable[]{});
        this.list = ta.list;
    }
    public TensorArrayV3(TensorArrayV3 ta, SDVariable[] inputs){
        super(ta.sameDiff, inputs);
        this.list = ta.list;
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


        val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",iddNode,graph);

        if (arr != null) {
            int idx = arr.getInt(0);
            addIArgument(idx);
        }

    }


    public TensorArrayV3(){
        this.list = new TensorList(this.getOwnName());
    }

    @Override
    public TensorList execute(SameDiff sameDiff) {
        /*val list = new TensorList(this.getOwnName());

        // we might have size here

        return list;*/
        return this.list;
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
        // get var associated with the tensor list

        getSameDiff().putListByName(list.getName(), list);
        val name = list.getName();
        if (getSameDiff().variableMap().containsKey(name)){
            return getSameDiff().variableMap().get(name);
        }
        return getSameDiff().var(list.getName(), new long[]{1});
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
        return this.sameDiff.var(Nd4j.create(ArrayUtil.toDouble(index)));
    }


    //----------- read ops-----------------\\
    public SDVariable read(int index){
        return new TensorArrayReadV3(getSameDiff(), new SDVariable[]{getVar(), intToVar(index)}).outputVariable();
    }
    public SDVariable read(SDVariable index){
        return new TensorArrayReadV3(getSameDiff(), new SDVariable[]{getVar(), index}).outputVariable();
    }
    public SDVariable gather(int... indices){
        return new TensorArrayGatherV3(getSameDiff(), new SDVariable[]{getVar(), intToVar(indices)}).outputVariable();
    }
    public SDVariable gather(SDVariable indices){
        return new TensorArrayGatherV3(getSameDiff(), new SDVariable[]{getVar(), indices}).outputVariable();
    }
    public SDVariable stack(){
        return new TensorArrayGatherV3(getSameDiff(), new SDVariable[]{getVar(), intToVar(-1)}).outputVariable();
    }

    public SDVariable concat(){
        return new TensorArrayConcatV3(getSameDiff(), new SDVariable[]{getVar()}).outputVariable();
    }

    //----------- write ops-----------------\\
    public void write(int index, SDVariable value){
        //return new TensorArrayV3(this,
        new TensorArrayWriteV3(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(index), value}).outputVariables();//);

    }
    public void write(SDVariable index, SDVariable value){
        System.out.println("TA write  - " + this.sameDiff);
        //return new TensorArrayV3(this,
        new TensorArrayWriteV3(getSameDiff(),
                new SDVariable[]{getVar(),
                        index, value}).outputVariables();//);

    }
    public void scatter(SDVariable value, int... indices){
        //return new TensorArrayV3(this,
        new TensorArrayScatterV3(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(indices),
                        value}).outputVariables();//);
    }
    public void scatter(SDVariable value, SDVariable indices){
        //return new TensorArrayV3(this,
        new TensorArrayScatterV3(getSameDiff(),
                new SDVariable[]{getVar(),
                        indices,
                        value}).outputVariables();//);
    }
    public void unstack(SDVariable value){
        //return new TensorArrayV3(this,
        new TensorArrayScatterV3(getSameDiff(),
                new SDVariable[]{getVar(),
                        intToVar(-1),
                        value}).outputVariables();//);
    }
}
