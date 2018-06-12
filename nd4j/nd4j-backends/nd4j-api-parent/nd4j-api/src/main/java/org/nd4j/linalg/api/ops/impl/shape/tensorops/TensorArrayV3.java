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
        sameDiff.putListByName(list.getName(), list);
        val name = list.getName();
        if (sameDiff.variableMap().containsKey(name)){
            return sameDiff.variableMap().get(name);
        }
        return sameDiff.var(list.getName(), new long[]{1});
    }

    private SDVariable intToVar(int... index){
        return this.sameDiff.var(Nd4j.create(ArrayUtil.toDouble(index)));
    }

    public SDVariable read(int index){
        return new TensorArrayReadV3(this.sameDiff, new SDVariable[]{getVar(), intToVar(index)}).outputVariable();
    }

    public TensorArrayV3 write(int index, SDVariable value){
        return new TensorArrayV3(this,
                new TensorArrayWriteV3(this.sameDiff,
                                       new SDVariable[]{getVar(),
                                       intToVar(index), value}).outputVariables());

    }

    public SDVariable stack(){
        return new TensorArrayGatherV3(this.sameDiff, new SDVariable[]{getVar(), intToVar(-1)}).outputVariable();
    }

    public SDVariable gather(int... index){
        return new TensorArrayGatherV3(this.sameDiff, new SDVariable[]{getVar(), intToVar(index)}).outputVariable();
    }



}
