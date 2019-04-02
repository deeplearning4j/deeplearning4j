package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Linspace op - with dynamic (SDVariable) args
 * @author Alex Black
 */
public class Linspace extends DynamicCustomOp {

    private DataType dataType;

    public Linspace(SameDiff sameDiff, SDVariable from, SDVariable to, SDVariable length, DataType dataType){
        super(sameDiff, new SDVariable[]{from, to, length});
        this.dataType = dataType;
    }

    public Linspace(){ }

    @Override
    public String opName(){
        return "lin_space";
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        return Collections.singletonList(dataType);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "LinSpace";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        dataType = TFGraphMapper.convertType(attributesForNode.get("T").getType());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(f().zerosLike(arg(0)), f().zerosLike(arg(1)), f().zerosLike(arg(2)));
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(){
        INDArray l = arg(2).getArr();
        if(l == null)
            return Collections.emptyList();
        return Collections.singletonList(LongShapeDescriptor.fromShape(new long[]{l.getLong(0)}, dataType));
    }
}
