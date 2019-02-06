package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
        return "linspace";
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
