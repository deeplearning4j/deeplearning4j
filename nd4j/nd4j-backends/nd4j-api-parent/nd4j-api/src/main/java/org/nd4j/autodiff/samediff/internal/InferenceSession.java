package org.nd4j.autodiff.samediff.internal;

import com.google.common.collect.Iterables;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray,DifferentialFunction> {

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public INDArray[] getOutputs(DifferentialFunction op) {
//        throw new UnsupportedOperationException("Not yet implemented");
        //TODO
        int numOutputs = op.getNumOutputs();
        if(numOutputs == -1)
            numOutputs = op.outputVariables().length;
        INDArray[] out = new INDArray[numOutputs];
        for( int i=0; i<numOutputs; i++ ){
            out[i] = Nd4j.scalar(0.0f);
        }
        return out;
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName) {
        //TODO actually set inputs etc. This is just placeholder for testing order etc
        return sameDiff.getFunctionById(opName);
    }


}
