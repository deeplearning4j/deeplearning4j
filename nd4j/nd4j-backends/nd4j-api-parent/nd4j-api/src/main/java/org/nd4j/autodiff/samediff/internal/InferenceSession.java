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

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray,DifferentialFunction> {

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public INDArray[] getOutputs(DifferentialFunction op) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


}
