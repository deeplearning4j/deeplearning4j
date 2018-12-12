package org.nd4j.autodiff.samediff.internal;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.Map;
import java.util.Set;

public class ShapeSession extends AbstractSession<LongShapeDescriptor, DifferentialFunction> {

    public ShapeSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> opInputs, Set<String> constAndPhInputs) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public LongShapeDescriptor[] getOutputs(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> inputs, Set<String> constAndPhInputs) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public LongShapeDescriptor getConstantOrVariable(String variableName) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
