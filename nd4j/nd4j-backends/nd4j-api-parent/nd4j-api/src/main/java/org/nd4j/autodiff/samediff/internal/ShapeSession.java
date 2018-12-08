package org.nd4j.autodiff.samediff.internal;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.Map;

public class ShapeSession extends AbstractSession<LongShapeDescriptor, DifferentialFunction> {

    public ShapeSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public LongShapeDescriptor[] getOutputs(DifferentialFunction op, VarId anOutput) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName, VarId anOutput) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void preprocessPlaceholderValues(Map<String, LongShapeDescriptor> placeholderValues) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
