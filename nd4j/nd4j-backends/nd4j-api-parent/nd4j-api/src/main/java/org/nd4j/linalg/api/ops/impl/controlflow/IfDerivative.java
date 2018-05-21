package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

@NoArgsConstructor
public class IfDerivative extends If {

    private If ifDelegate;

    public IfDerivative(If ifBlock) {
        super(ifBlock);
        this.ifDelegate = ifBlock;
    }

    @Override
    public Boolean getTrueBodyExecuted() {
        return ifDelegate.trueBodyExecuted;
    }


    @Override
    public SameDiff.SameDiffFunctionDefinition getFalseBody() {
        return ifDelegate.falseBody;
    }

    @Override
    public SameDiff getFalseBodyExecution() {
        return ifDelegate.falseBodyExecution;
    }

    @Override
    public String getBlockName() {
        return ifDelegate.blockName;
    }

    @Override
    public String getFalseBodyName() {
        return ifDelegate.falseBodyName;
    }

    @Override
    public SameDiff getLoopBodyExecution() {
        return ifDelegate.loopBodyExecution;
    }

    @Override
    public SameDiff.SameDiffConditional getPredicate() {
        return ifDelegate.getPredicate();
    }

    @Override
    public SameDiff getPredicateExecution() {
        return ifDelegate.predicateExecution;
    }

    @Override
    public List<long[]> calculateOutputShape() {
        return super.calculateOutputShape();
    }

    @Override
    public String opName() {
        return "if_bp";
    }

    @Override
    public List<SDVariable> diff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException("Unable to take the derivative of the derivative for if");
    }
}
