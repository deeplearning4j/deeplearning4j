package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.List;

/**
 * Equivalent to tensorflow's conditional op.
 * Runs one of 2 {@link SameDiff.SameDiffFunctionBody}
 * depending on a predicate {@link org.nd4j.autodiff.samediff.SameDiff.SameDiffConditional}
 *
 *
 * @author Adam Gibson
 */
public class If extends DifferentialFunction implements CustomOp {

    @Getter
    private SameDiff.SameDiffConditional predicate;
    @Getter
    private SameDiff.SameDiffFunctionBody trueBody,falseBody;

    @Builder
    public If(SameDiff.SameDiffConditional predicate, SameDiff.SameDiffFunctionBody trueBody, SameDiff.SameDiffFunctionBody falseBody) {
        this.predicate = predicate;
        this.trueBody = trueBody;
        this.falseBody = falseBody;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "if";
    }

    @Override
    public long opHash() {
        return 0;
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<INDArray> getInputArguments() {
        return null;
    }

    @Override
    public List<INDArray> getOutputArguments() {
        return null;
    }

    @Override
    public List<Integer> getIArguments() {
        return null;
    }

    @Override
    public List<Double> getTArguments() {
        return null;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return null;
    }
}
