package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.List;
import java.util.UUID;

/**
 * Equivalent to tensorflow's conditional op.
 * Runs one of 2 {@link SameDiff.SameDiffFunctionDefinition}
 * depending on a predicate {@link org.nd4j.autodiff.samediff.SameDiff.SameDiffConditional}
 *
 *
 * @author Adam Gibson
 */
public class If extends DifferentialFunction implements CustomOp {
    @Getter
    private SameDiff trueBlockExecution,falseBlockExecution,predicateBody;
    @Getter
    private SDVariable targetBoolean;

    @Getter
    private String blockName,falseBodyName,trueBodyName;

    private SDVariable[] inputVars;


    @Builder
    public If(String blockName,
              SameDiff parent,
              SDVariable[] inputVars,
              SameDiff.SameDiffFunctionDefinition conditionBody,
              SameDiff.SameDiffConditional predicate,
              SameDiff.SameDiffFunctionDefinition trueBody,
              SameDiff.SameDiffFunctionDefinition falseBody) {
        this.blockName = blockName;
        this.inputVars = inputVars;
        String falseBodyName = "false-body-" + UUID.randomUUID().toString();
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;
        this.falseBodyName = falseBodyName;
        parent.defineFunction(falseBodyName,falseBody);
        parent.defineFunction(trueBodyName,trueBody);
        //predicate execution reference storage
        SameDiff sameDiff = SameDiff.create();
        this.predicateBody = sameDiff;
        this.targetBoolean = predicate.eval(sameDiff,conditionBody,inputVars);
        //store a reference to both the true and false block
        this.trueBlockExecution = parent.getFunction(trueBodyName);
        this.falseBlockExecution = parent.getFunction(falseBodyName);
        addEdges(parent,this,opName());

    }


    @Override
    public NDArrayVertex getVertex() {
        return vertex;
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
