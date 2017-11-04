package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.*;

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
    private SameDiff loopBodyExecution,predicateExecution,falseBodyExecution;


    @Getter
    private SameDiff.SameDiffConditional predicate;
    @Getter
    private SameDiff.SameDiffFunctionDefinition trueBody,falseBody;

    @Getter
    private String blockName,trueBodyName,falseBodyName;

    @Getter
    private SDVariable[] inputVars;


    private Boolean trueBodyExecuted = null;

    @Getter
    private SDVariable targetBoolean;

    private SDVariable dummyResult;

    @Getter
    @Setter
    private SDVariable[] outputVars;


    @Builder
    public If(String blockName,
              SameDiff parent,
              SDVariable[] inputVars,
              SameDiff.SameDiffFunctionDefinition conditionBody,
              SameDiff.SameDiffConditional predicate,
              SameDiff.SameDiffFunctionDefinition trueBody,
              SameDiff.SameDiffFunctionDefinition falseBody) {
        this.sameDiff = parent;
        this.inputVars = inputVars;
        this.predicate = predicate;
        this.trueBody = trueBody;
        this.falseBody = falseBody;
        this.blockName = blockName;
        int[] vertexId = {parent.graph().nextVertexId()};
        this.dummyResult =  parent.var("dummyresult-" + UUID.randomUUID().toString(),new int[]{1,1},new ZeroInitScheme('f'),vertexId,0);
        this.vertexId = vertexId;
        int[] inputEdges = new int[inputVars.length];
        String[] opEdgeIds = new String[inputVars.length * 2];

        for(int i = 0; i < inputEdges.length; i++) {
            inputEdges[i] = inputVars[i].getVertexId()[0];
        }

        /**
         * Setup the opstate ids
         */
        int opEdgeIdIdx = 0;
        for(int i = 0; i < inputEdges.length; i++) {
            opEdgeIds[opEdgeIdIdx++] = String.valueOf(inputEdges[i]);
        }


        //create a samediff sub graph for running just the execution
        //return a reference to the loop for referencing during actual execution
        SameDiff sameDiff = SameDiff.create();
        //store the reference to the result array and the same diff execution instance
        this.targetBoolean = predicate.eval(sameDiff,conditionBody, inputVars);
        this.predicateExecution = sameDiff;
        //store references to the loop body
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;

        String falseBodyName = "false-body-" + UUID.randomUUID().toString();
        this.falseBodyName = trueBodyName;

        //running define function will setup a proper same diff instance
        this.loopBodyExecution = parent.defineFunction(trueBodyName,trueBody,inputVars);
        this.falseBodyExecution = parent.defineFunction(falseBodyName,falseBody,inputVars);
        parent.defineFunction(blockName,conditionBody,inputVars);
        parent.putSubFunction("predicate-eval-body-" + UUID.randomUUID().toString(),sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);
        parent.putFunction(vertexId,this);

        OpState opState = OpState.builder()
                .opName(opName())
                .opType(Op.Type.CONDITIONAL)
                .inPlace(false)
                .id(UUID.randomUUID().toString())
                .vertexIds(opEdgeIds)
                .build();

        parent.graph().addEdge(inputEdges,vertexId,opState,true);
    }


    /**
     * Toggle whether the true body was executed
     * or the false body
     * @param trueBodyExecuted
     */
    public void exectedTrueOrFalse(boolean trueBodyExecuted)  {
        if(trueBodyExecuted)
            this.trueBodyExecuted = true;
        else
            this.trueBodyExecuted = false;
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        if(trueBodyExecuted != null) {
            if(trueBodyExecuted) {
                Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = loopBodyExecution.execBackwards();
                for(SDVariable variable : outputVars) {

                }

            }
            else {
                Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = falseBodyExecution.execBackwards();

            }
        }
        else {

        }
        return ret;
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
        return Collections.emptyList();
    }

    @Override
    public List<INDArray> getOutputArguments() {
        return Collections.emptyList();

    }

    @Override
    public List<Integer> getIArguments() {
        return Collections.emptyList();
    }

    @Override
    public List<Double> getTArguments() {
        return Collections.emptyList();
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return Collections.emptyList();
    }
}
