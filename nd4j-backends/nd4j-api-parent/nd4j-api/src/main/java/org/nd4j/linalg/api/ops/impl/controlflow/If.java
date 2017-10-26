package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.ArrayList;
import java.util.Collections;
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

    private NDArrayInformation dummyResult;

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
        this.dummyResult = NDArrayInformation.newInfo(new int[]{1,1});
        this.vertexId = new int[] {parent.graph().nextVertexId()};
        NDArrayVertex dummyVertex = new NDArrayVertex(parent,this.vertexId[0],0,dummyResult);
        parent.graph().addVertex(dummyVertex);
        this.vertex = dummyVertex;
        int[] inputEdges = new int[inputVars.length];
        int[] outputEdges = new int[inputVars.length];
        String[] opEdgeIds = new String[inputVars.length * 2];
        NDArrayInformation[] results = new NDArrayInformation[inputVars.length];
        for(int i = 0; i < inputVars.length; i++) {
            inputVars[i] = parent.setupFunction(inputVars[i]);
            NDArrayInformation outputInfo = NDArrayInformation.newInfo(
                    inputVars[i].getInfo().getShape()
                    ,inputVars[i].getInfo().getWeightInitScheme());
            NDArrayVertex ndArrayVertex = new NDArrayVertex(parent,parent.graph().nextVertexId(),inputVars[i].depth() + 1, outputInfo);
            inputEdges[i] = inputVars[i].getVertex().vertexID();
            outputEdges[i] = ndArrayVertex.vertexID();
            results[i] = outputInfo;
            parent.graph().addVertex(ndArrayVertex);
            parent.addVariable(
                    SDVariable.builder()
                            .shape(inputVars[i].getShape())
                            .varName(inputVars[i].getVarName() + "-output")
                            .sameDiff(parent)
                            .arr(inputVars[i].getArr())
                            .info(outputInfo)
                            .vertexId(new int[]{ndArrayVertex.vertexID()})
                            .ndArrayVertex(ndArrayVertex)
                            .build());
        }


        /**
         * Setup the opstate ids
         */
        int opEdgeIdIdx = 0;
        for(int i = 0; i < inputEdges.length; i++) {
            opEdgeIds[opEdgeIdIdx++] = String.valueOf(inputEdges[i]);
        }

        for(int i = 0; i < inputEdges.length; i++) {
            opEdgeIds[opEdgeIdIdx++] = String.valueOf(outputEdges[i]);
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
        parent.getSameDiffFunctionInstances().put("predicate-eval-body",sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);

        OpState opState = OpState.builder()
                .opName(opName())
                .opType(Op.Type.CONDITIONAL)
                .differentialFunction(this)
                .inPlace(false)
                .results(results)
                .id(UUID.randomUUID().toString())
                .vertexIds(opEdgeIds)
                .build();

        parent.graph().addEdge(inputEdges,outputEdges,opState,true);
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
    public NDArrayVertex getVertex() {
        return vertex;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        if(trueBodyExecuted != null) {

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
