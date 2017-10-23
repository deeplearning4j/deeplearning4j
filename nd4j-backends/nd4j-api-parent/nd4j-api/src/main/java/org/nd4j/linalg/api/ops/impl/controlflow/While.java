package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
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
 * Equivalent to tensorflow's while loop
 * Takes in:
 * loopVars
 * loop body
 * condition
 *
 * runs loop till condition is false.
 * @author Adam Gibson
 */
public class While extends DifferentialFunction implements CustomOp {


    @Getter
    private SameDiff loopBodyExecution,predicateExecution;


    @Getter
    private SameDiff.SameDiffConditional predicate;
    @Getter
    private SameDiff.SameDiffFunctionDefinition trueBody;

    @Getter
    private String blockName,trueBodyName;

    @Getter
    private SDVariable[] inputVars;


    @Getter
    private SDVariable targetBoolean;

    private NDArrayInformation dummyResult;

    @Builder
    public While(String blockName,
                 SameDiff parent,
                 SDVariable[] inputVars,
                 SameDiff.SameDiffConditional predicate,
                 SameDiff.SameDiffFunctionDefinition condition,
                 SameDiff.SameDiffFunctionDefinition trueBody) {

        this.sameDiff = parent;
        this.inputVars = inputVars;
        this.predicate = predicate;
        this.trueBody = trueBody;
        this.blockName = blockName;
        this.dummyResult = NDArrayInformation.newInfo(new int[]{1,1});
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
        this.targetBoolean = predicate.eval(sameDiff,condition, inputVars);
        this.predicateExecution = sameDiff;
        //store references to the loop body
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;
        //running define function will setup a proper same diff instance
        parent.defineFunction(trueBodyName,trueBody,inputVars);
        parent.defineFunction(blockName,condition,inputVars);
        parent.getSameDiffFunctionInstances().put("predicate-eval-body",sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);

        OpState opState = OpState.builder()
                .opName(opName())
                .opType(Op.Type.LOOP)
                .differentialFunction(this)
                .inPlace(false)
                .results(results)
                .id(UUID.randomUUID().toString())
                .vertexIds(opEdgeIds)
                .build();

        parent.graph().addEdge(inputEdges,outputEdges,opState,true);


    }


    @Override
    public int[] getResultShape() {
        return dummyResult.getShape();
    }

    @Override
    public NDArrayInformation getResult() {
        return dummyResult;
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
        return "while";
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
        List<int[]> ret =  new ArrayList<>();
        for(DifferentialFunction var : args()) {
            ret.add(var.getShape());
        }
        return ret;
    }
}
