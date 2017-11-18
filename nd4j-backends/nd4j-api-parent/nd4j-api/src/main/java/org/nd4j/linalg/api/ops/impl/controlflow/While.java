package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.NodeDef;

import java.util.*;

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
@NoArgsConstructor
public class While extends DifferentialFunction implements CustomOp {


    @Getter
    protected SameDiff loopBodyExecution,predicateExecution;


    @Getter
    protected SameDiff.SameDiffConditional predicate;
    @Getter
    protected SameDiff.SameDiffFunctionDefinition trueBody;

    @Getter
    protected String blockName,trueBodyName;

    @Getter
    protected SDVariable[] inputVars;


    @Getter
    protected SDVariable targetBoolean;

    protected SDVariable dummyResult;

    @Getter
    @Setter
    protected SDVariable[] outputVars;

    @Getter
    protected int numLooped = 0;

    public While(While whileStatement) {
        this.sameDiff = whileStatement.sameDiff;
        this.outputVars = whileStatement.outputVars;
        this.loopBodyExecution = whileStatement.loopBodyExecution;
        this.numLooped = whileStatement.numLooped;
        this.args = whileStatement.args;
        this.dummyResult = whileStatement.dummyResult;
        this.predicate = whileStatement.predicate;
        this.predicateExecution = whileStatement.predicateExecution;
        this.shape = new int[] {1,1};
        addAsNewVertexId();
        f().addFunctionEdges(this);
        this.inputVars = whileStatement.inputVars;
        this.dummyResult =  this.sameDiff.var("dummyresult-" + UUID.randomUUID().toString(),new int[]{1,1},new ZeroInitScheme('f'),vertexId,0);
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


        OpState opState = OpState.builder()
                .opName(opName())
                .opType(opType())
                .inPlace(false)
                .id(UUID.randomUUID().toString())
                .vertexIds(opEdgeIds)
                .build();

        this.sameDiff.graph().addEdge(inputEdges,vertexId,opState,true);


    }



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
        //need to add the op to the list of ops to be executed when running backwards
        this.args = new DifferentialFunction[] {this};
        int[] vertexId = {parent.graph().nextVertexId()};

        this.dummyResult =  parent.var("dummyresult-" + UUID.randomUUID().toString(),new int[]{1,1},new ZeroInitScheme('f'),vertexId);
        this.vertexId = vertexId;
        parent.putFunction(vertexId,this);
        int[] inputEdges = new int[inputVars.length];
        String[] opEdgeIds = new String[inputVars.length];
        for(int i = 0; i < inputVars.length; i++) {
            inputVars[i] = parent.var(inputVars[i]);
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
        this.targetBoolean = predicate.eval(sameDiff,condition, inputVars);
        this.predicateExecution = sameDiff;
        //store references to the loop body
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;
        //running define function will setup a proper same diff instance
        parent.defineFunction(trueBodyName,trueBody,inputVars);
        parent.defineFunction(blockName,condition,inputVars);
        parent.putSubFunction("predicate-eval-body",sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);

        OpState opState = OpState.builder()
                .opName(opName())
                .opType(Op.Type.LOOP)
                .inPlace(false)
                .id(UUID.randomUUID().toString())
                .vertexIds(opEdgeIds)
                .build();

        parent.graph().addEdge(inputEdges,vertexId,opState,true);

    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        ret.add(new WhileDerivative(this));
        return ret;
    }



    /**
     * Increments the loop counter.
     * This should be called when the loop
     * actually executes.
     */
    public void incrementLoopCounter() {
        numLooped++;
    }



    @Override
    public int[] getResultShape() {
        return dummyResult.getShape();
    }

    @Override
    public SDVariable getResult() {
        return dummyResult;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith) {

    }

    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        return null;
    }

    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
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
        return opName().hashCode();
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


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "while_loop";
    }
}
