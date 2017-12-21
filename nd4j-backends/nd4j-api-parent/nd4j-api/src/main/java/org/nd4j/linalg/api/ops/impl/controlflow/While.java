package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

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
@Slf4j
public class While extends DifferentialFunction implements CustomOp {
    private AtomicInteger  startPosition;



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

    /**
     * Mainly meant for tensorflow import.
     * This allows {@link #initFromTensorFlow(NodeDef, SameDiff, Map, GraphDef)}
     * to continue from a parent while loop
     * using the same graph
     * @param startPosition the start position for the import scan
     */
    public While(AtomicInteger startPosition) {
        this.startPosition = startPosition;
    }

    public While(While whileStatement) {
        this.sameDiff = whileStatement.sameDiff;
        this.outputVars = whileStatement.outputVars;
        this.loopBodyExecution = whileStatement.loopBodyExecution;
        this.numLooped = whileStatement.numLooped;
        this.dummyResult = whileStatement.dummyResult;
        this.predicate = whileStatement.predicate;
        this.predicateExecution = whileStatement.predicateExecution;
        this.inputVars = whileStatement.inputVars;
        this.dummyResult =  this.sameDiff.var("dummyresult-" + UUID.randomUUID().toString(),new int[]{1,1},new ZeroInitScheme('f'));


    }



    @Builder
    public While(String blockName,
                 SameDiff parent,
                 SDVariable[] inputVars,
                 SameDiff.SameDiffConditional predicate,
                 SameDiff.SameDiffFunctionDefinition condition,
                 SameDiff.SameDiffFunctionDefinition trueBody) {
        init(blockName,parent,inputVars,predicate,condition,trueBody);
    }


    private void init(String blockName,
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
        this.dummyResult =  parent.var("dummyresult-" + UUID.randomUUID().toString(),new int[]{1,1},new ZeroInitScheme('f'));
        parent.putFunctionForId(getInstanceId(),this);

        parent.addArgsFor(inputVars,this);
        parent.addOutgoingFor(new SDVariable[]{dummyResult},this);


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

    }


    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[]{dummyResult};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        ret.addAll(Arrays.asList(new WhileDerivative(this).outputVariables()));
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
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //note that we initialize startPosition possibly from a parent context, the default should start from
        //
        // the passed in node, but we might be farther along in the import loop.

        val startPosition = this.startPosition == null ? new AtomicInteger(graph.getNodeList().indexOf(nodeDef)) : this.startPosition;
        this.startPosition = startPosition;
        this.sameDiff = initWith;
        val uniqueId = java.util.UUID.randomUUID().toString();
        this.blockName = uniqueId;


        // parsing declarations first. they all come as Expose ops
        int enterCnt = 0;

        Set<String> skipSet = new HashSet<>();
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            if (!tfNode.getOp().equalsIgnoreCase("enter")) {
                skipSet.add(tfNode.getName());
                break;
            }

            skipSet.add(tfNode.getName());

        }
        // now we're skipping Merge step, since we've already captured variables at Expose step
        int mergedCnt = 0;
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            if (!tfNode.getOp().equalsIgnoreCase("merge")) {
                break;
            }

            skipSet.add(tfNode.getName());

            mergedCnt++;
        }


        this.predicate = new SameDiff.DefaultSameDiffConditional();

        // now, we're adding conditional scope
        val conditional = SameDiff.create();
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            // we're parsing up to condition
            if (tfNode.getOp().equalsIgnoreCase("LoopCond")) {
                skipSet.add(tfNode.getName());
                startPosition.incrementAndGet();
                break;
            }

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = conditional.var(tfNode.getName(),TFGraphMapper.getInstance().getArrayFrom(tfNode,graph));
                log.info("Adding body var [{}:{}]", var.getVarName(), var.getVarName());

            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());
                val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());
                val varOutput = conditional.var(tfNode.getName(),sameDiff.getShapeForVarName(func.outputVariables()[0].getVarName()));
                func.initFromTensorFlow(tfNode,conditional,nodeDef.getAttrMap(),graph);
            }

            skipSet.add(tfNode.getName());
        }



        // time to skip some Switch calls
        int switchCnt = 0;
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            // we're parsing up to condition
            if (!tfNode.getOp().equalsIgnoreCase("Switch"))
                break;

            skipSet.add(tfNode.getName());

        }

        // now we're parsing Identity step
        int identityCnt = 0;
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());


            if (!tfNode.getOp().equalsIgnoreCase("Identity")) {
                break;
            }


            val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());
            val varOutput = conditional.var(tfNode.getName(),initWith.getShapeForVarName(func.outputVariables()[0].getVarName()));
            func.initFromTensorFlow(tfNode,initWith,nodeDef.getAttrMap(),graph);
            identityCnt++;


            skipSet.add(tfNode.getName());
        }


        // parsing body scope
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            if (skipSet.contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }

            if (tfNode.getOp().equalsIgnoreCase("NextIteration")) {
                skipSet.add(tfNode.getName());
                break;
            }

            if (skipSet.contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }



            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = initWith.var(tfNode.getName(),TFGraphMapper.getInstance().getArrayFrom(tfNode,graph));
                log.info("Adding body var [{}:{}]", var.getVarName(), var.getVarName());
            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                boolean isNewLoop = false;
                SameDiff potentialNewLoop = null;
                if (tfNode.getOp().equalsIgnoreCase("enter")) {
                    val frame_name = attributesForNode.get("frame_name").getS().toStringUtf8();
                    if (initWith.getFunction(frame_name) == null) {
                        potentialNewLoop = SameDiff.create();
                        initWith.putSubFunction(frame_name,potentialNewLoop);
                        isNewLoop = true;
                    }

                    this.loopBodyExecution = potentialNewLoop;

                }

                if (isNewLoop) {
                    log.info("NEW LOOP ----------------------------------------");
                    val func = new While(startPosition);
                    val varOutput = conditional.var(tfNode.getName(),initWith.getShapeForVarName(func.outputVariables()[0].getVarName()));
                    func.initFromTensorFlow(tfNode,initWith,nodeDef.getAttrMap(),graph);

                    log.info("END LOOP ----------------------------------------");
                } else {
                    val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());
                    val varOutput = conditional.var(tfNode.getName(),initWith.getShapeForVarName(func.outputVariables()[0].getVarName()));
                    func.initFromTensorFlow(tfNode,initWith,nodeDef.getAttrMap(),graph);
                }
            }

            skipSet.add(tfNode.getName());
        }


        // mapping NextIterations, to Return op
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            if (!tfNode.getOp().equalsIgnoreCase("NextIteration"))
                break;

            skipSet.add(tfNode.getName());


        }

        // we should also map While/Exit to libnd4j while
        int exitCnt = 0;
        for (; startPosition.get() < graph.getNodeCount(); startPosition.incrementAndGet()) {
            val tfNode = graph.getNode(startPosition.get());

            if (!tfNode.getOp().equalsIgnoreCase("Exit")) {
                skipSet.add(tfNode.getName());
                break;
            }

            skipSet.add(tfNode.getName());
        }


        log.info("-------------------------------------------");


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

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
    public INDArray[] outputArguments() {
        return new INDArray[0];
    }

    @Override
    public INDArray[] inputArguments() {
        return new INDArray[0];
    }

    @Override
    public int[] iArgs() {
        return new int[0];
    }

    @Override
    public double[] tArgs() {
        return new double[0];
    }

    @Override
    public void addIArgument(int... arg) {

    }

    @Override
    public void removeIArgument(Integer arg) {

    }

    @Override
    public Integer getIArgument(int index) {
        return null;
    }

    @Override
    public int numIArguments() {
        return 0;
    }

    @Override
    public void addTArgument(double... arg) {

    }

    @Override
    public void removeTArgument(Double arg) {

    }

    @Override
    public Double getTArgument(int index) {
        return null;
    }

    @Override
    public int numTArguments() {
        return 0;
    }

    @Override
    public void addInputArgument(INDArray... arg) {

    }

    @Override
    public void removeInputArgument(INDArray arg) {

    }

    @Override
    public INDArray getInputArgument(int index) {
        return null;
    }

    @Override
    public int numInputArguments() {
        return 0;
    }

    @Override
    public void addOutputArgument(INDArray... arg) {

    }

    @Override
    public void removeOutputArgument(INDArray arg) {

    }

    @Override
    public INDArray getOutputArgument(int index) {
        return null;
    }

    @Override
    public int numOutputArguments() {
        return 0;
    }
    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret =  new ArrayList<>();
        for(SDVariable var : args()) {
            ret.add(sameDiff.getShapeForVarName(var.getVarName()));
        }
        return ret;
    }

    @Override
    public CustomOpDescriptor getDescriptor() {
        return null;
    }

    @Override
    public void assertValidForExecution() {

    }

    @Override
    public void populateInputsAndOutputsFromSameDiff() {

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
