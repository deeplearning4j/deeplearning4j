package org.nd4j.linalg.api.ops;

import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Basic implementation for CustomOp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DynamicCustomOp extends DifferentialFunction implements CustomOp {

    private String opName;
    @Getter @Builder.Default private List<INDArray> inputArguments = new ArrayList<>();
    @Getter @Builder.Default private List<INDArray> outputArguments = new ArrayList<>();
    @Getter @Builder.Default private List<Double> tArguments = new ArrayList<>();
    @Getter @Builder.Default  private List<Integer> iArguments = new ArrayList<>();
    @Getter private boolean inplaceCall;
    @Getter private long hash;
    @Getter
    private NDArrayVertex[] outputs;
    @Getter
    protected DifferentialFunction[] outputFunctions;
    private List<int[]> outputShapes;

    public DynamicCustomOp() {
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }

    public DynamicCustomOp(String opName, SameDiff sameDiff, DifferentialFunction[] args) {
        super(sameDiff, args);
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
        addEdges(sameDiff,opName(), Op.Type.CUSTOM,extraArgs);
    }


    /**
     * Initialize this custom op with all of the
     * inputs, outputs, and respective
     * argumentts for execution
     * @param opName the name of the op to execute
     * @param inputs the inputs to the op
     * @param outputs the outputs of the op
     * @param tArguments the input float arguments
     * @param iArguments the input int arguments
     */
    public DynamicCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        inputArguments = new ArrayList<>(Arrays.asList(inputs));
        outputArguments = new ArrayList<>(Arrays.asList(outputs));
        this.opName = opName;
        this.tArguments = tArguments;
        this.iArguments = iArguments;
    }


    /**
     * Initialize this operation for execution (pre created ndarrays)
     * @param opName the operation name to use
     *               for invocation
     * @param inputs the inputs
     * @param outputs the outputs of the op
     */
    public DynamicCustomOp(String opName,INDArray[] inputs,INDArray[] outputs) {
        this(opName,inputs,outputs, Lists.<Double>newArrayList(), Lists.<Integer>newArrayList());
    }

    /**
     * Initialize this for {@link SameDiff} execution
     * Any extra int or float arguments for operations
     * must be added to the respective {@link #getTArguments()}
     *  or {@link #getIArguments()} lists upon construction
     * @param opName the operation name
     * @param sameDiff the samediff instance to use
     * @param args the arguments to use
     * @param inPlace whether the operation is in place or not
     *
     */
    public DynamicCustomOp(String opName,SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(sameDiff, inPlace, args);
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }

    protected DynamicCustomOp(String opName) {
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }



    /**
     * This method returns op name as string
     *
     * @return
     */
    @Override
    public String opName() {
        return opName;
    }

    @Override
    public int[] getVertexId() {
        return new int[] {getVertex().vertexID()};
    }

    @Override
    public NDArrayVertex[] getVertices() {
        return this.outputs;
    }

    @Override
    public NDArrayVertex getVertex() {
        if(this.outputs.length == 1)
            return this.outputs[0];
        else
            throw new UnsupportedOperationException("This op has more than one output.");
    }

    @Override
    public DifferentialFunction[] outputFunctions() {
        return outputFunctions;
    }

    /**
     * This method returns LongHash of the opName()
     *
     * @return
     */
    @Override
    public long opHash() {
        if (hash == 0) {
            val map = Nd4j.getExecutioner().getCustomOperations();
            val lcName = opName().toLowerCase();
            val desc = map.get(lcName);

            hash = desc.getHash();
        }

        return hash;
    }

    /**
     * This method takes custom opname, and return Op DynamicCustomOpsBuilder instance
     * @param opName
     * @return
     */
    public static DynamicCustomOpsBuilder builder(String opName) {
        val map = Nd4j.getExecutioner().getCustomOperations();
        val lcName = opName.toLowerCase();
        val desc = map.get(lcName);

        if (desc == null)
            throw new ND4JIllegalStateException("Unknown operations requested: [" + opName + "]");

        return new DynamicCustomOpsBuilder(opName, desc.getHash(), desc.getNumInputs(), desc.getNumOutputs(), desc.isAllowsInplace(), desc.getNumTArgs(), desc.getNumIArgs());
    }

    @Override
    public List<int[]> calculateOutputShape() {
        if(outputShapes != null)
            return outputShapes;
        return Nd4j.getExecutioner().calculateOutputShape(this);
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Please extend DynamicCustomOp to run samediff graph operations.");
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public int depth() {
        int maxDepth = 0;
        for(DifferentialFunction func : args()) {
            maxDepth = Math.max(maxDepth,func.depth());
        }

        return maxDepth;
    }

    @Override
    public List<DifferentialFunction> outputs() {
        return Arrays.asList(outputFunctions);
    }

    protected void addEdges(SameDiff sameDiff,
                            String opName,
                            Op.Type opType,
                            Object[] extraArgs) {
        for (DifferentialFunction input : args()) {
            validateFunctionReference(input);
            validateDifferentialFunctionGraph(input);
        }


        List<int[]> outputShapes = this.calculateOutputShape();
        int[] outputVertexIds = new int[outputShapes.size()];
        List<Integer> inputs = new ArrayList<>();
        for (int i = 0; i < args().length; i++) {
            DifferentialFunction differentialFunction = args()[i];
            List<DifferentialFunction> outputs = differentialFunction.outputs();
            for (DifferentialFunction output : outputs) {
                for (int vertexId : output.getOutputVertexIds()) {
                    if (!inputs.contains(vertexId))
                        inputs.add(vertexId);
                }
            }

        }

        this.outputs = new NDArrayVertex[outputShapes.size()];
        this.outputFunctions = new DifferentialFunction[outputShapes.size()];
        SDVariable[] resultInfo = new SDVariable[outputShapes.size()];
        for (int i = 0; i < outputShapes.size(); i++) {
            int nextVertexId = sameDiff.graph().nextVertexId();
            SDVariable variable = sameDiff.setupFunction(SDVariable.builder()
                    .varName(opName + "-UUID.randomUUID().toString()")
                    .shape(outputShapes.get(i))
                    .vertexId(new int[]{nextVertexId})
                    .varName(sameDiff.generateVariableName(opName, false))
                    .build());

            outputVertexIds[i] = variable.getVertex().vertexID();
            resultInfo[i] = variable;
            this.outputs[i] = variable.getVertex();
            this.outputFunctions[i] = variable;
        }

        int[] inputIds = Ints.toArray(inputs);


        String[] vertexIds = sameDiff.generateVertexIds(Ints.concat(inputIds, outputVertexIds));
        OpState opState = OpState.builder()
                .opType(opType).inPlace(inPlace)
                .differentialFunction(this)
                .opName(opName)
                .id(opName + "(" + vertexIds + ")")
                .vertexIds(sameDiff.generateVertexIds(Ints.concat(inputIds, outputVertexIds)))
                .extraArgs(extraArgs)
                .results(resultInfo)
                .build();


        /**
         * Create 1 opstate with all of the vertex ids
         * with all inputs and outputs representing the edge.
         */
        sameDiff.graph().addEdge(
                inputIds,
                outputVertexIds,
                opState, true);


        this.opState = opState;


    }


    public static class SameDiffBuilder extends DynamicCustomOpsBuilder {
        private SameDiff sameDiff;
        private List<DifferentialFunction> args = new ArrayList<>();

        private SameDiffBuilder(String opName,SameDiff sameDiff) {
            this(opName,sameDiff,0,0,0,false,0,0);
        }

        protected SameDiffBuilder(String opName, SameDiff sameDiff,long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
            super(opName, hash, numInputs, numOutputs, inplaceAllowed, numTArguments, numIArguments);
            this.sameDiff = sameDiff;
        }

        public SameDiffBuilder sameDiff(SameDiff sameDiff) {
            this.sameDiff = sameDiff;
            return this;
        }

        @Override
        public DynamicCustomOpsBuilder addInputs(INDArray... inputs) {
            throw new UnsupportedOperationException("Unable to add direct ndarrays. Please use the normal builder for that.");
        }

        @Override
        public DynamicCustomOpsBuilder addOutputs(INDArray... outputs) {
            throw new UnsupportedOperationException("Unable to add direct ndarrays. Please use the normal builder for that.");

        }


        public DynamicCustomOpsBuilder addInputs(DifferentialFunction... inputs) {
            for(DifferentialFunction  function : inputs) {
                args.add(function);
            }

            return this;
        }

        public DynamicCustomOpsBuilder addOutputs(DifferentialFunction... outputs) {
            throw new UnsupportedOperationException("Unable to add direct ndarrays. Please use the normal builder for that.");

        }


        @Override
        public DynamicCustomOp build() {
            DynamicCustomOp ret =  super.build();
            ret.setArgs(args.toArray(new DifferentialFunction[args.size()]));
            ret.setSameDiff(sameDiff);
            ret.outputShapes = outputShapes;
            ret.addEdges(sameDiff,opName, Op.Type.CUSTOM,null);
            return ret;
        }
    }


    public static SameDiffBuilder sameDiffBuilder(String opName,SameDiff sameDiff) {
        return new SameDiffBuilder(opName,sameDiff);
    }

    public static class DynamicCustomOpsBuilder {
        protected String opName;
        protected int numInputs;
        protected int numOutputs;
        protected int numTArguments;
        protected int numIArguments;
        protected boolean inplaceCall;
        protected boolean inplaceAllowed;
        protected long opHash;
        protected List<int[]> outputShapes = new ArrayList<>();

        private List<INDArray> inputArguments = new ArrayList<>();
        private List<INDArray> outputArguments = new ArrayList<>();
        private List<Double> tArguments = new ArrayList<>();
        private List<Integer> iArguments = new ArrayList<>();

        protected DynamicCustomOpsBuilder(String opName, long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
            this.opHash = hash;
            this.opName = opName;
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.numIArguments = numIArguments;
            this.numTArguments = numTArguments;
            this.inplaceAllowed = inplaceAllowed;
        }

        /**
         * This method
         * takes arbitrary number of input INDArrays in, as Op input
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param inputs
         * @return
         */
        public DynamicCustomOpsBuilder addInputs(INDArray... inputs) {
            // if we have positive value as numInputs - we should ensure equal amount of arguments
            if (numInputs >= 0) {
                if (inputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numInputs + " arguments. Null was passed instead.");

                if (numInputs > inputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numInputs + " arguments, but " + inputs.length + " was passed to constructor");
            }

            for (val in: inputs)
                inputArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of
         * output INDArrays in, to store operation result
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param outputs
         * @return
         */
        public DynamicCustomOpsBuilder addOutputs(INDArray... outputs) {
            if (numOutputs >= 0) {
                if (outputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numOutputs + " arguments. Null was passed instead.");

                if (numOutputs > outputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numOutputs + " arguments, but " + outputs.length + " was passed to constructor");
            }

            for (val in: outputs)
                outputArguments.add(in);

            return this;
        }

        /**
         * Whether an op call is in place or not.
         * @param reallyCall
         * @return
         */
        public DynamicCustomOpsBuilder callInplace(boolean reallyCall) {
            if (reallyCall && !inplaceAllowed)
                throw new ND4JIllegalStateException("Requested op can't be called inplace");

            this.inplaceCall = reallyCall;
            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(Integer... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments > iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param arg
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(int arg) {
            if (numIArguments != 1 && numIArguments > 0)
                throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. One arg was passed instead.");

            iArguments.add(arg);

            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(int... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments > iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of Double arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @return
         */
        public DynamicCustomOpsBuilder addFloatingPointArguments(Double... targs) {
            if (numTArguments >= 0) {
                if (targs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numTArguments + " integer arguments. Null was passed instead.");

                if (numTArguments > targs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numTArguments + " integer arguments, but " + targs.length + " was passed to constructor");
            }

            for (val in: targs)
                tArguments.add(in);

            return this;
        }


        /**
         * Adds an oup
         * @param shape
         * @return
         */
        public DynamicCustomOpsBuilder addOutputShape(int[] shape) {
            this.outputShapes.add(shape);
            return this;
        }




        public DynamicCustomOp build() {
            // Eventually we probably will lift this restriction
            //if (!inplaceCall && outputArguments.size() == 0)
            //    throw new ND4JIllegalStateException("If operation is not-inplace, it must have outputs defined");

            val result = new DynamicCustomOp(opName);
            result.inputArguments = inputArguments;
            result.outputArguments = outputArguments;
            result.iArguments = iArguments;
            result.tArguments = tArguments;
            result.inplaceCall = inplaceCall;
            result.hash = opHash;
            result.outputShapes = outputShapes;
            return result;
        }
    }
}
