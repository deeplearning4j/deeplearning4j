package org.nd4j.linalg.api.ops;

import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.FunctionProperties;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Basic implementation for CustomOp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DynamicCustomOp extends DifferentialFunction implements CustomOp {

    private String opName;
    @Builder.Default
    protected List<INDArray> inputArguments = new ArrayList<>();
    @Builder.Default
    protected List<INDArray> outputArguments = new ArrayList<>();


    @Builder.Default
    protected List<Double> tArguments = new ArrayList<>();
    @Builder.Default
    protected List<Long> iArguments = new ArrayList<>();
    @Getter
    @Setter
    protected boolean inplaceCall;
    @Getter
    private long hash;
    protected SDVariable[] outputVariables;
    private List<long[]> outputShapes;

    public DynamicCustomOp() {
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }


    // Hack to get output shape without data
    private Map<long[], INDArray> dummyArrays = new HashMap<>();

    private INDArray getDummyArray(long[] shape){
        INDArray dummy = dummyArrays.get(shape);
        if (dummy == null){
            dummy = Nd4j.zeros(shape);
            dummyArrays.put(shape, dummy);
        }
        return dummy;
    }

    private boolean dummies_set = false;
    private void setDummyInputs(){
        if(dummies_set){
            return;
        }
        dummies_set = true;
        val descriptor = getDescriptor();


        // this condition detects calls where
        // parameters are passed as inputs rather
        // iargs/targs
        if (args.length == 1 ||
            iArguments.size() > 1 ||
            tArguments.size() > 1 ||
            (descriptor.getNumIArgs() == 0 &&
             descriptor.getNumTArgs() == 0)){


            inputArguments.clear();// just in case
            for(val arg: args){
                val shape = arg.getShape();
                if (shape == null){
                    inputArguments.clear();
                    break;
                }
                inputArguments.add(getDummyArray(shape));
            }
        }
    }
    private SDVariable[] args;
    public DynamicCustomOp(String opName, SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
        this.args = args;
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();

    }


    /**
     * Initialize this custom op with all of the
     * inputs, outputs, and respective
     * arguments for execution
     *
     * @param opName     the opName of the op to execute
     * @param inputs     the inputs to the op
     * @param outputs    the outputs of the op
     * @param tArguments the input float arguments
     * @param iArguments the input int arguments
     */
    public DynamicCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        if (inputs != null)
            inputArguments = new ArrayList<>(Arrays.asList(inputs));
        if (outputs != null)
            outputArguments = new ArrayList<>(Arrays.asList(outputs));
        this.opName = opName;
        this.tArguments = tArguments;
        this.iArguments = new ArrayList<>();

        for (val a: iArguments)
            this.iArguments.add((Long) a.longValue());
    }


    /**
     * Initialize this operation for execution (pre created ndarrays)
     *
     * @param opName  the operation opName to use
     *                for invocation
     * @param inputs  the inputs
     * @param outputs the outputs of the op
     */
    public DynamicCustomOp(String opName, INDArray[] inputs, INDArray[] outputs) {
        this(opName, inputs, outputs, Lists.<Double>newArrayList(), Lists.<Integer>newArrayList());
    }

    /**
     * Initialize this for {@link SameDiff} execution
     * Any extra int or float arguments for operations
     * must be added to the respective TArguments
     * or IArguments lists upon construction
     *
     * @param opName   the operation opName
     * @param sameDiff the samediff instance to use
     * @param args     the arguments to use
     * @param inPlace  whether the operation is in place or not
     */
    public DynamicCustomOp(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, inPlace, args);
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
        this.inplaceCall = inPlace;
    }

    protected DynamicCustomOp(String opName) {
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }


    /**
     * This method returns op opName as string
     *
     * @return
     */
    @Override
    public String opName() {
        return opName;
    }


    @Override
    public SDVariable[] outputVariables() {
        return outputVariables(getOwnName() != null ? getOwnName() : opName());
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        if (this.outputVariables == null) {
            val outputNames = sameDiff.getOutputsForFunction(this);
            //no need to dynamically create if already exists
            if (outputNames != null) {
                outputVariables = new SDVariable[outputNames.length];
                for (int i = 0; i < outputVariables.length; i++) {
                    outputVariables[i] = sameDiff.getVariable(outputNames[i]);
                }

                return outputVariables;
            }


            val newVars = sameDiff.generateOutputVariableForOp(this, baseName); //Also adds outgoing
            if (isInplaceCall()) {
                if (args().length >= 1) {
                    val arr = args()[0].getArr();
                    if (arr != null) {
                        if (sameDiff.getArrForVarName(newVars[0].getVarName()) == null)
                            sameDiff.putArrayForVarName(newVars[0].getVarName(), arr);
                        else
                            sameDiff.updateArrayForVarName(newVars[0].getVarName(), arr);
                        addOutputArgument(arr);
                    }
                }

                return newVars;
            }


            val outputShapes = calculateOutputShape();
            if (newVars != null && outputShapes != null && !outputShapes.isEmpty()) {
                for (int i = 0; i < newVars.length; i++) {
                    if (newVars[i] == null)
                        continue;
                    attemptToGetOrCreateArrForVar(newVars[i], outputShapes.get(i));
                }
            } else if (getDescriptor().getNumOutputs() < 1  && getNumOutputs() < 1) {
                //this should only happen if we have no way of knowing how many
                //outputs are known from the descriptor
                return new SDVariable[0];
            } else {
                outputVariables = newVars;
                return newVars;
            }

            outputVariables = newVars;
            if (sameDiff.getOutputsForFunction(this) == null)
                sameDiff.addOutgoingFor(outputVariables, this);
            return newVars;
        } else {
            val shape = calculateOutputShape();
            if (shape != null && !shape.isEmpty()) {
                for (int i = 0; i < outputVariables.length; i++) {
                    val var = outputVariables[i];
                    if (var.getShape() == null) {
                        attemptToGetOrCreateArrForVar(var, shape.get(i));
                    }
                }
            }

        }

        return outputVariables;
    }


    private INDArray attemptToGetOrCreateArrForVar(SDVariable var, long[] currShape) {
        INDArray arr = null;
        if (Shape.isPlaceholderShape(var.getShape())) {
            if (var.getShape() == null) {
                val shape = calculateOutputShape();

                if (!shape.isEmpty()) {
                    if (currShape != null && !Shape.isPlaceholderShape(currShape)) {
                        sameDiff.putShapeForVarName(var.getVarName(), currShape);
                        arr = var.storeAndAllocateNewArray();
                    }

                } else
                    arr = null;
            }

        } else if (sameDiff.getArrForVarName(var.getVarName()) == null) {
            if (var.getShape() != null)
                arr = var.storeAndAllocateNewArray();
        } else {
            arr = var.getArr();
        }

        if (arr != null) {
            sameDiff.associateArrayWithVariable(arr, var);
            addOutputArgument(arr);
        }


        return arr;
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
            val desc = map.get(opName());
            if (desc == null) {
                throw new ND4JIllegalStateException("Op name " + opName() + " is missing!");
            }

            hash = desc.getHash();
        }

        return hash;
    }

    @Override
    public INDArray[] outputArguments() {
        if (!outputArguments.isEmpty()) {
            return outputArguments.toArray(new INDArray[outputArguments.size()]);
        }
        return new INDArray[0];
    }

    @Override
    public INDArray[] inputArguments() {
        if (!inputArguments.isEmpty())
            return inputArguments.toArray(new INDArray[inputArguments.size()]);
        return new INDArray[0];

    }

    @Override
    public long[] iArgs() {
        return Longs.toArray(iArguments);
    }

    @Override
    public double[] tArgs() {
        return Doubles.toArray(tArguments);
    }

    @Override
    public void addIArgument(int... arg) {
        for (long a: arg)
            iArguments.add(a);
    }

    @Override
    public void addIArgument(long... arg) {
        for (long a: arg)
            iArguments.add(a);
    }

    private void addIArgument(Integer... arg) {
        for (val a: arg)
            addIArgument((Long) a.longValue());
    }

    @Override
    public void removeIArgument(Integer arg) {
        iArguments.remove(arg);
    }

    @Override
    public Long getIArgument(int index) {
        return iArguments.get(index);
    }

    @Override
    public int numIArguments() {
        return iArguments.size();
    }

    @Override
    public void addTArgument(double... arg) {
        if (arg != null)
            addTArgument(Doubles.asList(arg).toArray(new Double[arg.length]));

    }

    private void addTArgument(Double... arg) {
        tArguments.addAll(Arrays.asList(arg));
    }

    @Override
    public void removeTArgument(Double arg) {
        tArguments.remove(arg);
    }

    @Override
    public Double getTArgument(int index) {
        return tArguments.get(index);
    }

    @Override
    public int numTArguments() {
        return tArguments.size();
    }

    @Override
    public void addInputArgument(INDArray... arg) {
        for (int i = 0; i < arg.length; i++) {
            if (arg[i] == null)
                throw new ND4JIllegalStateException("Input " + i + " was null!");
        }


        inputArguments.addAll(Arrays.asList(arg));

        val args = sameDiff != null ? args() : null;
        val arrsSoFar = inputArguments();
        //validate arrays passed in, keep in mind that
        //this is a cumulative algorithm so we should always
        //refresh the current list
        if (args != null) {
            for (int i = 0; i < args.length; i++) {

                // it's possible to get into situation where number of args > number of arrays AT THIS MOMENT
                if (i >= arrsSoFar.length)
                    continue;

                if (!Arrays.equals(args[i].getShape(), arrsSoFar[i].shape()))
                    throw new ND4JIllegalStateException("Illegal array passed in. Expected shape " + Arrays.toString(args[i].getShape()) + " and received array with shape " + Arrays.toString(arg[i].shape()));
            }
        }
    }

    @Override
    public void removeInputArgument(INDArray arg) {
        inputArguments.remove(arg);
    }

    @Override
    public INDArray getInputArgument(int index) {
        return inputArguments.get(index);
    }

    public void setInputArgument(int index, INDArray input) {
        inputArguments.set(index, input);
    }

    public void setOutputArgument(int index, INDArray output) {
        outputArguments.set(index, output);
    }

    @Override
    public int numInputArguments() {
        return inputArguments.size();
    }

    @Override
    public void addOutputArgument(INDArray... arg) {
        for (int i = 0; i < arg.length; i++) {
            if (arg[i] == null)
                throw new ND4JIllegalStateException("Output " + i + " was null!");
        }
        outputArguments.addAll(Arrays.asList(arg));
    }

    @Override
    public void removeOutputArgument(INDArray arg) {
        outputArguments.remove(arg);
    }

    @Override
    public INDArray getOutputArgument(int index) {
        return outputArguments.get(index);
    }

    @Override
    public int numOutputArguments() {
        return outputArguments.size();
    }


    @Override
    public int opNum() {
        return (int) opHash();
    }

    /**
     * This method takes custom opname, and return Op DynamicCustomOpsBuilder instance
     *
     * @param opName
     * @return
     */
    public static DynamicCustomOpsBuilder builder(String opName) {
        val map = Nd4j.getExecutioner().getCustomOperations();
        val lcName = map.containsKey(opName) ? opName : opName.toLowerCase();
        val desc = map.get(lcName);

        if (desc == null)
            throw new ND4JIllegalStateException("Unknown operations requested: [" + opName + "]");

        return new DynamicCustomOpsBuilder(lcName, desc.getHash(), desc.getNumInputs(), desc.getNumOutputs(), desc.isAllowsInplace(), desc.getNumTArgs(), desc.getNumIArgs());
    }

    @Override
    public List<long[]> calculateOutputShape() {
        val descriptor = getDescriptor();
        for (val arg : args()) {
            if (sameDiff.isPlaceHolder(arg.getVarName()) && !sameDiff.shapeAlreadyExistsForVarName(arg.getVarName())) {
                if(log.isTraceEnabled()){
                    log.trace("Could not calculate output shape for op {}: arg \"{}\" is placeholder", getClass().getName(),
                            arg.getVarName());
                }
                return Collections.emptyList();
            }
        }

        if (outputShapes != null)
            return outputShapes;


        //not fully initialized: missing integer args
        if (descriptor.getNumIArgs() >= 0 && numIArguments() < descriptor.getNumIArgs()) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} IArgs specified, " +
                                "{} required)", getClass().getName(),numIArguments(), descriptor.getNumIArgs());
            }
            return Collections.emptyList();
        }


        //not fully initialized: missing floating point args
        if (descriptor.getNumTArgs() >= 0 && numTArguments() < descriptor.getNumTArgs()) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} TArgs specified, " +
                        "{} required)", getClass().getName(),numTArguments(), descriptor.getNumTArgs());
            }
            return Collections.emptyList();
        }

        //not fully initialized: missing INDArray input args
        if(descriptor.getNumInputs() >= 0 && numInputArguments() < descriptor.getNumInputs()){
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} input (INDArray) args specified, " +
                        "{} required)", getClass().getName(),numInputArguments(), descriptor.getNumInputs());
            }
            return Collections.emptyList();
        }


        /**
         * Note that we are currently getting shape errors
         * because the input and output arguments are not specified.
         */

        setDummyInputs();
        try {
            return Nd4j.getExecutioner().calculateOutputShape(this);
        }
       catch (Exception e){
           return Collections.emptyList();
       }
    }

    @Override
    public CustomOpDescriptor getDescriptor() {
        val map = Nd4j.getExecutioner().getCustomOperations();
        return map.get(opName());
    }

    @Override
    public void assertValidForExecution() {
        val descriptor = getDescriptor();
        if (descriptor == null)
            throw new NoOpNameFoundException("No descriptor found for op name " + opName());


        if (descriptor.getNumInputs() > 0 && numInputArguments() < descriptor.getNumInputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + ": Number of inputs is invalid for execution. Specified " + numInputArguments() + " but should be " + descriptor.getNumInputs());

        if (descriptor.getNumOutputs() > 0 && numOutputArguments() < descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + ": Number of outputs is invalid for execution. Specified " + numOutputArguments() + " but should be " + descriptor.getNumOutputs());

        //< 0 means dynamic size
        if (descriptor.getNumIArgs() >= 0 && numIArguments() < descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + ": Number of integer arguments is invalid for execution. Specified " + numIArguments() + " but should be " + descriptor.getNumIArgs());

        if (descriptor.getNumTArgs() >= 0 && numTArguments() < descriptor.getNumTArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + ": Number of inputs is invalid for execution. Specified " + numTArguments() + " but should be " + descriptor.getNumTArgs());

    }

    public void updateInputsFromSameDiff() {
        val inputs = sameDiff.getInputsForFunction(this);

        inputArguments.clear();

        for (val input: inputs)
            inputArguments.add(sameDiff.getArrForVarName(input));
    }

    @Override
    public void populateInputsAndOutputsFromSameDiff() {
        val descriptor = getDescriptor();
        if (descriptor == null)
            throw new ND4JIllegalStateException("No op found for " + opName());

        log.debug("Op <{}>, isInplace: {}", opName(), isInplaceCall());


        if (isInplaceCall()) {
            if (numInputArguments() != descriptor.getNumInputs()) {
                //clear just in case
                val args = args();
                inputArguments.clear();
                for (val arg : args) {
                    inputArguments.add(arg.getArr());
                }
            }

            if (numOutputArguments() != descriptor.getNumOutputs()) {
                //clear just in case
                val arg = args()[0];
                outputArguments.clear();
                outputArguments.add(arg.getArr());

            }
        } else {
            if (numInputArguments() != descriptor.getNumInputs()) {
                //clear just in case
                val args = args();
                inputArguments.clear();
                boolean nullArr = false;
                for (val arg : args()) {
                    //we should not attempt to resolve
                    //outputs when null inputs exist
                    if (arg.getArr() == null) {
                        nullArr = true;
                        log.warn("No input found for " + arg.getVarName() + " and op name " + opName());
                        continue;
                    }
                }


                if (!nullArr) {
                    for (val arg : args) {
                        inputArguments.add(arg.getArr());
                    }
                }
            }

            val numOutputs = numOutputArguments();
            val dO = descriptor.getNumOutputs();

            if (numOutputArguments() != descriptor.getNumOutputs()) {
                //clear just in case
                val outputVars = outputVariables();
                outputArguments.clear();
                for (val arg : outputVars) {
                    INDArray arr = arg.getArr();
                    if (arr == null) {
                        outputArguments.clear();
                        //placeholder case,  will try to resolve shapes one more time
                        val shapes = calculateOutputShape();
                        for (int i = 0; i < shapes.size(); i++) {
                            if (outputVars[i].getArr() != null)
                                outputArguments.add(outputVars[i].getArr());
                            else if (outputVars[i].getShape() == null) {
                                sameDiff.putShapeForVarName(outputVars[i].getVarName(), shapes.get(i));
                                outputArguments.add(outputVars[i].storeAndAllocateNewArray());
                            } else
                                throw new ND4JIllegalStateException("Unable to resolve for variable " + outputVars[i].getVarName());
                        }
                    } else
                        outputArguments.add(arr);
                }
            }
        }

    }


    /**
     * Return function properties for the given function
     *
     * @return
     */
    public FunctionProperties asProperties() {
        return FunctionProperties.builder()
                .name(opName())
                .l(iArguments)
                .d(tArguments)
                .fieldNames(propertiesForFunction())
                .build();
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Please extend DynamicCustomOp.doDiff to support SameDiff backprop " +
                "operations. Op: " + getClass().getName());
    }

    @Override
    public String toString() {
        return opName();
    }


    public static class SameDiffBuilder extends DynamicCustomOpsBuilder {
        private SameDiff sameDiff;
        private List<DifferentialFunction> args = new ArrayList<>();
        private List<DifferentialFunction> outputs = new ArrayList<>();

        private SameDiffBuilder(String opName, SameDiff sameDiff) {
            this(opName, sameDiff, 0, 0, 0, false, 0, 0);
        }

        protected SameDiffBuilder(String opName, SameDiff sameDiff, long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
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
            for (DifferentialFunction function : inputs) {
                args.add(function);
            }

            return this;
        }

        public DynamicCustomOpsBuilder addOutputs(DifferentialFunction... outputs) {
            this.outputs.addAll(Arrays.asList(outputs));
            return this;

        }


        @Override
        public DynamicCustomOp build() {
            DynamicCustomOp ret = super.build();
            ret.setSameDiff(sameDiff);
            ret.outputShapes = outputShapes;
            sameDiff.putFunctionForId(ret.getOwnName(), ret);
            if (outputs.isEmpty() && !outputShapes.isEmpty()) {
                for (int i = 0; i < outputShapes.size(); i++) {
                    outputs.add(sameDiff.var(sameDiff.generateNewVarName("dynamiccustomop", i), outputShapes.get(i)));
                }

            }

            sameDiff.putFunctionForId(ret.getOwnName(), ret);
            ret.outputVariables = outputs.toArray(new SDVariable[outputs.size()]);

            return ret;
        }
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }


    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }


    public static SameDiffBuilder sameDiffBuilder(String opName, SameDiff sameDiff) {
        return new SameDiffBuilder(opName, sameDiff);
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
        protected List<long[]> outputShapes = new ArrayList<>();

        private List<INDArray> inputArguments = new ArrayList<>();
        private List<INDArray> outputArguments = new ArrayList<>();
        private List<Double> tArguments = new ArrayList<>();
        private List<Long> iArguments = new ArrayList<>();

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

            for (val in : inputs)
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

            for (val in : outputs)
                outputArguments.add(in);

            return this;
        }


        /**
         * Whether an op call is in place or not.
         *
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
        public DynamicCustomOpsBuilder addIntegerArguments(List<Integer> iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments > iargs.size())
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects at least " + numIArguments + " integer arguments, but "
                            + iargs.size() + " was passed to constructor");
            }

            for (val in : iargs)
                iArguments.add(in.longValue());

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
        public DynamicCustomOpsBuilder addIntegerArguments(long arg) {
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

            for (val in : iargs)
                iArguments.add((long) in);

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

            for (val in : targs)
                tArguments.add(in);

            return this;
        }


        /**
         * Adds an oup
         *
         * @param shape
         * @return
         */
        public DynamicCustomOpsBuilder addOutputShape(int[] shape) {
            this.outputShapes.add(ArrayUtil.toLongArray(shape));
            return this;
        }

        public DynamicCustomOpsBuilder addOutputShape(long[] shape) {
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

        public int getNumOutputs(){
            return -1;
        }
    }
}
