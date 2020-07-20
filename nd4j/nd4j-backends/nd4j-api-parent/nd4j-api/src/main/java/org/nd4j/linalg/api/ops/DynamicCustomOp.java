/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.guava.collect.Lists;
import org.nd4j.shade.guava.primitives.Doubles;
import org.nd4j.shade.guava.primitives.Longs;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Array;
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

    @Builder.Default
    protected List<Boolean> bArguments = new ArrayList<>();

    @Builder.Default
    protected List<DataType> dArguments = new ArrayList<>();

    @Builder.Default
    protected List<Integer> axis = new ArrayList<>();

    @Getter
    @Setter
    protected boolean inplaceCall;
    @Getter
    private long hash;
    protected SDVariable[] outputVariables;
    private List<LongShapeDescriptor> outputShapes;

    public DynamicCustomOp() {
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
        bArguments = new ArrayList<>();
        dArguments = new ArrayList<>();
    }

    public DynamicCustomOp(SameDiff sameDiff, SDVariable arg) {
        this(sameDiff, wrapOrNull(arg));
    }

    public DynamicCustomOp(SameDiff sameDiff, SDVariable[] args) {
        this(null, sameDiff, args);
    }

    public DynamicCustomOp(String opName, SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
        bArguments = new ArrayList<>();
        dArguments = new ArrayList<>();
    }

    public DynamicCustomOp(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments) {
        this(opName, (input == null ? null : new INDArray[]{input}), (output == null ? null : new INDArray[]{output}), tArguments, iArguments);
    }

    public DynamicCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments) {
        this(opName, inputs, outputs, tArguments, ArrayUtil.toList(iArguments));
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
        if(tArguments == null) {
            this.tArguments = new ArrayList<>();
        } else {
            this.tArguments = tArguments;
        }
        this.iArguments = new ArrayList<>();

        if(iArguments != null) {
            for (val a : iArguments)
                this.iArguments.add((Long) a.longValue());
        }
        bArguments = new ArrayList<>();
        dArguments = new ArrayList<>();
    }

    /**
     * Initialize this operation for execution (pre created ndarrays)
     *
     * @param inputs  the inputs
     * @param outputs the outputs of the op, may be null
     */
    public DynamicCustomOp(INDArray[] inputs, INDArray[] outputs) {
        this(null, inputs, outputs);
    }


    /**
     * Initialize this operation for execution (pre created ndarrays)
     *
     * @param opName  the operation opName to use for invocation
     * @param inputs  the inputs
     * @param outputs the outputs of the op, may be null
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
        bArguments = new ArrayList<>();
        dArguments = new ArrayList<>();
        this.inplaceCall = inPlace;
    }

    public DynamicCustomOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        this(null, sameDiff, args, inPlace);
    }

    protected DynamicCustomOp(String opName) {
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
        bArguments = new ArrayList<>();
        dArguments = new ArrayList<>();
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
            val outputNames = sameDiff.getOutputsForOp(this);
            //no need to dynamically create if already exists
            if (outputNames != null) {
                outputVariables = new SDVariable[outputNames.length];
                for (int i = 0; i < outputVariables.length; i++) {
                    outputVariables[i] = sameDiff.getVariable(outputNames[i]);
                }

                return outputVariables;
            }

            val newVars = sameDiff.generateOutputVariableForOp(this, baseName, false); //Also adds outgoing
            if (isInplaceCall()) {
                if (args().length >= 1) {
                    val arr = args()[0].getArr();
                    if (arr != null) {
                        sameDiff.setArrayForVariable(newVars[0].name(), arr);
                        addOutputArgument(arr);
                    }
                }

                return newVars;
            }

            outputVariables = newVars;
            if (sameDiff.getOutputsForOp(this) == null)
                sameDiff.addOutgoingFor(outputVariables, this);
            return newVars;
        }

        return outputVariables;
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
    public int numDArguments() {
        return dArguments.size();
    }

    @Override
    public List<INDArray> outputArguments() {
        return outputArguments;
    }

    @Override
    public List<INDArray> inputArguments() {
        return inputArguments;
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
    public DataType[] dArgs() {
        return dArguments.toArray(new DataType[dArguments.size()]);
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
        return iArguments == null ? 0 : iArguments.size();
    }

    @Override
    public int numBArguments() {
        return bArguments == null ? 0 : bArguments.size();
    }

    @Override
    public void addTArgument(double... arg) {
        if (arg != null)
            addTArgument(Doubles.asList(arg).toArray(new Double[arg.length]));
    }

    @Override
    public void addDArgument(DataType... arg) {
        if (dArguments == null)
            dArguments = new ArrayList<>();

        if (arg != null)
            dArguments.addAll(Arrays.asList(arg));
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
        return tArguments == null ? 0 : tArguments.size();
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
                if (i >= arrsSoFar.size())
                    continue;

                if (!Arrays.equals(args[i].getShape(), arrsSoFar.get(i).shape()))
                    throw new ND4JIllegalStateException("Illegal array passed in as argument [" + i + "]. Expected shape " + Arrays.toString(args[i].getShape()) + " and received array with shape " + Arrays.toString(arg[i].shape()));
            }
        }
    }

    @Override
    public void removeInputArgument(INDArray arg) {
        inputArguments.remove(arg);
    }

    @Override
    public INDArray getInputArgument(int index) {
        if(inputArguments == null || index >= inputArguments.size())
            return null;
        return inputArguments.get(index);
    }

    public void setInputArgument(int index, INDArray input) {
        if(index >= inputArguments.size() ){
            List<INDArray> oldArgs = inputArguments;
            inputArguments = new ArrayList<>(index+1);
            inputArguments.addAll(oldArgs);
            while(inputArguments.size() <= index)
                inputArguments.add(null);
        }
        inputArguments.set(index, input);
    }

    public void setInputArguments(INDArray... inputs){
        inputArguments.clear();
        if(inputs != null && inputs.length > 0) {
            Collections.addAll(inputArguments, inputs);
        }
    }

    public void setOutputArgument(int index, INDArray output) {
        while(index >= outputArguments.size()){
            //Resize list, in case we want to specify arrays not in order they are defined
            //For example, index 1 on empty list, then index 0
            outputArguments.add(null);
        }
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
        if(outputArguments == null || index >= outputArguments.size())
            return null;
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
    public List<LongShapeDescriptor> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        val descriptor = getDescriptor();
        if (outputShapes != null && !outputShapes.isEmpty())
            return outputShapes;

        if (descriptor == null) {
            throw new IllegalStateException("Could not find descriptor for op: " + opName()
                    + (DynamicCustomOp.class == this.getClass() ? "" : " - class: " + getClass().getName()));
        }


        //not fully initialized: missing integer args
        int nI = oc != null ? oc.numIArguments() : numIArguments();
        if (descriptor.getNumIArgs() >= 0 && nI < descriptor.getNumIArgs()) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} IArgs specified, " +
                        "{} required)", getClass().getName(), nI, descriptor.getNumIArgs());
            }
            return Collections.emptyList();
        }


        //not fully initialized: missing floating point args
        int nT = oc != null ? oc.numTArguments() : numTArguments();
        if (descriptor.getNumTArgs() >= 0 && nT < descriptor.getNumTArgs()) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} TArgs specified, " +
                        "{} required)", getClass().getName(), nT, descriptor.getNumTArgs());
            }
            return Collections.emptyList();
        }

        //not fully initialized: missing INDArray input args
        int nIn = oc != null ? oc.numInputArguments() : numInputArguments();
        if(descriptor.getNumInputs() >= 0 && nIn < descriptor.getNumInputs()){
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: not fully initialized ({} input (INDArray) args specified, " +
                        "{} required)", getClass().getName(), nIn, descriptor.getNumInputs());
            }
            return Collections.emptyList();
        }

        List<LongShapeDescriptor> ret;
        if(oc == null)
            ret = Nd4j.getExecutioner().calculateOutputShape(this);
        else
            ret = Nd4j.getExecutioner().calculateOutputShape(this, oc);
        return ret;
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

        if (descriptor.getNumInputs() > 0 && numInputArguments() < descriptor.getNumInputs()) {
            if(sameDiff == null) {
                throw new ND4JIllegalStateException("Op [" + opName() + "] failure for [" + this.getOwnName() + "]: Number of inputs is invalid for execution. "
                        + numInputArguments() + " were provided but " + descriptor.getNumInputs() + " are required for execution");
            } else {
                String[] inputNames = sameDiff.getInputsForOp(this);
                String[] arrayShapes = new String[inputNames.length];
                for( int i=0; i<inputNames.length; i++ ){
                    INDArray arr = sameDiff.getVariable(inputNames[i]).getArr();
                    arrayShapes[i] = (arr == null ? "<no array present>" : Arrays.toString(arr.shape()));
                }
                throw new ND4JIllegalStateException("Op [" + opName() + "] failure for [" + this.getOwnName() + "]: Number of inputs is invalid for execution. "
                        + numInputArguments() + " were provided but " + descriptor.getNumInputs() + " are required for execution. Input variable names: " + Arrays.toString(inputNames)
                        + ". Input variable array shapes: " + Arrays.toString(arrayShapes));
            }
        }

        if (descriptor.getNumOutputs() > 0 && numOutputArguments() < descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op [" + opName() +"] failure for [" + this.getOwnName() + "]: Number of outputs is invalid for execution. Specified [" + numOutputArguments() + "] but should be [" + descriptor.getNumOutputs()  +"]");

        //< 0 means dynamic size
        if (descriptor.getNumIArgs() >= 0 && numIArguments() < descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op [" + opName() +"] failure for [" + this.getOwnName() + "]: Number of integer arguments is invalid for execution. Specified [" + numIArguments() + "] but should be [" + descriptor.getNumIArgs()  +"]");

        if (descriptor.getNumTArgs() >= 0 && numTArguments() < descriptor.getNumTArgs())
            throw new ND4JIllegalStateException("Op [" + opName() + "] failure for [" + this.getOwnName() + "]: Number of inputs is invalid for execution. Specified [" + numTArguments() + "] but should be [" + descriptor.getNumTArgs() +"]");

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

    @Override
    public boolean[] bArgs() {
        val result = new boolean[bArguments == null ? 0 : bArguments.size()];

        for (int e = 0; e < result.length; e++)
            result[e] = bArguments.get(e);

        return result;
    }

    @Override
    public void addBArgument(boolean... arg) {
        if(arg != null) {
            for (val b : arg)
                bArguments.add(b);
        }
    }

    @Override
    public Boolean getBArgument(int index) {
        return bArguments.get(index);
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
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public void clearArrays(){
        inputArguments.clear();
        outputArguments.clear();
    }

    protected static SDVariable[] wrapOrNull(SDVariable in){
        return in == null ? null : new SDVariable[]{in};
    }

    protected static INDArray[] wrapOrNull(INDArray in){
        return in == null ? null : new INDArray[]{in};
    }

    protected static <T> T[] wrapFilterNull(T... in){
        int count = 0;
        for( int i=0; i<in.length; i++ ) {
            if (in[i] != null) count++;
        }
        T[] out = (T[]) Array.newInstance(in.getClass().getComponentType(), count);
        int j=0;
        for( int i=0; i<in.length; i++ ){
            if(in[i] != null){
                out[j++] = in[i];
            }
        }
        return out;
    }

    public static class DynamicCustomOpsBuilder {
        protected String opName;
        protected int numInputs;
        protected int numOutputs;
        protected int numTArguments;
        protected int numIArguments;
        protected int numBArguments;
        protected boolean inplaceCall;
        protected boolean inplaceAllowed;
        protected long opHash;
        protected List<LongShapeDescriptor> outputShapes = new ArrayList<>();

        private List<INDArray> inputArguments = new ArrayList<>();
        private List<INDArray> outputArguments = new ArrayList<>();
        private List<Double> tArguments = new ArrayList<>();
        private List<Long> iArguments = new ArrayList<>();
        private List<DataType> dArguments = new ArrayList<>();
        private List<Boolean> bArguments = new ArrayList<>();

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
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public DynamicCustomOpsBuilder addBooleanArguments(boolean... bargs) {
            for (val in : bargs)
                bArguments.add(in);

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
        /*
        public DynamicCustomOpsBuilder addOutputShape(int[] shape) {
            this.outputShapes.add(ArrayUtil.toLongArray(shape));
            return this;
        }

        public DynamicCustomOpsBuilder addOutputShape(long[] shape) {
            this.outputShapes.add(shape);
            return this;
        }
*/

        public DynamicCustomOpsBuilder addOutputShape(LongShapeDescriptor shape) {
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
            result.bArguments = bArguments;
            result.dArguments = dArguments;
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
