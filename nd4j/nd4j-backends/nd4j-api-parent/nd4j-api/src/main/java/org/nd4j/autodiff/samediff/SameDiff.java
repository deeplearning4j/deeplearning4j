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

package org.nd4j.autodiff.samediff;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import com.rits.cloning.Cloner;
import com.rits.cloning.IFastCloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.internal.*;
import org.nd4j.autodiff.samediff.ops.*;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.autodiff.util.cloner.DataBufferFastCloner;
import org.nd4j.autodiff.util.cloner.INDArrayFastCloner;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.graph.*;
import org.nd4j.jackson.objectmapper.holder.ObjectMapperHolder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4UnresolvedOutputVariables;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.DeviceLocalNDArray;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * SameDiff is the entrypoint for ND4J's automatic differentiation functionality.
 * <p>
 * You define a graph symbolically.
 * <p>
 * That graph accumulates operations.
 * <p>
 * In order to execute the graph, you run one of the execution methods, such as {@link #exec(Map, String...)}
 */
@AllArgsConstructor
@Builder
@Slf4j
public class SameDiff extends SDBaseOps {

    //Fields for graph structure and execution
    @Getter     //TODO use package private instead of public getters?
    private final Map<String,Variable> variables = new LinkedHashMap<>();         //Use linked hash map to guarantee iteration order based on order they were added. Used in inputs() and flatbuffers serde
    @Getter
    private final Map<String,SameDiffOp> ops = new LinkedHashMap<>();
    @Getter
    private final Map<Long,InferenceSession> sessions = new ConcurrentHashMap<>();      //Key: thread ID

    private final Map<String,DeviceLocalNDArray> constantArrays = new ConcurrentHashMap<>();
    private final Map<String,DeviceLocalNDArray> variablesArrays = new ConcurrentHashMap<>();     //TODO issues with DeviceLocal +  mutable / changed during training?
    private final Map<Long,Map<String,INDArray>> placeholdersPerThread = new ConcurrentHashMap<>(); //Placeholders for each thread - if the user sets them

    private final List<String> lossVariables = new ArrayList<>();

    ///////////////////////////////////////
    //Fields related to training
    @Getter
    private TrainingConfig trainingConfig;                          //Configuration for training. Must be set for training/evaluation, but not for other operations
    @Getter
    private boolean initializedTraining;                            //True if training setup has been done
    @Getter
    private INDArray updaterState;                                  //Updater state array (1d, length equal to number of trainable parameters)
    @Getter
    private Map<String,INDArray> updaterViews;                      //Views of updaterState array for each trainable parameter
    @Getter
    private Map<String,GradientUpdater> updaterMap;                 //GradientUpdater instance for each trainable parameter

    ////////////////////////////////////////
    //map a function's instance id to a base name, used for propagating variable names
    //for output during import
    private Map<String, String> baseNameForFunctionInstanceId;

    private DifferentialFunctionFactory functionFactory;
    @Deprecated //TO BE REMOVED - to ShapeSession
    private Map<String, long[]> variableNameToShape;                //Key: SDVariable name. Value: shape for that variable
    @Deprecated //TO BE REMOVED - to Variable
    private Map<String, SDVariable> forwardVarForGrad;

    // counter for auto-naming variables
    private int variableId = 0;

    ////////////////////////////////////////

    /** Op creator object for math operations */
    public final SDMath math = new SDMath(this);
    /** Op creator object for random number generation operations */
    public final SDRandom random = new SDRandom(this);
    /** Op creator object for general neural network operations */
    public final SDNN nn = new SDNN(this);
    /** Op creator object for convolutional neural network operations */
    public final SDCNN cnn = new SDCNN(this);
    /** Op creator object for recurrent neural network operations */
    public final SDRNN rnn = new SDRNN(this);
    /** Op creator object for loss function operations */
    public final SDLoss loss = new SDLoss(this);

    /** Op creator object for math operations */
    public SDMath math(){
        return math;
    }

    /** Op creator object for random number generation operations */
    public SDRandom random(){
        return random;
    }

    /** Op creator object for general neural network operations */
    public SDNN nn(){
        return nn;
    }

    /** Op creator object for convolutional neural network operations */
    public SDCNN cnn(){
        return cnn;
    }

    /** Op creator object for recurrent neural network operations */
    public SDRNN rnn(){
        return rnn;
    }

    /** Op creator object for loss function operations */
    public SDLoss loss(){
        return loss;
    }



    /**
     * For import, many times we have variables
     * that map to properties. Most common
     * we will have an input to a function that is mapped to an ndarray.
     * That ndarray is usually a scalar shape.
     * <p>
     * That array with a scalar shape can be something like an axis.
     * <p>
     * We often don't know that array's value till run time.
     * This map stores variable names  that we should resolve
     * from samediff. We use the value of that array
     * to update the properties.
     */
    private Map<String, List<String>> propertiesToResolve;

    /**
     * A map of own name to
     * the properties of the function (things like execution axes etc)
     * The valid values can be:
     * int
     * long
     * INDArray
     */
    private Map<String, Map<String, Object>> propertiesForFunction;

    @Deprecated //TO BE REMOVED - to Variable
    private Map<String, long[]> placeHolderOriginalShapes;
    private Map<String, SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String, SameDiff> sameDiffFunctionInstances;
    private Set<String> placeHolderFunctions;
    private static Cloner cloner = newCloner();
    private static Map<String, Method> opMethods;

    private Table<String, String, String> fieldVariableResolutionMapping;

    // flag, shows if graph was already registered with libnd4j
    private transient AtomicBoolean wasRegistered = new AtomicBoolean(false);


    //debug mode variables
    @Getter
    private boolean debugMode;
    private Map<int[], Op> opsForResult;
    private boolean resolvedVariables = false;


    @Getter
    @Setter
    boolean logExecution = true;

    @Getter
    private SameDiff parent;

    @Getter
    private SameDiff child;

    public final static String TRAINING_CONFIG_JSON_ZIP_ENTRY_NAME = "trainingConfig.json";
    public final static String SAMEDIFF_FILE_ENTRY_NAME = "samediff.fb";

    static {
        opMethods = new HashMap<>();
        Method[] methods = SameDiff.class.getDeclaredMethods();
        for (Method method : methods) {
            if (method.getReturnType().equals(SDVariable.class)) {
                opMethods.put(method.getName(), method);
            }
        }
    }

    /**
     * @return New cloner object. NOTE: INTENDED FOR DEVELOPER USE ONLY
     */
    public static Cloner newCloner() {
        Cloner cloner = new Cloner();

        //Implement custom cloning for INDArrays (default can have problems with off-heap and pointers)
        //Sadly: the cloner library does NOT support interfaces here, hence we need to use the actual classes
        //cloner.registerFastCloner(INDArray.class, new INDArrayFastCloner());  //Does not work due to interface
        IFastCloner fc = new INDArrayFastCloner();
        cloner.registerFastCloner(Nd4j.getBackend().getNDArrayClass(), fc);

        //Same thing with DataBuffers: off heap -> cloner library chokes on them, but need to know the concrete
        // buffer classes, not just the interface
        IFastCloner fc2 = new DataBufferFastCloner();
        DataBufferFactory d = Nd4j.getDataBufferFactory();
        doReg(cloner, fc2, d.intBufferClass());
        doReg(cloner, fc2, d.longBufferClass());
        doReg(cloner, fc2, d.halfBufferClass());
        doReg(cloner, fc2, d.floatBufferClass());
        doReg(cloner, fc2, d.doubleBufferClass());
        doReg(cloner, fc2, CompressedDataBuffer.class);
        return cloner;
    }

    private static void doReg(Cloner cl, IFastCloner fc, Class<?> c) {
        if (c != null)
            cl.registerFastCloner(c, fc);
    }


    /**
     * Update the opName for the variable
     * with the given vertex id
     *
     * @param varName  the vertex id to update
     * @param withName thew new opName
     */
    public void updateVariableName(String varName, String withName) {
        SDVariable oldVarNameRef = getVariable(varName);
        Variable v = variables.remove(varName);
        String oldVarName = varName;
        oldVarNameRef.setVarName(withName);
        v.setName(withName);
        variables.put(withName, v);

        for(SameDiffOp op : ops.values()){
            List<String> outputsOfOp = op.getOutputsOfOp();
            if(outputsOfOp != null && !outputsOfOp.isEmpty()) {
                for (int i = 0; i < outputsOfOp.size(); i++) {
                    if (outputsOfOp.get(i).equals(oldVarName)) {
                        outputsOfOp.set(i, withName);
                    }
                }
            }

            List<String> inputsToOp = op.getInputsToOp();
            if(inputsToOp != null && !inputsToOp.isEmpty()) {
                for (int i = 0; i < inputsToOp.size(); i++) {
                    if (inputsToOp.get(i).equals(oldVarName)) {
                        inputsToOp.set(i, withName);
                    }
                }
            }
        }

//        if (variableNameToArr.containsKey(oldVarName)) {
//            val arr = variableNameToArr.remove(oldVarName);
//            variableNameToArr.put(withName, arr);
//        }


        if (variableNameToShape.containsKey(oldVarName)) {
            val shape = variableNameToShape.remove(oldVarName);
            variableNameToShape.put(withName, shape);
        }

        if (forwardVarForGrad.containsKey(oldVarName)) {
            val forwardGrad = forwardVarForGrad.remove(oldVarName);
            forwardVarForGrad.put(withName, forwardGrad);
        }


        if (v.getInputsForOp() != null) {
            List<String> funcNames = v.getInputsForOp();
            for (String s : funcNames) {
                DifferentialFunction func = ops.get(s).getOp();
                if (func instanceof BaseOp) {
                    BaseOp baseOp = (BaseOp) func;
                    if (baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                        baseOp.setXVertexId(withName);
                    }

                    if (baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                        baseOp.setYVertexId(withName);
                    }

                    if (baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                        baseOp.setZVertexId(withName);
                    }

                }
            }
        }


        if (v.getOutputOfOp() != null) {
            DifferentialFunction func = ops.get(v.getOutputOfOp()).getOp();
            if (func instanceof BaseOp) {
                BaseOp baseOp = (BaseOp) func;
                if (baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                    baseOp.setXVertexId(withName);
                }

                if (baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                    baseOp.setYVertexId(withName);
                }

                if (baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                    baseOp.setZVertexId(withName);
                }
            }
        }
    }


    /**
     * Clears debugging state and disables debug mode.
     */
    public SameDiff disableDebugging() {
        debugMode = false;
        return this;
    }

    /**
     * Enables tracing of graphs automatically.
     */
    public SameDiff enableDebugMode() {
        debugMode = true;
        return this;
    }

    /**
     * Returns this samediff instance's {@link DifferentialFunctionFactory}
     *
     * @return
     */
    public DifferentialFunctionFactory f() {
        return functionFactory;
    }


    /**
     * @param sameDiff
     * @return
     */
    public SDVariable invokeGraphOn(SameDiff sameDiff) {
        //map the new vertices on to the old ones
        Map<Integer, Integer> thisVertexIdToNew = new HashMap<>();
        int idx = 1;
        for (val var : variables()) {
            SDVariable clone = cloner.deepCloneDontCloneInstances(var, var.getSameDiff());
            SDVariable newVar = sameDiff.var(clone);
            if (var.getArr() != null && var.getVariableType() != VariableType.ARRAY) {      //ARRAY type = "activations" - are overwritten anyway
                sameDiff.associateArrayWithVariable(var.getArr(), newVar);
            }


            thisVertexIdToNew.put(idx, idx);
            clone.setSameDiff(sameDiff);
            idx++;

        }


        val newFunctions = new LinkedHashMap<String, DifferentialFunction>();
        for (SameDiffOp op : ops.values()) {
            DifferentialFunction function = op.getOp();
            if (function instanceof SDVariable) {
                continue;
            }

            DifferentialFunction clone = cloner.deepCloneDontCloneInstances(
                    function,
                    function.getSameDiff());
            clone.setSameDiff(sameDiff);
            clone.setOwnName(function.getOwnName());
            if (sameDiff.functionExists(function.getOwnName()))
                sameDiff.putFunctionForId(function.getOwnName(), function);
            newFunctions.put(function.getOwnName(), clone);

            val argsForFunction = function.args();
            val outputsForFunction = function.outputVariables();


            //note that these have the same variable names
            sameDiff.addArgsFor(argsForFunction, clone);
            sameDiff.addOutgoingFor(outputsForFunction, function);

            for (val arg : clone.args()) {
                arg.setSameDiff(sameDiff);
            }

            for (val output : clone.outputVariables()) {
                output.setSameDiff(sameDiff);
            }

            sameDiff.ops.put(function.getOwnName(), op);
        }

        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }


    /**
     * Returns true if the given function id exists
     *
     * @param id the function id to test for
     * @return true if the function id exists, false otherwise
     */
    public boolean functionExists(String id) {
        return ops.containsKey(id);
    }

    public DifferentialFunction functionOutputFor(String varName){
        if(variables.get(varName).getOutputOfOp() == null)
            return null;
        String outName = variables.get(varName).getOutputOfOp();
        if(outName == null)
            return null;
        return ops.get(outName).getOp();
    }

    /**
     * Get the function by the {@link DifferentialFunction#getOwnName()}
     *
     * @param id the id of the function
     * @return the function for the given id if it exists
     */
    public DifferentialFunction getFunctionById(@NonNull String id) {
        if (!ops.containsKey(id)) {
            throw new ND4JIllegalStateException("No function with id " + id + " found!");
        }
        return ops.get(id).getOp();
    }


    /**
     * Put the function for the given id
     *
     * @param id       the id of the function
     * @param function the function
     */
    public void putFunctionForId(String id, DifferentialFunction function) {
        if (ops.containsKey(id) && ops.get(id).getOp() == null) {
            throw new ND4JIllegalStateException("Function by id already exists!");
        } else if (function instanceof SDVariable) {
            throw new ND4JIllegalStateException("Function must not be a variable!");
        }

        if(ops.containsKey(id)){

        } else {
            ops.put(id, SameDiffOp.builder().name(id).op(function).build());
        }
    }


    /**
     * Returns the name(s) of the inputs for the given function
     *
     * @param function the function to get the inputs for
     * @return the input ids for a given function
     */
    public String[] getInputsForFunction(DifferentialFunction function) {
        if (!ops.containsKey(function.getOwnName()))
            throw new ND4JIllegalStateException("Illegal function instance id found " + function.getOwnName());
        List<String> inputs = ops.get(function.getOwnName()).getInputsToOp();
        return inputs == null ? null : inputs.toArray(new String[inputs.size()]);
    }

    /**
     * Returns the name(s) of the outputs for the given function
     *
     * @param function the function to get the outputs for
     * @return the outputs ids for a given function
     */
    public String[] getOutputsForFunction(DifferentialFunction function) {
        if (!ops.containsKey(function.getOwnName()))
            throw new ND4JIllegalStateException("Illegal function instance id found " + function.getOwnName());
        List<String> outputs = ops.get(function.getOwnName()).getOutputsOfOp();
        return outputs == null ? null : outputs.toArray(new String[outputs.size()]);
    }


    /**
     * Get the output variable(s) for the specified differential function
     *
     * @param function the function reference to get the output variable(s) for
     * @return the output variables for the given function
     */
    public SDVariable[] getOutputVariablesForFunction(DifferentialFunction function) {
        val inputs = getOutputsForFunction(function);
        if (inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
        }

        return vars;
    }


    /**
     * Get the input variable(s) for the specified differential function
     *
     * @param function the function reference to get the input variable(s) for
     * @return the input variables for the given function
     */
    public SDVariable[] getInputVariablesForFunction(DifferentialFunction function) {
        val inputs = getInputsForFunction(function);
        if (inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
            if (vars[i] == null) {
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            }
        }

        return vars;
    }


    public void setArrayForVariable(@NonNull String varName, @NonNull INDArray arr){
        Preconditions.checkState(variables.containsKey(varName), "No variable with name \"%s\" exists", varName);

        SDVariable v = getVariable(varName);
        if(v.isConstant()) {
            constantArrays.put(varName, new DeviceLocalNDArray(arr));
        } else if(v.getVariableType() == VariableType.VARIABLE) {
            variablesArrays.put(varName, new DeviceLocalNDArray(arr));
        } else if(v.isPlaceHolder()){
            long tid = Thread.currentThread().getId();
            if(!placeholdersPerThread.containsKey(tid)){
                placeholdersPerThread.put(tid, new HashMap<String, INDArray>());
            }
            placeholdersPerThread.get(tid).put(varName, arr);
        } else {
            throw new UnsupportedOperationException("Cannot set variable of type " + v.getVariableType() + " using this method");
        }
    }


    /**
     * Get the shape for the given vertex id.
     * Note that if an array is defined, it will use the shape of the array instead.
     * <p>
     * A shape *and* an array should not be defined at the same time.
     * This wastes memory. The internal map used for tracking shapes for particular
     * vertex ids should also delete redundant shapes stored to avoid redundant sources of information.
     *
     * @param varName the vertex id to get the shape for
     * @return the shape for the given vertex if any.
     */
    public long[] getShapeForVarName(String varName) {
        if (arrayAlreadyExistsForVarName(varName)) {
            return getVariable(varName).getArr().shape();
        }
        return variableNameToShape.get(varName);
    }

    public LongShapeDescriptor getShapeDescriptorForVarName(String varName) {
        if (getVariable(varName).getArr() != null) {
            return getVariable(varName).getArr().shapeDescriptor();
        }
        // FIXME: do we really want this Nd4j.dataType() here?
        return LongShapeDescriptor.fromShape(variableNameToShape.get(varName), Nd4j.dataType());
    }


    /**
     * Associate a vertex id with the given shape.
     *
     * @param varName the vertex id to associate
     * @param shape   the shape to associate with
     * @see #putShapeForVarName(String, long[])
     * @see #putOrUpdateShapeForVarName(String, long[], boolean)
     */
    @Deprecated
    public void putShapeForVarName(String varName, long[] shape) {
        if (shape == null) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }

        if (variableNameToShape.containsKey(varName)) {
            throw new ND4JIllegalStateException("Shape for " + varName + " already exists!");
        }

        variableNameToShape.put(varName, shape);
    }


    public void putShapeForVarName(String varName, LongShapeDescriptor shape) {
        val v = getVariable(varName);
        putShapeForVarName(varName, shape.getShape());
        v.setDataType(shape.dataType());
    }

    /**
     * Put or update the shape for the given variable name. Optionally supports clearing the specified variable's
     * INDArray if it's shape does not match the new shape
     * @param varName                   Variable name
     * @param shape                     Shape to put
     * @param clearArrayOnShapeMismatch If false: no change to arrays. If true: if an INDArray is defined for the specified
     *                                  variable name, it will be removed from the graph (to be later re-generated) if
     *                                  its shape does not match the specified shape
     */
    @Deprecated
    public void putOrUpdateShapeForVarName(String varName, long[] shape, boolean clearArrayOnShapeMismatch){
        Preconditions.checkNotNull(shape, "Cannot put null shape for variable: %s", varName);
        if(variableNameToShape.containsKey(varName)){
//            updateShapeForVarName(varName, shape, clearArrayOnShapeMismatch);
            //TODO
        } else {
            putShapeForVarName(varName, shape);
        }
    }

    /**
     * Returns true if the given vertex id and shape already exist.
     *
     * @param varName the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean shapeAlreadyExistsForVarName(String varName) {
        return variableNameToShape.containsKey(varName) || arrayAlreadyExistsForVarName(varName);
    }


    /**
     * Returns true if the given vertex id and {@link INDArray} already exist.
     *
     * @param varName the vertex id
     * @return true if a vertex with the given INDArray exists, and it has an INDArray associated with it
     */
    public boolean arrayAlreadyExistsForVarName(String varName) {
        SDVariable var = getVariable(varName);
        switch(var.getVariableType()){
            case VARIABLE:
                return variablesArrays.containsKey(varName);
            case ARRAY:
                long tid = Thread.currentThread().getId();
                return sessions.containsKey(tid) && sessions.get(tid).contains(varName, InferenceSession.OUTER_FRAME, 0, null);
            case CONSTANT:
                return constantArrays.containsKey(varName);
            case PLACEHOLDER:
                return placeholdersPerThread.containsKey(Thread.currentThread().getId()) &&
                        placeholdersPerThread.get(Thread.currentThread().getId()).containsKey(varName);
            default:
                throw new RuntimeException("Unknown variable type: " + var.getVariableType());
        }
    }

    /**
     * Get an {@link INDArray} for a given vertex id, or null if none exists
     *
     * @param varName Variable name to get the array for
     * @return Array, or null if none exists
     */
    public INDArray getArrForVarName(@NonNull String varName) {
        Preconditions.checkState(variables.containsKey(varName), "No variable found with name \"%s\"", varName);
        SDVariable v = variables.get(varName).getVariable();
        switch(v.getVariableType()){
            case VARIABLE:
                if(!variablesArrays.containsKey(varName)) {
                    //VARIBALE type arrays should have a parameter initializer...
                    // we should use this to azy init the array if none is present
                    v.storeAndAllocateNewArray();
                }
                return variablesArrays.get(varName).get();
            case CONSTANT:
                if(!constantArrays.containsKey(varName))
                    return null;
                return constantArrays.get(varName).get();
            case ARRAY:
                //Only stored in inference session...
                InferenceSession s = sessions.get(Thread.currentThread().getId());
                if(s == null)
                    return null;
                return s.get(varName, InferenceSession.OUTER_FRAME, 0, null, false);
            case PLACEHOLDER:
                long tid = Thread.currentThread().getId();
                if(placeholdersPerThread.get(tid) == null || !placeholdersPerThread.get(tid).containsKey(varName))
                    return null;
                return placeholdersPerThread.get(tid).get(varName);
            default:
                throw new RuntimeException("Unknown variable type: " + v.getVariableType());
        }
    }

    /**
     * Associate the array with the given variable.
     *
     * @param arr      the array to get the variable for
     * @param variable the name of the variable to associate the array with
     */
    public void associateArrayWithVariable(INDArray arr, @NonNull String variable) {
    Preconditions.checkState(variables.containsKey(variable), "Cannot associate array with variable \"%s\": " +
            "variable \"%s\" does not exist in this SameDiff instance", variable, variable);
        associateArrayWithVariable(arr, this.getVariable(variable));
    }

    /**
     * Associate the array with the given variable.
     *
     * @param arr      the array to get the variable for
     * @param variable the variable to associate the array with
     */
    public void associateArrayWithVariable(INDArray arr, SDVariable variable) {
        if (variable == null) {
            throw new ND4JIllegalArgumentException("Variable must not be null!");
        }
        if (arr == null) {
            throw new ND4JIllegalArgumentException("Array must not be null");
        }

        if (variable.dataType() != arr.dataType())
            arr = arr.castTo(variable.dataType());

        Preconditions.checkState(variable.dataType() == arr.dataType(), "Variable \"%s\" has datatype %s: cannot associate array with type %s with this variable",
                variable.getVarName(), variable.dataType(), arr.dataType());

        // FIXME: remove this before release
        if (sessions.get(Thread.currentThread().getId()) == null) {
            sessions.put(Thread.currentThread().getId(), new InferenceSession(this));
        }

        boolean duped = false;
        if(arr.isAttached()) {
            arr = arr.detach();
            duped = true;
        }
        if(arr.isView()) {
            arr = arr.dup();
            duped = true;
        }

        if(!duped && variable.getVariableType() == VariableType.VARIABLE) {
            for (DeviceLocalNDArray otherArr : variablesArrays.values()) {
                if (otherArr.get() == arr) {    //Check for exact same object, to avoid array reuse (can result in unexpected behaviour)
                    arr = arr.dup();
                    break;
                }
            }
        }

        switch(variable.getVariableType()){
            case VARIABLE:
                variablesArrays.put(variable.getVarName(), new DeviceLocalNDArray(arr));
                break;
            case CONSTANT:
                constantArrays.put(variable.getVarName(), new DeviceLocalNDArray(arr));
                break;
            case ARRAY:
                // FIXME: remove this before release
                val session = sessions.get(Thread.currentThread().getId());
                val varId = session.newVarId(variable.getVarName(), AbstractSession.OUTER_FRAME, 0, null);
                session.getNodeOutputs().put(varId, arr);
                //throw new UnsupportedOperationException("Cannot associate array with SDVariable of type ARRAY");
            case PLACEHOLDER:
                long tid = Thread.currentThread().getId();
                if(!placeholdersPerThread.containsKey(tid)){
                    placeholdersPerThread.put(tid, new HashMap<String, INDArray>());
                }
                placeholdersPerThread.get(tid).put(variable.getVarName(), arr);
                break;
            default:
                throw new IllegalStateException("Unknown variable type: " + variable.getVariableType());
        }

        //putOrUpdateShapeForVarName(variable.getVarName(), arr.shape(), true);

        //Also update nested SameDiff instances (such as gradient function)
        if(sameDiffFunctionInstances != null && sameDiffFunctionInstances.size() > 0){
            for(Map.Entry<String,SameDiff> e : sameDiffFunctionInstances.entrySet()){
                SameDiff sd = e.getValue();
                SDVariable v = sd.getVariable(variable.getVarName());
                if(v != null){
                    sd.associateArrayWithVariable(arr, v);
                }
            }
        }
    }


    /**
     * Associate a {@link SameDiff} namespace as a sub function.
     *
     * @param name      the opName of the function
     * @param nameSpace the namespace
     */
    public void putSubFunction(String name, SameDiff nameSpace) {
        if (sameDiffFunctionInstances.containsKey(name) && sameDiffFunctionInstances.get(name) != nameSpace) {
            throw new ND4JIllegalStateException("Unable to replace samediff namespace. Please choose another opName");
        }

        sameDiffFunctionInstances.put(name, nameSpace);
    }


    /**
     * Return a copy of the internal variable map
     *
     * @return Map of variables by name
     */
    public Map<String, SDVariable> variableMap() {
        Map<String,SDVariable> ret = new LinkedHashMap<>();
        for(Variable v : variables.values()){
            ret.put(v.getName(), v.getVariable());
        }
        return ret;
    }


    /**
     * Invoke an op by opName
     *
     * @param op the op
     * @param x  the first input
     * @param y  the second input
     * @return the result variable
     */
    @Deprecated //TO BE REMOVED - should not be part of public API
    public SDVariable invoke(Op op, SDVariable x, SDVariable y) {
        if (!opMethods.containsKey(op.opName())) {
            throw new ND4JIllegalStateException("Illegal method opName " + op.opName());
        }

        if (x != null && y != null) {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x, y);
            } catch (Exception e) {

            }
        } else {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x);
            } catch (Exception e) {

            }
        }

        throw new ND4JIllegalStateException("Illegal method opName " + op.opName());
    }

    /**
     * The set of defined SameDiff function names. SameDiff function instances should not be confused
     * with DifferentialFunction ops; an example of a SameDiff function instance is the gradient "grad" function
     *
     * @return Set of defined SameDiff function instance names
     */
    public Collection<String> definedFunctionNames() {
        return this.sameDiffFunctionInstances.keySet();
    }

    /**
     * Invoke an op by opName
     *
     * @param op the op
     * @param x  the first input
     * @return the result variable
     */
    public SDVariable invoke(Op op, SDVariable x) {
        return invoke(op, x, null);
    }

    private SameDiff() {
        functionFactory = new DifferentialFunctionFactory(this);
        sameDiffFunctionDefinitionMap = new LinkedHashMap<>();
        sameDiffFunctionInstances = new LinkedHashMap<>();
        forwardVarForGrad = new LinkedHashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        variableNameToShape = new LinkedHashMap<>();
        placeHolderOriginalShapes = new LinkedHashMap<>();
        placeHolderFunctions = new LinkedHashSet<>();
        baseNameForFunctionInstanceId = new LinkedHashMap<>();
        propertiesToResolve = new LinkedHashMap<>();
        propertiesForFunction = new LinkedHashMap<>();
        fieldVariableResolutionMapping = HashBasedTable.create();

    }

    /**
     * Adds a property that needs to be resolve for later.
     * These variables are typically values that are arrays
     * that are named but have an unknown value till execution time.
     * <p>
     * This is very common for model import.
     *
     * @param forFunction the function to add the property to resolve for
     * @param arrayName   the array name
     */
    public void addPropertyToResolve(DifferentialFunction forFunction, String arrayName) {
        if (!propertiesToResolve.containsKey(forFunction.getOwnName())) {
            List<String> newVal = new ArrayList<>();
            newVal.add(arrayName);
            propertiesToResolve.put(forFunction.getOwnName(), newVal);
        } else {
            List<String> newVal = propertiesToResolve.get(forFunction.getOwnName());
            newVal.add(arrayName);
        }
    }

    /**
     * Return the properties to resolve for the given function.
     * This is typically used right before execution in model import in
     * {@link DifferentialFunction#resolvePropertiesFromSameDiffBeforeExecution()}
     *
     * @param function the function get the properties to resolve for
     * @return the properties to resolve for the given function
     */
    public List<String> propertiesToResolveForFunction(DifferentialFunction function) {
        if (!propertiesToResolve.containsKey(function.getOwnName()))
            return Collections.emptyList();

        return propertiesToResolve.get(function.getOwnName());
    }


    /**
     * Returns true if the given function has ndarray properties to resolve.
     *
     * @param function the function to check
     * @return true if the function has yet to be resolved properties
     */
    public boolean hasPropertiesToResolve(DifferentialFunction function) {
        return propertiesToResolve.containsKey(function.getOwnName());
    }


    /**
     * Get the property for a given function
     *
     * @param functionInstance the function to get the
     *                         property for
     * @param propertyName     the name of the property to get
     * @param <T>              the inferred return type
     * @return the property for the given function
     */
    public <T> T getPropertyForFunction(DifferentialFunction functionInstance, String propertyName) {
        if (!propertiesForFunction.containsKey(functionInstance.getOwnName())) {
            return null;
        } else {
            val map = propertiesForFunction.get(functionInstance.getOwnName());
            return (T) map.get(propertyName);

        }
    }

    /**
     * Add a property for the given function
     *
     * @param functionFor  the function add a property for
     * @param propertyName the property name
     * @param property     the property value
     */
    public void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, INDArray property) {
        addPropertyForFunction(functionFor, propertyName, (Object) property);
    }


    /**
     * Add a property for the given function
     *
     * @param functionFor  the function to add the property for
     * @param propertyName the name of the property to add the value for
     * @param property     the property value to add
     */
    public void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, long property) {
        addPropertyForFunction(functionFor, propertyName, (Object) property);
    }


    private void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, Object propertyValue) {
        if (!propertiesForFunction.containsKey(functionFor.getOwnName())) {
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put(propertyName, propertyValue);
            propertiesForFunction.put(functionFor.getOwnName(), fields);
        } else {
            val fieldMap = propertiesForFunction.get(functionFor.getOwnName());
            if (fieldMap.containsKey(propertyName)) {
                throw new ND4JIllegalStateException("Attempting to override property " + propertyName);
            }

            fieldMap.put(propertyName, propertyValue);
        }
    }


    /**
     * Adds a field name -> variable name mapping for a given function.<br>
     * This is used for model import where there is an unresolved variable at the time of calling any
     * {@link org.nd4j.imports.graphmapper.GraphMapper#importGraph(File)}
     * .
     * <p>
     * This data structure is typically accessed during {@link DifferentialFunction#resolvePropertiesFromSameDiffBeforeExecution()}
     * <p>
     * When a function attempts to resolve variables right before execution, there needs to be a way of knowing
     * which variable in a samediff graph should map to a function's particular field name
     *
     * @param function  the function to map
     * @param fieldName the field name for the function to map
     * @param varName   the variable name of the array to get from samediff
     */
    public void addVariableMappingForField(DifferentialFunction function, String fieldName, String varName) {
        fieldVariableResolutionMapping.put(function.getOwnName(), fieldName, varName);
    }

    /**
     * Get the variable name to use
     * for resolving a given field
     * for a given function during import time.
     * This method is u sed during {@link DifferentialFunction#resolvePropertiesFromSameDiffBeforeExecution()}
     *
     * @param function  the function to get the variable name for
     * @param fieldName the field name to resolve for
     * @return the resolve variable name if any
     */
    public String getVarNameForFieldAndFunction(DifferentialFunction function, String fieldName) {
        return fieldVariableResolutionMapping.get(function.getOwnName(), fieldName);
    }

    /**
     * Sets a base name for the function id.
     * This is used for when calling {@link #generateOutputVariableForOp(DifferentialFunction, String)}
     * for ensuring original names for model import map to current samediff names
     * when names are generated.
     *
     * @param baseName the base name to add
     * @param function the function to declare a base name for.
     */
    public void setBaseNameForFunctionInstanceId(String baseName, DifferentialFunction function) {
        baseNameForFunctionInstanceId.put(function.getOwnName(), baseName);
    }

    /**
     * Returns the base name for the given function
     * if any (may return null)
     *
     * @param function the function to get the base name for
     * @return the base name for the given function (if any) based
     * on the function's instance id.
     */
    public String getBaseNameForFunction(DifferentialFunction function) {
        return baseNameForFunctionInstanceId.get(function.getOwnName());
    }


    /**
     * Attempts to insert the {@link DifferentialFunction} reference in to this {@link SameDiff} instance.
     * If the given array field with the given index already exists, it will do a reference check to ensure that the 2
     * array fields are the same. If not, an exception is thrown.<br>
     * If the instances are the same (by semantics, not reference) then it will just return the original instance.
     * This is to ensure that instances that are created are unique and reference checked.
     *
     * @param function the array field to attempt to create
     * @return Original instance
     */
    public <X extends SDVariable> X setupFunction(X function) {
        Preconditions.checkNotNull(function, "Passed in function must not be null!");
        if (function instanceof SDVariable) {
            if (function.getSameDiff() != this) {
                function.setSameDiff(this);
            }
            return function;
        }
        return function;
    }


    /**
     * Adds outgoing arguments to the graph for the specified DifferentialFunction
     * Also checks for input arguments and updates the graph adding an appropriate edge when the full graph is declared.
     *
     * @param variables Variables - arguments for the specified differential function
     * @param function Differential function
     */
    public void addOutgoingFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            varNames[i] = variables[i].getVarName();
        }

        addOutgoingFor(varNames, function);
    }


    /**
     * Adds outgoing arguments to the graph for the specified DifferentialFunction
     * Also checks for input arguments and updates the graph adding an appropriate edge when the full graph is declared.
     *
     * @param varNames Name of the variables that are outputs of the specified differential function
     * @param function Differential function
     */
    public void addOutgoingFor(String[] varNames, DifferentialFunction function) {

        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");


        if (ops.get(function.getOwnName()).getOutputsOfOp() != null && !ops.get(function.getOwnName()).getOutputsOfOp().isEmpty()) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for " + function);
        }

        if (varNames == null)
            throw new ND4JIllegalStateException("Var names can not be null!");


        for (int i = 0; i < varNames.length; i++) {
            if (varNames[i] == null)
                throw new ND4JIllegalStateException("Variable name elements can not be null!");
        }

        ops.get(function.getOwnName()).setOutputsOfOp(Arrays.asList(varNames));

        for (String resultName : varNames) {
            variables.get(resultName).setOutputOfOp(function.getOwnName());
        }
    }

    /**
     * Adds incoming arguments for the specified differential function to the graph
     *
     * @param variables Name of the variables that are arguments (inputs) to the specified function
     * @param function  Function
     */
    public void addArgsFor(String[] variables, DifferentialFunction function) {
        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        //double check if function contains placeholder args
        for (val varName : variables) {
            if (isPlaceHolder(varName)) {
                placeHolderFunctions.add(function.getOwnName());
            }
        }

        //Add function if it doesn't exist
        //TODO could "not existing" be a bug sometimes?
        if(!ops.containsKey(function.getOwnName())){
            ops.put(function.getOwnName(), SameDiffOp.builder().name(function.getOwnName()).op(function).build());
        }

        //Update variable 'inputs to op' accounting for repeated inputs (like y = x+x)
        ops.get(function.getOwnName()).setInputsToOp(Arrays.asList(variables));     //Duplicate variables OK/required here

        for (String variableName : variables) {
            List<String> funcs = this.variables.get(variableName).getInputsForOp();
            if (funcs == null) {
                funcs = new ArrayList<>();
                this.variables.get(variableName).setInputsForOp(funcs);
            }
            if(!funcs.contains(function.getOwnName()))  //Avoid duplicates for function names.
                funcs.add(function.getOwnName());
        }
    }


    /**
     * Adds incoming arguments for the specified differential function to the graph
     *
     * @param variables variables that are arguments (inputs) to the specified function
     * @param function  Function
     */
    public void addArgsFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            if (variables[i] == null)
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            varNames[i] = variables[i].getVarName();
        }
        addArgsFor(varNames, function);
    }

    /**
     * Get the differential function (if any) that this variable is the output for
     *
     * @param variableName Name of the variable
     * @return The differential function that this variable is an output of, or null if it is not the output of a function
     */
    public DifferentialFunction getVariableOutputFunction(String variableName) {
        Preconditions.checkState(variables.containsKey(variableName), "No variable with name \"%s\" found in graph", variableName);
        if(variables.get(variableName).getOutputOfOp() == null)
            return null;
        return ops.get(variables.get(variableName).getOutputOfOp()).getOp();
    }


    /**
     * Returns true if this function already has defined arguments
     *
     * @param function the function to check
     * @return true if the function has args, false otherwise
     */
    public boolean hasArgs(DifferentialFunction function) {
        List<String> vertexIdArgs = ops.get(function.getOwnName()).getInputsToOp();
        return vertexIdArgs != null && vertexIdArgs.size() > 0;
    }

    /**
     * Get an array of differential functions that have been defined for this SameDiff instance
     * @return Array of differential functions
     */
    public DifferentialFunction[] functions() {
        List<DifferentialFunction> out = new ArrayList<>(ops.size());
        for(SameDiffOp op : ops.values()){
            out.add(op.getOp());
        }
        return out.toArray(new DifferentialFunction[out.size()]);
    }


    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (variables != null ? variables.hashCode() : 0);
        return result;
    }


    /**
     * Create a new SameDiff instance from an existing instance.
     * Note that state (variables and functions) is shared between the two SameDiff instance
     *
     * @param originalSameDiff Original SameDiff instance
     * @return Copy
     */
    public static SameDiff create(SameDiff originalSameDiff) {
        SameDiff ret = SameDiff.builder()
                .sameDiffFunctionInstances(originalSameDiff.sameDiffFunctionInstances)
                .build();
        ret.variables.putAll(originalSameDiff.variables);
        //ensuring proper sameDiff reference
        DifferentialFunctionFactory differentialFunctionFactory = new DifferentialFunctionFactory(ret);
        ret.functionFactory = differentialFunctionFactory;
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SameDiff sameDiff = (SameDiff) o;

        if (variables != null ? !variables.equals(sameDiff.variables) : sameDiff.variables != null)
            return false;
        if (sameDiffFunctionDefinitionMap != null ? !sameDiffFunctionDefinitionMap.equals(sameDiff.sameDiffFunctionDefinitionMap) : sameDiff.sameDiffFunctionDefinitionMap != null)
            return false;
        return sameDiffFunctionInstances != null ? sameDiffFunctionInstances.equals(sameDiff.sameDiffFunctionInstances) : sameDiff.sameDiffFunctionInstances == null;
    }

    /**
     * Create a new (empty) SameDiff instance without any functions or variables
     * @return New SameDiff instance
     */
    public static SameDiff create() {
        return new SameDiff();
    }

    /**
     * Clone/duplicate the SameDiff instance, including arrays etc. The returned SameDiff instance should have no
     * shared state with the original instance
     * @return The cloned SameDiff instance
     */
    public SameDiff dup() {
        Cloner cloner = newCloner();
        SameDiff clone = cloner.deepClone(this);
        //TODO don't clone sessions in the first place!
        clone.sessions.clear();
        return clone;
    }


    /**
     * Count the number of elements in all arrays, according to {@link SDVariable#getShape()}
     * @return Number of array elements for all variables
     */
    public long numElements() {
        long ret = 0;
        for (SDVariable variable : variables()) {
            long[] shape = variable.getShape();
            if(shape != null) {
                ret += ArrayUtil.prod(shape);
            }
        }
        return ret;
    }

    /**
     * Returns the inputs (placeholders)
     * for the samediff graph
     * @return the inputs for this graph
     */
    public List<String> inputs() {
        List<String> out = new ArrayList<>();
        for(String s : variables.keySet()){
            if(isPlaceHolder(s))
                out.add(s);
        }
        return out;
    }

    /**
     * Outputs are those variables (not placeholders, constants, etc) that are the output of a function that aren't the
     * input to any other ops.
     * Usually these are the output of the last function(s) in the SameDiff instance.
     * @return The (inferred) outputs of the SameDiff instance, in no particular order
     */
    public List<String> outputs(){
        List<String> out = new ArrayList<>();
        for(Variable v : variables.values()){
            if(v.getVariable().isConstant() || v.getVariable().isPlaceHolder() ||                   //Exclude constants and placeholders
                    (v.getInputsForOp() != null && !v.getInputsForOp().isEmpty()) ||                //Exclude variables that are inputs to ops
                    (v.getControlDepsForOp() != null && !v.getControlDepsForOp().isEmpty()) ||      //Exclude variables that are control dependency inputs to ops
                    (v.getControlDepsForVar() != null && !v.getControlDepsForVar().isEmpty())) {    //Exclude variables that are control dependency inputs to other variables (mainly for import of cond etc ops)
                continue;
            }

            //Also exclude assert etc ops - doesn't make sense to return these "outputs" to user
            if(v.getOutputOfOp() != null){
                String opName = v.getOutputOfOp();
                SameDiffOp o = ops.get(opName);
                if(o.getOp() instanceof Assert){
                    continue;
                }

                //A bit of a hack for TF import: some TF graphs have Switch ops, where the output of one branch isn't consumed
                // by any ops. Consequently, during execution this "output" might never be available. So we'll exclude the output of execution here
                if(o.getOp() instanceof Switch){
                    continue;
                }
            }


            out.add(v.getName());
        }
        return out;
    }

    /**
     * The list of all variables in the graph
     *
     * @return All variables in the graph
     */
    public List<SDVariable> variables() {
        return new ArrayList<>(variableMap().values());
    }

    /**
     * Get the names of variables (if any) that have been marked as loss variables to be minimized.<br>
     * Variables can be marked as loss variables in a few different ways:<br>
     * (a) Losses are automatically added when creating loss functions via {@link #sd()}<br>
     * (b) Via {@link #setLossVariables(String...)}, @link #addLossVariable(String)} or {@link SDVariable#markAsLoss()}<br>
     * (c) Via {@link TrainingConfig#setLossVariables(List)}<br>
     */
    public List<String> getLossVariables(){
        return this.lossVariables;
    }

    /**
     * Clear/remove any existing loss variables, and set the loss variables to the specified variable names.<br>
     * See {@link #addLossVariable(String)} for more details
     * @param lossVariableNames Names of variables to be loss function variables
     */
    public void setLossVariables(String... lossVariableNames){
        this.lossVariables.clear();
        for(String s : lossVariableNames){
            addLossVariable(s);
        }
    }

    /**
     * Mark the specified variable as a loss function variable. This means that this variable will be minimized via backprop during training.<br>
     * This will add the variable as a loss to any others - i.e., if multiple variables are marked as losses, their values will be summed
     * to give the total network loss.<br>
     * Note that only floating point (Float16/32/64) variables may be marked as a loss.<br>
     * Note also that only ARRAY type SDVariables can be marked as losses to be minimized. That is, we cannot mark the value
     * of a constant, variable or placeholder to be minimized as doing so would not make sense.<br>
     */
    public void addLossVariable(@NonNull String variableName){
        Preconditions.checkState(hasVariable(variableName), "No variable with name \"%s\" exists", variableName);
        SDVariable v = getVariable(variableName);
        Preconditions.checkState(v.dataType().isFPType(), "Only floating point type variables can be marked as losses to be minimized." +
                " SDVariable \"%s\" has datatype %s", variableName, v.dataType());
        Preconditions.checkState(v.getVariableType() == VariableType.ARRAY, "Only ARRAY type SDVariables can be marked as losses to be minimized." +
                " SDVariable \"%s\" has variable type %s", variableName, v.getVariableType());
        if(!lossVariables.contains(variableName)){
            lossVariables.add(variableName);
        }
    }

    /**
     * Set the training configuration ({@link TrainingConfig}) for the SameDiff instance.
     * A TrainingConfig must be set before the SameDiff instance can be trained via the fit methods
     * @param trainingConfig Training configuration
     */
    public void setTrainingConfig(TrainingConfig trainingConfig){
        this.trainingConfig = trainingConfig;
    }

    /**
     * Fit the SameDiff instance based on a single DataSet (i.e., a single minibatch for one iteration).<br>
     * This method can only be used for singe input, single output SameDiff instances as DataSet only supports a
     * single input and a single output.<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param dataSet The DataSet (single minibatch) to peform training on
     */
    public void fit(DataSet dataSet){
        fit(new SingletonMultiDataSetIterator(dataSet.toMultiDataSet()), 1, false);
    }

    /**
     * Fit the SameDiff instance based on a single MultiDataSet (i.e., a single minibatch for one iteration).<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param dataSet The DataSet (single minibatch) to peform training on
     */
    public void fit(MultiDataSet dataSet){
        fit(new SingletonMultiDataSetIterator(dataSet), 1, false);
    }

    /**
     * Fit the SameDiff instance based on DataSetIterator for the specified number of epochs.<br>
     * This method can only be used for singe input, single output SameDiff instances as DataSet only supports a
     * single input and a single output.<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param iter      The iterator to train the SameDiff instance with
     * @param numEpochs The number of epochs for training. Must be > 0
     */
    public void fit(DataSetIterator iter, int numEpochs) {
        fit(new MultiDataSetIteratorAdapter(iter), numEpochs, true);
    }

    /**
     * Fit the SameDiff instance based on MultiDataSetIterator for the specified number of epochs.<br>
     * This method can both singe input, single output and multi-input, multi-output SameDiff instances<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param iter      The iterator to train the SameDiff instance with
     * @param numEpochs The number of epochs for training. Must be > 0
     */
    public void fit(MultiDataSetIterator iter, int numEpochs){
        fit(iter, numEpochs, true);
    }

    //Synchronized for thread safety
    protected synchronized void fit(MultiDataSetIterator iter, int numEpochs, boolean incrementEpochCount){
        Preconditions.checkNotNull(iter, "Iterator must not be null");
        Preconditions.checkState(numEpochs > 0, "Number of training epochs must be a positive number. Got: %s", numEpochs);
        Preconditions.checkState(trainingConfig != null, "No training configuration has been set. A training configuration must " +
                "be set before training. Use setTrainingConfig(TrainingConfig)");
        Preconditions.checkState(numEpochs == 1 || iter.resetSupported(), "Cannot train for multiple epochs on an iterator that" +
                " does not support resetting");

        if(!iter.hasNext() && iter.resetSupported())
            iter.reset();

        boolean performedValidation = false;

        for(int i = 0; i < numEpochs; i++) {
            while (iter.hasNext()) {
                org.nd4j.linalg.dataset.api.MultiDataSet ds = iter.next();
                if(!performedValidation){
                    Preconditions.checkState(trainingConfig.getDataSetFeatureMapping().size() == ds.numFeatureArrays(),
                            "The number of dataset feature mapping variables set in the training configuration (%s) must match" +
                                    " the number of dataset feature arrays (%s)", trainingConfig.getDataSetFeatureMapping().size(), ds.numFeatureArrays());
                    List<String> labelMapping = trainingConfig.getDataSetLabelMapping();
                    int lblSize = labelMapping == null ? 0 : labelMapping.size();
                    Preconditions.checkState(lblSize == ds.numLabelsArrays(),
                            "The number of dataset label mapping variables set in the training configuration (%s) must match" +
                                    " the number of dataset label arrays (%s)", lblSize, ds.numLabelsArrays());

                    performedValidation = true;
                }

                //Create placeholder variable map
                Map<String, INDArray> placeholders = toPlaceholderMap(ds);

                Preconditions.checkState(placeholders.size() > 0, "No placeholder variables were set for training");
                resolveVariablesWith(placeholders);

                //Calculate gradients:
                execBackwards(placeholders);


                //Apply updater:
                if (!initializedTraining)
                    initializeTraining();

                int iteration = trainingConfig.getIterationCount();
                int e = trainingConfig.getEpochCount();
                for (String s : trainingConfig.getTrainableParams()) {
                    //TODO fix using inference session
                    INDArray param = variables.get(s).getVariable().getArr();
                    INDArray grad = variables.get(s).getVariable().getGradient().getArr();
                    //Note: don't need to divide by minibatch - that should be handled in loss function and hence loss function gradients,
                    // which should flow through to here

                    //Pre-apply regularization (L1, L2)
                    List<Regularization> r = trainingConfig.getRegularization();
                    int iterCount = trainingConfig.getIterationCount();
                    int epochCount = trainingConfig.getEpochCount();
                    double lr = trainingConfig.getUpdater().hasLearningRate() ? trainingConfig.getUpdater().getLearningRate(iteration, epochCount) : 1.0;
                    if(r != null && r.size() > 0){
                        for(Regularization reg : r){
                            if(reg.applyStep() == Regularization.ApplyStep.BEFORE_UPDATER){
                                reg.apply(param, grad, lr, iterCount, epochCount);
                            }
                        }
                    }

                    //Apply updater. Note that we need to reshape to [1,length] for updater
                    INDArray reshapedView = Shape.newShapeNoCopy(grad, new long[]{1, grad.length()}, grad.ordering() == 'f');       //TODO make sure we always reshape in same order!
                    Preconditions.checkState(reshapedView != null, "Error reshaping array for parameter \"%s\": array is a view?", s);
                    GradientUpdater u = updaterMap.get(s);
                    try {
                        u.applyUpdater(reshapedView, iteration, e);
                    } catch (Throwable t) {
                        throw new RuntimeException("Error applying updater " + u.getClass().getSimpleName() + " to parameter \"" + s
                                + "\": either parameter size is inconsistent between iterations, or \"" + s + "\" should not be a trainable parameter?", t);
                    }

                    //Post-apply regularization (weight decay)
                    if(r != null && r.size() > 0){
                        for(Regularization reg : r){
                            if(reg.applyStep() == Regularization.ApplyStep.POST_UPDATER){
                                reg.apply(param, grad, lr, iterCount, epochCount);
                            }
                        }
                    }

                    if (trainingConfig.isMinimize()) {
                        param.subi(grad);
                    } else {
                        param.addi(grad);
                    }
                }

                trainingConfig.incrementIterationCount();
            }

            if(i < numEpochs - 1) {
                iter.reset();
            }

            if(incrementEpochCount)
                trainingConfig.incrementEpochCount();
        }
    }

    /**
     * Calculate the regularization (L1, L2 and/or WeightDecay) component of the loss function for the current parameters..
     * Note that the training configuration must be set (via {@link #setTrainingConfig(TrainingConfig)}) before this
     * method can be called
     *
     * @return The regularization component of the score/loss function
     */
    public double calcRegularizationScore() {
        Preconditions.checkState(trainingConfig != null, "No training configuration has been set. A training configuration must " +
                "be set before calculating the L2 loss. Use setTrainingConfig(TrainingConfig)");

        if(trainingConfig.getRegularization() == null || trainingConfig.getRegularization().isEmpty()){
            return 0.0;
        }

        if(trainingConfig.getTrainableParams() == null || trainingConfig.getTrainableParams().isEmpty())
            initializeTraining();

        List<Regularization> l = trainingConfig.getRegularization();
        double loss = 0.0;
        for (String s : trainingConfig.getTrainableParams()) {
            for(Regularization r : l){
                INDArray arr = getVariable(s).getArr();
                loss += r.score(arr, trainingConfig.getIterationCount(), trainingConfig.getEpochCount());
            }
        }
        return loss;
    }

    /**
     * Perform setup for training. Does the following:
     * 1. Infer the set of trainable parameters - unless specified manually by the user
     * 2. Set up the updaters
     */
    protected void initializeTraining(){
        if(!initializedTraining) {
            if(trainingConfig == null) {
                throw new ND4JIllegalStateException("Please specify a training config with setTrainingConfig");
            }
            //First: infer the variables to be optimized if required
            if(trainingConfig.getTrainableParams() == null || trainingConfig.getTrainableParams().size() == 0) {
                //Variable is trainable if it's not the output of some function
                //TODO also - should be floating point type
                List<String> trainVarList = new ArrayList<>();
                for(Variable var : variables.values()){
                    SDVariable v = var.getVariable();
                    String n = v.getVarName();
                    if(variables.get(n).getOutputOfOp() == null &&       //Is a leaf (not the output of a function)
                            !isPlaceHolder(n) &&                                //and not a placeholder
                            !variables.get(n).getVariable().isConstant() &&     //and not a constant
                            (trainingConfig.getDataSetFeatureMapping() == null || !trainingConfig.getDataSetFeatureMapping().contains(n))   &&  //and not an input (this really should be a placeholder, but we can't guarantee that...)
                            (trainingConfig.getDataSetLabelMapping() == null || !trainingConfig.getDataSetLabelMapping().contains(n))   &&      //and not a label (this really should be a placeholder, but we can't guarantee that...)
                            (trainingConfig.getDataSetFeatureMaskMapping() == null || !trainingConfig.getDataSetFeatureMaskMapping().contains(n))   &&  //and not a feature mask (this really should be a placeholder, but we can't guarantee that...)
                            (trainingConfig.getDataSetLabelMaskMapping() == null || !trainingConfig.getDataSetLabelMaskMapping().contains(n))){  //and not a label input (this really should be a placeholder, but we can't guarantee that...)
                        trainVarList.add(n);
                    }
                }

                trainingConfig.setTrainableParams(trainVarList);
                log.info("Inferred trainable variables: {}", trainVarList);
            }

            //Allocate updater state
            long numTrainableParams = 0;
            DataType dt = null;             //TODO support mixed precision variables - https://github.com/deeplearning4j/deeplearning4j/issues/6992
            for(String s : trainingConfig.getTrainableParams()) {
                SDVariable v = variables.get(s).getVariable();
                Preconditions.checkState(v != null, "No variable found for trainable parameter name \"%s\"", s);

                INDArray arr = v.getArr();
                Preconditions.checkState(arr != null, "No array found for trainable parameter \"%s\"", s);
                numTrainableParams += arr.length();
                if(dt == null)
                    dt = arr.dataType();
            }

            long updaterStateSize = trainingConfig.getUpdater().stateSize(numTrainableParams);

            if(updaterStateSize > 0) {
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    updaterState = Nd4j.createUninitialized(dt, 1, updaterStateSize);
                }
            }

            long viewSoFar = 0;
            updaterViews = new HashMap<>();
            updaterMap = new HashMap<>();
            for(String s : trainingConfig.getTrainableParams()) {
                long thisSize = trainingConfig.getUpdater().stateSize(variables.get(s).getVariable().getArr().length());
                INDArray view = (updaterStateSize == 0 || thisSize == 0 ? null :
                        updaterState.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(viewSoFar, viewSoFar + thisSize)));

                updaterViews.put(s, view);
                updaterMap.put(s, trainingConfig.getUpdater().instantiate(view, true));
                viewSoFar += thisSize;
            }

            initializedTraining = true;
        }
    }

    /**
     * Convert the MultiDataSet to a {@code Map<String,INDArray>} based on the TrainingConfig settings.
     * The key is the placeholder/variable that the value INDArray should be associated with.
     *
     * @param ds MultiDataSet - source of the features/labels
     * @return MultiDataSet converted to a Map, based on TrainingConfig
     */
    private Map<String,INDArray> toPlaceholderMap(org.nd4j.linalg.dataset.api.MultiDataSet ds) {
        Map<String,INDArray> placeholders = new HashMap<>();
        int count = 0;
        for(String s : trainingConfig.getDataSetFeatureMapping()){
            placeholders.put(s, ds.getFeatures(count++));
        }
        count = 0;
        if(trainingConfig.getDataSetLabelMapping() != null) {
            //Labels may be null in some models (unsupervised etc)
            for (String s : trainingConfig.getDataSetLabelMapping()) {
                placeholders.put(s, ds.getLabels(count++));
            }
        }

        if(trainingConfig.getDataSetFeatureMaskMapping() != null && trainingConfig.getDataSetFeatureMaskMapping().size() > 0){
            count = 0;
            for(String s : trainingConfig.getDataSetFeatureMaskMapping()){
                if(s == null) {
                    count++;
                    continue;
                }
                placeholders.put(s, ds.getFeaturesMaskArray(count++));
            }
        }

        if(trainingConfig.getDataSetLabelMaskMapping() != null && trainingConfig.getDataSetLabelMaskMapping().size() > 0){
            count = 0;
            for(String s : trainingConfig.getDataSetLabelMaskMapping()){
                if(s == null) {
                    count++;
                    continue;
                }
                placeholders.put(s, ds.getLabelsMaskArray(count++));
            }
        }
        return placeholders;
    }

    /**
     * Evaluate the performance of a single variable's prediction.<br>
     * For example, if the variable to evaluatate was called "softmax" you would use:
     * <pre>
     * {@code Evaluation e = new Evaluation();
     * sameDiff.evaluate(iterator, "softmax", e);}
     * </pre>
     *
     * @param iterator       Iterator as source of data to evaluate
     * @param outputVariable The variable to evaluate
     * @param evaluations    The evaluations to perform
     */
    public void evaluate(DataSetIterator iterator, String outputVariable, IEvaluation... evaluations) {
        Preconditions.checkArgument(evaluations != null && evaluations.length > 0, "No evaluations were passed to the evaluate method");
        evaluate(new MultiDataSetIteratorAdapter(iterator), Collections.singletonMap(outputVariable, Arrays.asList(evaluations)),
                Collections.singletonMap(outputVariable, 0));
    }

    /**
     * Evaluation for multiple-output networks.<br>
     * See {@link #evaluate(MultiDataSetIterator, Map, Map)}
     */
    public void evaluate(DataSetIterator iterator, Map<String,IEvaluation> variableEvals){
        Map<String,Integer> map = new HashMap<>();
        Map<String,List<IEvaluation>> variableEvalsList = new HashMap<>();
        for(String s : variableEvals.keySet()){
            map.put(s, 0);  //Only 1 possible output here with DataSetIterator
            variableEvalsList.put(s, Collections.singletonList(variableEvals.get(s)));
        }
        evaluate(new MultiDataSetIteratorAdapter(iterator), variableEvalsList, map);
    }

    /**
     * Evaluation for multiple output networks - one ore more
     * See {@link #evaluate(MultiDataSetIterator, Map, Map)}
     */
    public void evaluateMultiple(DataSetIterator iterator, Map<String,List<IEvaluation>> variableEvals){
        Map<String,Integer> map = new HashMap<>();
        for(String s : variableEvals.keySet()){
            map.put(s, 0);  //Only 1 possible output here with DataSetIterator
        }
        evaluate(new MultiDataSetIteratorAdapter(iterator), variableEvals, map);
    }

    /**
     * Evaluate the performance of a single variable's prediction.<br>
     * For example, if the variable to evaluatate was called "softmax" you would use:
     * <pre>
     * {@code Evaluation e = new Evaluation();
     * sameDiff.evaluate(iterator, "softmax", e);}
     * </pre>
     *
     * @param iterator       Iterator as source of data to evaluate
     * @param outputVariable The variable to evaluate
     * @param labelIndex     The index of the target variable's labels in the iterator
     * @param evaluations    The evaluations to perform
     */
    public void evaluate(MultiDataSetIterator iterator, String outputVariable, int labelIndex, IEvaluation... evaluations) {
        Preconditions.checkArgument(evaluations != null && evaluations.length > 0, "No evaluations were passed to the evaluate method");
        evaluate(iterator, Collections.singletonMap(outputVariable, Arrays.asList(evaluations)),
                Collections.singletonMap(outputVariable, labelIndex));
    }

    /**
     * Perform evaluation using classes such as {@link org.nd4j.evaluation.classification.Evaluation} for classifier outputs
     * and {@link org.nd4j.evaluation.regression.RegressionEvaluation} for regression outputs.<br>
     * <br>
     * <b>Example: classifier evaluation</b><br>
     * Predictions variable name: "softmaxOutput"<br>
     * Evaluations to perform: {@link org.nd4j.evaluation.classification.Evaluation}<br>
     * Data: single input, single output MultiDataSets<br>
     * Code:<br>
     * <pre>
     * {@code
     * MultiDataSetIterator data = ...
     * Map<String,List<IEvaluation>> evals = Collections.singletonMap("softmaxOutput",Collections.singletonList(new Evaluation()));
     * Map<String,Integer> labelMapping = Collections.singletonMap("softmaxOutput",0);  //Compare: "softmaxOutput" vs. MultiDataSet.getLabels(0)
     * }
     * </pre>
     *
     * @param iterator               The iterator - the source of the data for evaluation
     * @param variableEvals          The evaluations to perform. Key: the name of the variable. Value: the evaluations to perform
     * @param predictionLabelMapping The output/label mapping. Key: the name of the variable.
     */
    public void evaluate(MultiDataSetIterator iterator, Map<String,List<IEvaluation>> variableEvals, Map<String,Integer> predictionLabelMapping){
        Preconditions.checkState(trainingConfig != null, "Training config has not been set");

        Preconditions.checkState(variableEvals.keySet().equals(predictionLabelMapping.keySet()), "Keysets for variable evaluations" +
                " and for the prediction label mapping must be equal. Keys for variables to evaluate: %s vs. keys for label mapping: %s", variableEvals.keySet(), predictionLabelMapping.keySet());

        if(!iterator.hasNext() && iterator.resetSupported())
            iterator.reset();

        List<String> reqVars = new ArrayList<>(variableEvals.keySet());

        while(iterator.hasNext()){
            MultiDataSet ds = iterator.next();
            Map<String,INDArray> placeholderMap = toPlaceholderMap(ds);

            Map<String,INDArray> m = exec(placeholderMap, reqVars);

            for(Map.Entry<String,List<IEvaluation>> e : variableEvals.entrySet()){
                INDArray prediction = m.get(e.getKey());
                for(IEvaluation eval : e.getValue()){
                    //TODO masking, time series, etc

                    INDArray label = ds.getLabels(predictionLabelMapping.get(e.getKey()));
                    eval.eval(label, prediction);
                }
            }
        }
    }

    /**
     * Do inference on a network with a single input.<br>
     * For example, if the variable to infer was called "softmax" you would use:
     * <pre>
     * {@code
     * sameDiff.output(iterator, "softmax");}
     * </pre>
     *
     * @param dataSet        The data to evaluate
     * @param outputs        The variables to evaluate
     */
    public Map<String, INDArray> output(DataSet dataSet, String... outputs){
        return output(new SingletonMultiDataSetIterator(dataSet.toMultiDataSet()), outputs).get(0);
    }

    /**
     * Do inference on a network with a single input.<br>
     * For example, if the variable to infer was called "softmax" you would use:
     * <pre>
     * {@code
     * sameDiff.output(iterator, "softmax");}
     * </pre>
     *
     * @param iterator       Iterator as source of data to evaluate
     * @param outputs        The variables to evaluate
     */
    public List<Map<String, INDArray>> output(DataSetIterator iterator, String... outputs){
        return output(new MultiDataSetIteratorAdapter(iterator), outputs);
    }

    /**
     * Perform inference.<br>
     * <br>
     * <b>Example: classifier inference</b><br>
     * Predictions variable name: "softmaxOutput"<br>
     * Evaluations to perform: {@link org.nd4j.evaluation.classification.Evaluation}<br>
     * Data: single output MultiDataSets<br>
     * Code:<br>
     * <pre>
     * {@code
     * MultiDataSetIterator data = ...
     * sameDiff.output(iterator, "softmaxOutput);
     * }
     * </pre>
     *
     * @param iterator  The iterator - the source of the data for inference
     * @param outputs   The set of outputs to report.  If null, defaults to all outputs of this SameDiff.
     */
    public List<Map<String, INDArray>> output(MultiDataSetIterator iterator, String... outputs){
        Preconditions.checkState(trainingConfig != null, "Training config has not been set");

        List<String> reqVars;

        if(outputs != null){
            reqVars = Arrays.asList(outputs);
        } else {
            reqVars = outputs();
        }

        List<Map<String, INDArray>> predictions = new ArrayList<>();

        if(!iterator.hasNext() && iterator.resetSupported())
            iterator.reset();

        while(iterator.hasNext()){
            MultiDataSet ds = iterator.next();
            Map<String,INDArray> placeholderMap = toPlaceholderMap(ds);

            predictions.add(exec(placeholderMap, reqVars));
        }

        return predictions;
    }


    public SDVariable one(String name, int... shape){
        return one(name, Nd4j.defaultFloatingPointType(), shape);
    }

    public SDVariable one(String name, long... shape){
        return one(name, Nd4j.defaultFloatingPointType(), shape);
    }


    /**
     * Create a new variable with the specified shape, with all values initialized to 1.0
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return var(name, new ConstantInitScheme('f', 1.0), dataType, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 1.0
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return var(name, new ConstantInitScheme('f', 1.0), dataType, shape);
    }



    public SDVariable zero(String name, long... shape){
        return zero(name, Nd4j.defaultFloatingPointType(), shape);
    }

    public SDVariable zero(String name, int... shape){
        return zero(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 0
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return var(name, new ZeroInitScheme(), dataType, shape);
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 0
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return var(name, new ZeroInitScheme(), dataType, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create an SDVariable with a fixed/constant value, with a generated name
     * @param constant Value for the constant SDVariable
     * @return
     */
    public SDVariable constant(@NonNull INDArray constant){
        return constant(getNewVarName(), constant);
    }

    /**
     * Create an SDVariable with a fixed/constant value
     * @param name  Name of the constant SDVariable
     * @param constant Value for the constant SDVariable
     * @return
     */
    public SDVariable constant(@NonNull String name, @NonNull INDArray constant){
        Preconditions.checkState(!variables.containsKey(name), "Variable with name \"%s\" already exists", name);
        SDVariable v = new SDVariable(name, VariableType.CONSTANT, this, constant.shape(), constant.dataType(), null);
        variables.put(name, Variable.builder().name(name).variable(v).build());
        constantArrays.put(name, new DeviceLocalNDArray(constant));
        return v;
    }

    /**
     * Return a variable of given shape in which all values have a given constant value.
     *
     * @param value constant to set for each value
     * @param shape shape of the variable as long array
     * @return A new SDVariable of provided shape with constant value.
     */
    @Deprecated
    public SDVariable constant(SDVariable value, long... shape) {
        return constant(null, value, shape);
    }

    /**
     * Return a variable of given shape in which all values have a given constant value.
     *
     * @param name  Name of the new SDVariable
     * @param value constant to set for each value
     * @param shape shape of the variable as long array
     * @return A new SDVariable of provided shape with constant value.
     */
    @Deprecated
    public SDVariable constant(String name, SDVariable value, long... shape) {
        SDVariable ret = f().constant(value, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Create a variable with a place holder
     * @param name the name of the variable
     * @param shape the shape of the variable if any
     * @return
     */
    public SDVariable placeHolder(String name, org.nd4j.linalg.api.buffer.DataType dataType, long...shape) {
        SDVariable ret = new SDVariable(name, VariableType.PLACEHOLDER, this, shape, dataType, null);
        variables.put(name, Variable.builder().name(name).variable(ret).build());
        return ret;
    }

    /**
     * Variable initialization with a specified {@link WeightInitScheme}
     *
     * @param name             the name of the variable
     * @param shape            the shape of the array to be created
     * @param weightInitScheme the weight initialization scheme
     * @return the created variable
     */
    public SDVariable var(@NonNull String name, @NonNull WeightInitScheme weightInitScheme, @NonNull org.nd4j.linalg.api.buffer.DataType dataType, @NonNull long... shape) {
        return var(name, VariableType.VARIABLE, weightInitScheme, dataType, shape);
    }

    //TODO only allowing null datatype for TF import (it's fixed in a later step) - don't want this in the public API!
    public SDVariable var(@NonNull String name, @NonNull VariableType variableType, WeightInitScheme weightInitScheme,
                             org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        if (variables.containsKey(name) && variables.get(name).getVariable().getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name + " already exists.");

        if (name == null || name.length() < 1)
            name = getNewVarName();


        SDVariable ret = new SDVariable(name, variableType, this, shape, dataType, weightInitScheme);
        addVariable(ret);

        if(variableType == VariableType.PLACEHOLDER){
            setOriginalPlaceHolderShape(name, shape);
            putShapeForVarName(name, shape);
        }
        return ret;
    }

    public SDVariable var(@NonNull String name, @NonNull LongShapeDescriptor shape, WeightInitScheme weightInitScheme) {
        return var(name, weightInitScheme, shape.dataType(), shape.getShape());
    }


    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        Preconditions.checkNotNull(shape != null, "Invalid shape: shape may not be null");
        if(Shape.isPlaceholderShape(shape)){
            return placeHolder(name, dataType, shape);
        }
        return var(name, new ZeroInitScheme(), dataType, shape);
    }

    public SDVariable var(String name, LongShapeDescriptor shapeDesc) {
        Preconditions.checkNotNull(shapeDesc != null, "Invalid shape: shape may not be null");
        return var(name, shapeDesc, new ZeroInitScheme());
    }

    public SDVariable var(String name, int... shape){
        return var(name, Nd4j.defaultFloatingPointType(), shape);
    }

    public SDVariable var(String name, long... shape){
        return var(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        Preconditions.checkNotNull(shape, "Invalid shape: shape may not be null");
        if(Shape.isPlaceholderShape(shape)){
            return placeHolder(name, dataType, ArrayUtil.toLongArray(shape));
        }
        return var(name, new ZeroInitScheme(), dataType, ArrayUtil.toLongArray(shape));
    }


    /**
     * Initialize a {@link SDVariable} reference tying this variable to this samediff instance.
     * <p>
     * {@link NDArraySupplierInitScheme} is used to ensure that if the array is allocated anywhere
     * and {@link SameDiff} instance to exist as a copy of the variable.
     *
     * @param v Variable
     * @return
     */
    public SDVariable var(@NonNull final SDVariable v) {
        if (variables.containsKey(v.getVarName()) && variables.get(v.getVarName()).getVariable().getArr() != null)
            return variables.get(v.getVarName()).getVariable();

        if (v.getVarName() == null || v.getVarName().length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        VariableType vt = v.getVariableType();
        NDArraySupplierInitScheme s = null;
        switch(vt){
            case VARIABLE:
                s = new NDArraySupplierInitScheme(v.getArr());
                //Intentional fallthrough
            case ARRAY:
                SDVariable ret = new SDVariable(v.getVarName(), v.getVariableType(), this, v.getShape(), v.dataType(), s);
                return addVariable(ret);
            case CONSTANT:
                return constant(v.getVarName(), v.getArr());
            case PLACEHOLDER:
                return placeHolder(v.getVarName(), v.dataType(), v.placeholderShape());
            default:
                throw new RuntimeException("Unknown/not supported variable type: " + vt);
        }
    }

    private String getNewVarName() {
        String varName = "sd_var_" + String.valueOf(variableId);
        while (variables.containsKey(varName)) {
            variableId++;
            varName = "sd_var_" + String.valueOf(variableId);
        }
        return varName;
    }

    /**
     * Creates a {@link SDVariable} with the specified shape and a generated name<br>
     * Any array will be generated with all zeros for the values
     *
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return var(getNewVarName(), dataType, shape);
    }

    /**
     * Creates a {@link SDVariable} with the specified shape and a generated name<br>
     * Any array will be generated with all zeros for the values
     *
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return var(getNewVarName(), dataType, shape);
    }

    /**
     * Creates a {@link SDVariable} with the specified shape and a generated name. The associated array will
     * then be generated using the specified weight initialization scheme
     *
     * @param weightInitScheme The weight initialization scheme to use when generating an INDArray
     * @param shape            the shape of the variable
     * @return the created variable
     */
    public SDVariable var(WeightInitScheme weightInitScheme, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return var(getNewVarName(), weightInitScheme, dataType, shape);
    }

    /**
     * Create an {@link SDVariable} with a generated name, and assocate the specified array with it
     * @param arr Array to associate with the new variable
     * @return New SDVariable
     * @see #var(String, INDArray)
     */
    public SDVariable var(INDArray arr) {
        return var(getNewVarName(), arr);
    }

    /**
     * Create an {@link SDVariable} with the specified name, and assocate the specified array with it
     * @param arr Array to associate with the new variable
     * @return New SDVariable with the specified name and array
     */
    public SDVariable var(String name, INDArray arr) {
        if (variables.containsKey(name) && variables.get(name).getVariable().getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name + " already exists.");


        if (name == null || name.length() < 1)
            name = getNewVarName();

        if (arr == null)
            throw new IllegalArgumentException("Array for " + name + " must not be null");

        boolean duped = false;
        if(arr.isAttached()) {
            arr = arr.detach();
            duped = true;
        }
        if(arr.isView()) {
            arr = arr.dup();
            duped = true;
        }

        if(!duped) {
            for (DeviceLocalNDArray otherArr : variablesArrays.values()) {
                if (otherArr.get() == arr) {    //Check for exact same object, to avoid array reuse (can result in unexpected behaviour)
                    arr = arr.dup();
                    break;
                }
            }
        }

        SDVariable ret = new SDVariable(name, VariableType.VARIABLE, this, arr.shape(), arr.dataType(), new NDArraySupplierInitScheme(arr));

        associateArrayWithVariable(arr, ret);
        if (ArrayUtil.prod(arr.shape()) == 1) {
            try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                ret.setScalarValue(Nd4j.scalar(arr.getDouble(0)));
            }
        }

        addVariable(ret);
        if (getShapeForVarName(name) == null)
            putShapeForVarName(name, arr.shape());
        return ret;
    }

    /**
     * Convert the specified variable to a constant. This is equivalent to "freezing" a variable so that it's value
     * won't be changed by further training.<br>
     * This can only be done for variables and placeholders, not ARRAY type variables (which are usually network activations).
     * As a constant, this variable will no longer be modified by any subsequent training.
     *
     * @param variable Variable to convert to a constant
     * @return The (now constant) SDVariable
     */
    public SDVariable convertToConstant(@NonNull SDVariable variable) {
        convertToConstants(Collections.singletonList(variable));
        return variable;
    }

    /**
     * Convert all of the specified variables to constants. This is equivalent to "freezing" the variables so that their values
     * won't be changed by further training.<br>
     * This can only be done for variables and placeholders, not ARRAY type variables (which are usually network activations).
     * As constants, these variables will no longer be modified by any subsequent training.
     *
     * @param variables Variables to convert to constants
     * @return The (now constant) SDVariables
     */
    public void convertToConstants(List<SDVariable> variables){
        if(variables.size() == 0)
            return;
        boolean allConst = true;
        for(SDVariable variable : variables) {
            if (variable.getVariableType() != VariableType.CONSTANT) {
                allConst = false;
                Preconditions.checkState(variable.getVariableType() != VariableType.ARRAY, "Cannot convert variable of type ARRAY to a constant: %s", variable);
            }
        }
        if(allConst){
            return; //No op
        }

        //Remove all sessions in case they have any cached arrays/state
        sessions.clear();

        //If gradient function has been defined, remove it (so it will be recreated later)
        sameDiffFunctionInstances.remove("grad");

        for(SDVariable variable : variables ) {
            String n = variable.getVarName();
            INDArray arr = variable.getArr();
            Preconditions.checkNotNull(arr, "Could not get array for variable %s: if this is a placeholder, use SDVariable.setArray before converting", variable);

            constantArrays.put(n, new DeviceLocalNDArray(arr));
            variablesArrays.remove(n);
            if(!placeholdersPerThread.isEmpty()){
                for(Map<String,INDArray> m : placeholdersPerThread.values()){
                    m.remove(n);
                }
            }

            variable.setVariableType(VariableType.CONSTANT);
        }


        if(trainingConfig != null){
            Set<String> toRemove = new HashSet<>();
            boolean anyTrainableParmsModified = false;
            List<String> origTrainableParams = trainingConfig.getTrainableParams();
            for(SDVariable v : variables){
                toRemove.add(v.getVarName());
                if(!anyTrainableParmsModified && origTrainableParams.contains(v.getVarName())){
                    anyTrainableParmsModified = true;
                }
            }


            //Remove updater state for this variable: updaterState, updaterViews, updaterMap
            if(anyTrainableParmsModified) {
                List<String> newTrainableParams = new ArrayList<>();
                for (String s : origTrainableParams) {
                    if (!toRemove.contains(s)) {
                        newTrainableParams.add(s);
                    }
                }
                trainingConfig.setTrainableParams(newTrainableParams);
            }

            if(initializedTraining){
                List<INDArray> newUpdaterState = new ArrayList<>();
                for (String s : origTrainableParams) {
                    INDArray stateArr = updaterViews.get(s);
                    if (!toRemove.contains(s)) {
                        newUpdaterState.add(stateArr);
                    }
                }

                updaterState = newUpdaterState.isEmpty() ? null : Nd4j.concat(0, newUpdaterState.toArray(new INDArray[newUpdaterState.size()]));
                //Now, update updaterViews map:
                long viewSoFar = 0;
                updaterViews = new HashMap<>();
                updaterMap = new HashMap<>();
                for(String s : trainingConfig.getTrainableParams()) {
                    long thisSize = trainingConfig.getUpdater().stateSize(this.variables.get(s).getVariable().getArr().length());
                    INDArray view = (updaterState == null || thisSize == 0 ? null :
                            updaterState.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(viewSoFar, viewSoFar + thisSize)));

                    updaterViews.put(s, view);
                    updaterMap.put(s, trainingConfig.getUpdater().instantiate(view, false));
                    viewSoFar += thisSize;
                }
            }
        }
    }

    /**
     * Convert the specified variable to a VARIABLE type SDVariable.<br>
     * This can only be done for constants and placeholders, not ARRAY type variables (which are usually network activations).
     * As a variable, this variable will modified during any subsequent training.
     *
     * @return This variable (now a variable type SDVariable)
     */
    public SDVariable convertToVariable(@NonNull SDVariable constant) {
        convertToVariables(Collections.singletonList(constant));
        return constant;
    }

    /**
     * Convert the specified variables to VARIABLE type SDVariables.<br>
     * This can only be done for constants and placeholders, not ARRAY type variables (which are usually network activations).
     * As variables, this variable will modified during any subsequent training.
     */
    public void convertToVariables(@NonNull List<SDVariable> constants){
        if(constants.size() == 0)
            return;
        boolean allConst = true;
        for(SDVariable variable : constants) {
            if (variable.getVariableType() != VariableType.VARIABLE) {
                allConst = false;
            }
            Preconditions.checkState(variable.getVariableType() != VariableType.ARRAY, "Cannot convert variable of type ARRAY to a variable: %s", variable);
        }
        if(allConst){
            return; //No op
        }

        //Remove all sessions in case they have any cached arrays/state
        sessions.clear();

        //If gradient function has been defined, remove it (so it will be recreated later)
        sameDiffFunctionInstances.remove("grad");

        for(SDVariable variable : constants) {
            String n = variable.getVarName();
            INDArray arr = variable.getArr();
            Preconditions.checkNotNull(arr, "Could not get array for variable %s: if this is a placeholder, use SDVariable.setArray before converting", variable);

            variablesArrays.put(n, new DeviceLocalNDArray(arr));
            constantArrays.remove(n);
            if(!placeholdersPerThread.isEmpty()){
                for(Map<String,INDArray> m : placeholdersPerThread.values()){
                    m.remove(n);
                }
            }

            variable.setVariableType(VariableType.VARIABLE);
        }


        //For training: need to add new updater state
        if(trainingConfig != null){
            List<String> newTrainableParams = new ArrayList<>(trainingConfig.getTrainableParams());
            List<String> convertedToVars = new ArrayList<>();
            for(SDVariable v : constants){
                newTrainableParams.add(v.getVarName());
                convertedToVars.add(v.getVarName());
            }
            trainingConfig.setTrainableParams(newTrainableParams);


            //Add updater state for this variable: updaterState, updaterViews, updaterMap
            if(initializedTraining){
                long extraStateSize = 0;
                for (String s : convertedToVars) {
                    INDArray arr = getVariable(s).getArr();
                    long stateSize = trainingConfig.getUpdater().stateSize(arr.length());
                    extraStateSize += stateSize;
                }
                if(extraStateSize > 0) {
                    INDArray newState = Nd4j.createUninitialized(updaterState.dataType(), 1, extraStateSize);

                    updaterState = (updaterState == null ? newState : Nd4j.concat(1, updaterState, newState));
                    //Now, update updaterViews map:
                    long viewSoFar = 0;
                    updaterViews = new HashMap<>();
                    updaterMap = new HashMap<>();
                    for (String s : trainingConfig.getTrainableParams()) {
                        long thisSize = trainingConfig.getUpdater().stateSize(this.variables.get(s).getVariable().getArr().length());
                        INDArray view = (updaterState == null || thisSize == 0 ? null :
                                updaterState.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(viewSoFar, viewSoFar + thisSize)));

                        updaterViews.put(s, view);
                        boolean init = convertedToVars.contains(s); //Only initialize/zero the states for the new variables
                        updaterMap.put(s, trainingConfig.getUpdater().instantiate(view, init));
                        viewSoFar += thisSize;
                    }
                }
            }
        }
    }


    /**
     * Remove an argument for a function. Note that if this function does not contain the argument, it will just be a no op.
     *
     * @param varName  the variable name to remove
     * @param function the function to remove the argument from
     */
    public void removeArgFromFunction(String varName, DifferentialFunction function) {
        val args = function.args();

        for (int i = 0; i < args.length; i++) {
            if (args[i].getVarName().equals(varName)) {
                /**
                 * Since we are removing the variable reference
                 * from the arguments we need to  update both
                 * the reverse and forward arguments.
                 */
                List<String> reverseArgs = ops.get(function.getOwnName()).getInputsToOp();
                val newArgs = new ArrayList<String>(args.length - 1);
                for (int arg = 0; arg < args.length; arg++) {
                    if (!reverseArgs.get(arg).equals(varName)) {
                        newArgs.add(reverseArgs.get(arg));
                    }
                }

                ops.get(function.getOwnName()).setInputsToOp(newArgs);
                break;
            }
        }
    }

    /**
     * Get the variable based on the opName
     *
     * @param name the opName of the variable
     * @return the variabel instance if there is one
     */
    public SDVariable getVariable(String name) {
        Variable v = variables.get(name);
        return v == null ? null : v.getVariable();
    }

    public boolean hasVariable(String name){
        return variables.containsKey(name);
    }


    /**
     * Get the gradient for the variable with the specified name.<br>
     * The gradient variable is the variable that represents the derivative of the loss function with respect
     * to the output of this variable. I.e., if this variable is X and loss function is L, then gradient() returns the
     * variable representing dL/dX<br>
     * Note that only floating point variables can have gradients.<br>
     * Note also that a gradient may not yet be defined, and/or if no loss function variables have been set.<br>
     * You can set the loss function variables using {@link SameDiff#setLossVariables(String...)} and then create the
     * gradient functions using {@link SameDiff#createGradFunction()}. Alternatively, the gradient function will be
     * created automatically when training is performed.
     *
     * @param varName the vertex id
     * @return the gradient for this variable or null
     */
    public SDVariable getGradForVariable(String varName) {
        Preconditions.checkState(variables.containsKey(varName), "No variable with name \"%s\" exists", varName);
        SDVariable v = getVariable(varName);
        Preconditions.checkState(v.dataType().isFPType(), "Cannot get gradient of %s variable \"%s\": only floating" +
                " point variables have gradients", varName, v.dataType());
        //Gradients are being placed in the inner "grad" function SameDiff instance, but not the outer one
        if (variables.containsKey(varName) && variables.get(varName).getGradient() != null) {
            return variables.get(varName).getGradient();
        } else if(sameDiffFunctionInstances.containsKey("grad") && sameDiffFunctionInstances.get("grad").variables.containsKey(varName)){
            return sameDiffFunctionInstances.get("grad").variables.get(varName).getGradient();
        }
        return null;
    }

    /**
     * Determine if the specified variable has a gradient with respect to the current loss. Note that:
     * (a) Non-floating-point variables (integer, string, etc) will never have gradients<br>
     * (b) This method will return false if no gradient function has been created yet. See {@link SameDiff#createGradFunction()}
     * and {@link SameDiff#setLossVariables(String...)}<br>
     * (c) Floating point variables may not have any gradient if the specified loss variables does not depend on the
     * specified variable at all. In this case, "no gradient" for floating point is equivalent to "always 0"<br>
     *
     * @param varName Name of the variable to check the existence of a gradient variable for
     * @return True if a gradient variable exists for the specified variable, for the current loss
     */
    public boolean variableHasGradient(String varName){
        Preconditions.checkState(variables.containsKey(varName), "No variable with name \"%s\" exists", varName);
        SDVariable v = getVariable(varName);
        if(!v.dataType().isFPType())
            return false;

        return getGradForVariable(varName) != null;
    }


    /**
     * Assign a SDVariable to represent the gradient of the SDVariable with the specified name
     *
     * @param variableName the variable name to assign the gradient variable for
     * @param variable     the gradient variable
     */
    public void setGradientForVariableName(String variableName, SDVariable variable) {
        Preconditions.checkState(variables.containsKey(variableName), "No variable exists with name \"%s\"", variableName);
        if (variable == null) {
            throw new ND4JIllegalStateException("Unable to set null gradient for variable name " + variableName);
        }
        variables.get(variableName).setGradient(variable);
    }


    /**
     * @param varName
     * @param forwardVariable
     */
    public void setForwardVariableForVarName(String varName, SDVariable forwardVariable) {
        forwardVarForGrad.put(varName, forwardVariable);
    }

    /**
     * Get the gradient for the variable with the specified variable name.
     * Note that in order to run this function, {@link #execBackwards()} must be executed first.
     * All gradient functions are obtained from the results of the execBackwards call.
     *
     * @param varName the variable name to get the gradient variable for.
     * @return The gradient variable for the specified variable
     */
    public SDVariable grad(String varName) {
        if (!sameDiffFunctionInstances.containsKey("grad")) {
            throw new IllegalStateException("Unable to obtain gradient. Please run execBackwards() first.");
        }

        SameDiff grad = getFunction("grad");
        SDVariable var = grad.getVariable(varName);
        return getFunction("grad").getGradForVariable(var.getVarName());
    }


    /**
     * Create a new double scalar (rank 0) SDVariable with the specified value
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, double value) {
        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new float scalar (rank 0) SDVariable with the specified value
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, float value) {
        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new integer scalar (rank 0) SDVariable with the specified value
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, int value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new long scalar (rank 0) SDVariable with the specified value
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, long value) {
        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new scalar (rank 0) SDVariable with the specified value and datatype
     *
     * @param name     Name of the SDVariable
     * @param dataType Data type of the scalar
     * @param value    Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, DataType dataType, Number value) {
        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(dataType, value));
        }
    }


    /**
     * Add the specified variable to this SameDiff instance
     * @param variable Variable to add
     */
    public SDVariable addVariable(SDVariable variable) {
        Preconditions.checkState(variable.getSameDiff() == this, "Samediff instance must be the same.");

        if (variables.containsKey(variable.getVarName()) && !variables.get(variable.getVarName()).getVariable().equals(variable)) {
            throw new IllegalArgumentException("Variable already found with variable opName " + variable.getVarName());
        }

        Preconditions.checkState(variable.getSameDiff() == this, "Same diff instance for variable must be the same!");
        variables.put(variable.getVarName(), Variable.builder().name(variable.getVarName()).variable(variable).build());
        return variable;
    }


    /**
     * Generate a new variable name based on the uniqueness of the base name and arg index<br>
     * For example, if baseName = "X" will return:<br>
     * "X" if "X" does not already exist, or "X:argIndex" if argIndex > 0<br>
     * "X_1" if "X" already exists, or "X_1:argIndex" if argIndex > 0<br>
     * "X_2" if "X" and "X_1" already exists, or "X_2:argIndex" if argIndex > 0<br>
     * And so on, until an unused name is found
     *
     * @param baseName the base name to use (use function.opName() where function is a {@link DifferentialFunction}
     * @param argIndex the arg index
     * @return the new generated name
     */
    public String generateNewVarName(String baseName, int argIndex) {
        if (!variables.containsKey(baseName) && argIndex == 0) {
            return baseName;
        }

        //need to find a new name
        int count = 0;
        String name = baseName + (count == 0 ? "" : "_" + count) + (argIndex > 0 ? ":" + argIndex : "");
        while (getVariable(name) != null) {
            name = baseName + "_" + (++count) + (argIndex > 0 ? ":" + argIndex : "");
        }

        if (getVariable(name) != null) {
            throw new ND4JIllegalStateException("Converged on already generated variable!");
        }
        return name;
    }


    /**
     * Generate the variables based on the given input op and return the output variable names.
     *
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function, String baseName, boolean isImport) {
        //xyz ops only have 1 output
        //if there is already a base name defined, use that
        if (baseName == null || baseName.isEmpty() && getBaseNameForFunction(function) != null)
            baseName = getBaseNameForFunction(function);

        if (baseName == null)
            baseName = function.opName();

        //First: calculate output data types. We can always calculate output data types, even if the input arrays
        //are not available - *except for sometimes during import, until all ops/variables have been added*
        List<org.nd4j.linalg.api.buffer.DataType> outputDataTypes = null;

        if(!isImport) {
            List<org.nd4j.linalg.api.buffer.DataType> inputDataTypes = new ArrayList<>();
            List<String> fnInputs = ops.get(function.getOwnName()).getInputsToOp();
            if (fnInputs != null) {
                for (String var : fnInputs) {
                    inputDataTypes.add(variables.get(var).getVariable().dataType());
                }
            }
            outputDataTypes = function.calculateOutputDataTypes(inputDataTypes);
        }

        val outputShape = function.calculateOutputShape();
        if (outputShape == null || outputShape.isEmpty()) {
            if (function instanceof CustomOp) {
                CustomOp customOp = (CustomOp) function;
                //can't guess number of outputs, variable
                int num_outputs = function.getNumOutputs(); //Use this in preference - if set. Descriptor might specify 2, but it can sometimes be 2+
                if (num_outputs <= 0) {
                    val descriptor = customOp.getDescriptor();
                    if (descriptor != null) {
                        num_outputs = descriptor.getNumOutputs();
                    }
                    if (num_outputs <= 0) {
                        throw new ND4UnresolvedOutputVariables("Could not determine number of output variables for op "
                                + function.getOwnName() + " - " + function.getClass().getSimpleName() + ". Ops can override" +
                                " getNumOutputs() to specify number of outputs if required");
                    }
                }
                char ordering = 'c';
                SDVariable[] args = function.args();
                if (args != null && args.length > 0 && args[0].getArr() != null) {  //Args may be null or length 0 for some ops, like eye
                    ordering = function.args()[0].getArr().ordering();
                }
                SDVariable[] ret = new SDVariable[num_outputs];

                //Infer the output types: we can always determine datatype but not always shapes
                Preconditions.checkState(isImport || num_outputs == 0 || (outputDataTypes != null && outputDataTypes.size() == num_outputs),
                        "Incorrect number of output datatypes: got %s but expected datatypes for %s outputs - %s (op: %s)",
                        (outputDataTypes == null ? null : outputDataTypes.size()), num_outputs, outputDataTypes, function.getClass().getSimpleName());

                //dynamic shapes
                //When importing from TF: convention is "unstack", "unstack:1", "unstack:2", ...
                for (int i = 0; i < ret.length; i++) {
                    SDVariable var = (i == 0 ? getVariable(baseName) : getVariable(baseName + ":" + i));
                    if (var == null) {
                        //Generate new variable name if one with the specified name doesn't exist
                        //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme

                        org.nd4j.linalg.api.buffer.DataType dataType  = isImport ? null : outputDataTypes.get(i);
                        var = var(generateNewVarName(baseName, i), VariableType.ARRAY, null, dataType, (long[])null);
                    }
                    var.setOutputIndex(i);
                    var.setCreator(function);
                    ret[i] = var;
                }

                //Update the internal state: outgoing variables for function
                if (getOutputsForFunction(function) == null)
                    addOutgoingFor(ret, function);

                return ret;
            }

            //this is for unresolved shapes, we know xyz is always 1 output
            else if (function instanceof BaseOp && outputShape.isEmpty()) {
                SDVariable[] ret = new SDVariable[1];
                SDVariable checkGet = getVariable(baseName);
                char ordering = 'c';
                SDVariable[] args = function.args();
                if (args != null && args.length > 0 && function.args()[0].getArr() != null) { //Args may be null or length 0 for some ops, like eye
                    ordering = function.args()[0].getArr().ordering();
                }
                if (checkGet == null) {
                    //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme
                    org.nd4j.linalg.api.buffer.DataType dataType  = outputDataTypes.get(0);
                    checkGet = var(baseName, VariableType.ARRAY, null, dataType, (long[])null);
                }

                if (checkGet == null) {
                    //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme
                    org.nd4j.linalg.api.buffer.DataType dataType  = outputDataTypes.get(0);
                    checkGet = var(baseName, VariableType.ARRAY, null, dataType, (long[])null);
                }

                checkGet.setOutputIndex(0);
                checkGet.setCreator(function);
                ret[0] = checkGet;


                //Update the internal state: outgoing variables for function
                if (getOutputsForFunction(function) == null)
                    addOutgoingFor(ret, function);

                return ret;
            }
        }

        //Check that output shapes and output dtypes actually match (they should)
        if(!isImport) {
            for (int i = 0; i < outputShape.size(); i++) {
                org.nd4j.linalg.api.buffer.DataType shapeDataType = outputShape.get(i).dataType();
                org.nd4j.linalg.api.buffer.DataType calcType = outputDataTypes.get(i);
                Preconditions.checkState(calcType == shapeDataType, "Calculated output data types do not match for shape calculation vs. datatype calculation:" +
                        " %s vs %s for op %s output %s", shapeDataType, calcType, function.getClass().getName(), i);
            }
        }

        char ordering = 'c';
        if (function.args() != null && function.args().length > 0 && function.args()[0].getArr() != null) {
            ordering = function.args()[0].getArr().ordering();
        }

        SDVariable[] ret = new SDVariable[outputShape.size()];

        // ownName/baseName will be used to get variables names
        val ownName = function.getOwnName();
        val rootName = baseName;
        for (int i = 0; i < ret.length; i++) {
            LongShapeDescriptor shape = outputShape.get(i);
            // it should be: rootName:index. i.e.: split:1, split:2, split:3, split:4 etc
            baseName = rootName + (i > 0 ? ":" + i : "");
            SDVariable checkGet = getVariable(baseName);
            if (checkGet == null) {
                // obviously - there's no such var, just add it
                //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme


                checkGet = var(baseName, VariableType.ARRAY, null, shape.dataType(), shape.getShape());
            } else if (shape != null && !shapeAlreadyExistsForVarName(checkGet.getVarName())) {
                // var exists, let's update its shape
                putShapeForVarName(checkGet.getVarName(), shape);
            } else if (shape != null && shapeAlreadyExistsForVarName(checkGet.getVarName())) {
                // no-op.
                // TODO: maybe we should check shapes equality here?
                // it's either var that already exist, or something bad happening
            }

            if (checkGet == null) {
                org.nd4j.linalg.api.buffer.DataType dataType = org.nd4j.linalg.api.buffer.DataType.FLOAT;     //TODO FIX THIS
                checkGet = var(baseName + (i > 0 ? ":" + i : ""), new ZeroInitScheme(ordering), dataType, shape.getShape());
            }

            checkGet.setOutputIndex(i);
            checkGet.setCreator(function);
            ret[i] = checkGet;
        }

        return ret;
    }

    /**
     * Generate the variables based on the given input op
     * and return the output variable names.
     *
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function) {
        return generateOutputVariableForOp(function, function.opName(), false);
    }

    /**
     * Get a SameDiff function instance given the name of the function
     *
     * @param functionName the name of the function
     * @return the same diff function instance defined for the given name
     */
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }


    /**
     * Creates a while statement
     *
     * @param sameDiffConditional
     * @param loopBody
     * @return
     */
    public While whileStatement(SameDiffConditional sameDiffConditional,
                                SameDiffFunctionDefinition conditionBody,
                                SameDiffFunctionDefinition loopBody
            , SDVariable[] inputVars) {
        return While.builder()
                .inputVars(inputVars)
                .condition(conditionBody)
                .predicate(sameDiffConditional)
                .trueBody(loopBody)
                .parent(this)
                .blockName("while-" + UUID.randomUUID().toString())
                .build();
    }

    /**
     * @param conditional
     * @param trueBody
     * @param falseBody
     * @return
     */
    public If ifStatement(SameDiffConditional conditional,
                          SameDiffFunctionDefinition conditionBody,
                          SameDiffFunctionDefinition trueBody,
                          SameDiffFunctionDefinition falseBody
            , SDVariable[] inputVars) {
        return If.builder()
                .conditionBody(conditionBody)
                .falseBody(falseBody)
                .trueBody(trueBody)
                .predicate(conditional)
                .inputVars(inputVars)
                .parent(this)
                .blockName("if-" + UUID.randomUUID().toString())
                .build();
    }


    public TensorArray tensorArray(DataType dataType) {
        TensorArray ta = new TensorArray(this, dataType);
        SDVariable[] outVars = ta.outputVariables();
        return ta;
    }

    /**
     * @param functionName
     * @param with
     */

    public SDVariable invokeFunctionOn(String functionName, SameDiff with) {
        SameDiff instance = sameDiffFunctionInstances.get(functionName);
        SDVariable ret = instance.invokeGraphOn(with);

        return ret;
    }


    /**
     * @param function
     */
    public SameDiff defineFunction(String function, SameDiffFunctionDefinition functionDefinition, SDVariable[] variables) {
        if (!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            this.child = sub;
            sub.parent = this;
            //setup subgraph
            //re execute to populate subgraph
            SDVariable[] ret = new SDVariable[variables.length];
            for (int i = 0; i < ret.length; i++) {
                ret[i] = sub.var(variables[i]);
            }

            functionDefinition.define(sub, null, ret);
            sameDiffFunctionInstances.put(function, sub);
        }
        this.child = null;
        return sameDiffFunctionInstances.get(function);
    }


    /**
     * @param function
     */
    public void defineFunction(String function, SameDiffFunctionDefinition functionDefinition) {
        defineFunction(function, functionDefinition, new LinkedHashMap<String, INDArray>());
    }

    /**
     * @param function
     * @param functionDefinition
     * @param inputs
     */
    public void defineFunction(String function,
                               SameDiffFunctionDefinition functionDefinition,
                               Map<String, INDArray> inputs) {
        if (!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            //setup subgraph
            //re execute to populate subgraph
            functionDefinition.define(sub, inputs, null);

            sameDiffFunctionInstances.put(function, sub);
        }

    }

    @Deprecated
    public INDArray execAndEndResult(){
        List<String> outputs = outputs();
        Preconditions.checkState(outputs.size() == 1, "Method can only be used with SameDiff instances with a single output");
        long tid = Thread.currentThread().getId();
        Map<String,INDArray> placeholders = placeholdersPerThread.get(tid);
        return execSingle(placeholders, outputs.get(0));
    }

    /**
     * Create (if required) and then calculate the variable gradients (backward pass) for this graph.<br>
     * After execution, the gradient arrays can be accessed using {@code myVariable.getGradient().getArr()}<br>
     * <b>Note</b>: This method by default calculates VARIABLE type SDVariable gradients only (as well as any other
     * gradients needed to calculate the variable gradients). That is, placeholder, constant, etc gradients are not
     * calculated. If these gradients are required, they can be calculated using {@link #execBackwards(Map, List)} instead,
     * which allows specifying the set of SDVariables to calculate the gradients for. For example,
     * {@code execBackwards(placeholders, Arrays.asList(myPlaceholder.gradient().getVarName())}. In some cases,
     * {@link #createGradFunction()} may need to be called first
     *
     * @param placeholders Values for the placeholder variables in the graph. For graphs without placeholders, use null or an empty map
     */
    public void execBackwards(Map<String,INDArray> placeholders){
        if (getFunction("grad") == null) {
            createGradFunction();
        }

        //Collect (unique) list of gradient names...
        Set<String> varGradNames = new HashSet<>();
        for(Variable v : variables.values()){
            if(v.getVariable().getVariableType() == VariableType.VARIABLE){
                SDVariable g = v.getVariable().gradient();
                varGradNames.add(g.getVarName());
            }
        }

        //Edge case: if no variables, no variable gradients to calculate...
        if(varGradNames.isEmpty()){
            log.warn("Skipping gradient execution (backward pass) - no variables to be calculated (graph does not contain any VARIABLE type SDVariables).\n" +
                    "If gradients for other variables (such as placeholders) are required, use execBackwards(Map, List) instead");
            return;
        }

        List<String> vargradNamesList = new ArrayList<>(varGradNames);
        execBackwards(placeholders, vargradNamesList);
    }

    public void execBackwards(Map<String,INDArray> placeholders, String... variableGradNamesList){
        execBackwards(placeholders, Arrays.asList(variableGradNamesList));
    }

    /**
     * As per {@link #execBackwards(Map)}, but the set of gradients to calculate can be specified manually.<br>
     * For example, to calculate the gradient for placeholder variable "myPlaceholder", use
     * {@code execBackwards(placeholders, Arrays.asList(myPlaceholder.gradient().getVarName())}.
     *
     * @param placeholders Values for the placeholder variables in the graph. For graphs without placeholders, use null or an empty map
     * @param variableGradNamesList Names of the gradient variables to calculate
     */
    public void execBackwards(Map<String,INDArray> placeholders, List<String> variableGradNamesList){
        if (getFunction("grad") == null) {
            createGradFunction();
        }

        log.trace("About to execute backward function");

        //Edge case: if no variables, no variable gradients to calculate...
        if(variableGradNamesList.isEmpty()){
            log.warn("Skipping gradient calculation (backward pass) - no variables to be calculated (variableGradNamesList is empty)");
            return;
        }

        sameDiffFunctionInstances.get("grad").exec(placeholders, variableGradNamesList);
    }

    /**
     * Create the gradient function (for calculating gradients via {@link #execBackwards(Map)}) if it is not already defined.
     * Users do not usually need to call this function manually, as it is called as required in the aforementioned method.
     * <br><br>
     * If the gradient function already exists, this method is a no-op.<br>
     * After this method returns, the SameDiff function instance for the gradient can be accessed using {@link #getFunction(String)}
     * with name "grad" as the argument.
     */
    public void createGradFunction() {
        if(lossVariables.isEmpty()){
            if(trainingConfig != null && trainingConfig.getLossVariables() != null && !trainingConfig.getLossVariables().isEmpty()){
                lossVariables.addAll(trainingConfig.getLossVariables());
            } else {
                List<String> outputs = outputs();
                if (outputs.size() == 1) {
                    log.info("Inferring output \"{}\" as loss variable as none were previously set. Use SameDiff.setLossVariables() to override", outputs.get(0));
                    lossVariables.add(outputs.get(0));
                }
            }
        }

        Preconditions.checkState(!lossVariables.isEmpty(), "Cannot create gradient function: " +
                "No loss variables (variables to minimize) have been specified. Loss variables are the variables that" +
                " represent the loss/cost/score to be minimized during training, and that all gradients are calculated with respect to.\n" +
                " Losses can be specified either in TrainingConfiguration (Builder.minimize(...)) or via SameDiff.setLossVariables()/addLossVariable()");

        if (log.isTraceEnabled()) {
            log.trace("Defining function \"grad\"");
        }


        /*
        Defining gradient function:

        Starting point:
        (a) Set of loss function variables - i.e., one or more variables representing loss to be minimized
        (b) Set of floating point variables we want to train (Variable type SDVariables only - not constants, arrays, placeholders)

        Observation: A trainable parameter only has a gradient defined if there is a floating point path between the variable and the loss.
        for example: X(fp) -> cast(int) -> cast(fp) -> loss - X has no gradient defined

        Algorithm for backprop:

        Step 1: Determine if variable requires a gradient (is trainable)
        How? Walk backward on op graph starting at loss variable(s), along FP variables only.
        Collect FP variables in set as we go.
        This gives us a subgraph "connected to loss by FP path" - gradient for FP variable is defined only if it's in that set/subgraph.


        Step 2: Determine minimal set of variables (including array type SDVariables - i.e., activations) we need gradients for
        Consider following graph: X(fp) -> cast(int) -> cast(fp) -> lots of FP ops -> loss
        unless we need them for other variables, there's zero point calculating the activation gradients for the "cast(fp) -> lots of FP ops" part of the graph, as the gradient from that branch won't go anywhere.
        How to determine minimal subset? Start with FP graph from step 1... then keep pruning leaves until the only remaining leaves are those FP variables that we need gradients for.

        Step 3: Differentiate ops in minimal subgraph
        The only major issue here is with multiple output ops, where only one of the outputs lead to the loss.
        For example, X -> slice -> (A,B); B -> loss, with A being unused.

         */



        final SameDiff outer = this;
        defineFunction("grad", new SameDiffFunctionDefinition() {

            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                //Propagate graph to this samediff instance which will also contain the backward
                if (SameDiff.this.debugMode) {
                    sameDiff.enableDebugMode();
                }

                outer.invokeGraphOn(sameDiff);
                if (debugMode) {
                    //Expect incoming args and outgoing args to be the same
                    Preconditions.checkState(sameDiff.ops.keySet().equals(ops.keySet()), "ops keysets not equal");
                }

                List<SameDiffOp> allFunctions = new ArrayList<>(sameDiff.ops.values());
                if (allFunctions.isEmpty()) {
                    throw new ND4JIllegalStateException("No ops found!");
                }

                for (SameDiffOp op : allFunctions) {
                    DifferentialFunction func = op.getOp();
                    if (func instanceof SDVariable) {
                        continue;
                    }

                    val args = func.args();
                    for (val arg : args)
                        arg.setSameDiff(sameDiff);
                    val outputs = func.outputVariables();
                    for (val output : outputs)
                        output.setSameDiff(sameDiff);
                    func.setSameDiff(sameDiff);
                }

                List<SDVariable> finalOutputs = new ArrayList<>(lossVariables.size());
                SDVariable initialGrad = sameDiff.var("one-var", Nd4j.scalar(1.0f));
                for(String s : lossVariables){
                    Preconditions.checkNotNull(s, "Encountered null value in loss variables. Null loss variables are not allowed." +
                            " Use SameDiff.setLossVariables with non-null array names to fix");
                    Preconditions.checkState(variables.containsKey(s), "Specified loss function variable \"%s\" does not exist", s);
                    SDVariable v = variables.get(s).getVariable();
                    Preconditions.checkState(v.dataType().isFPType(), "Specified loss function variable \"%s\" is not a floating" +
                            "point variable (datatype: %s). Only floating point variables may be used as loss function variable", s, v.dataType());
                    v = v.sum();    //If output is not a scalar: we'll use loss = v.sum(), same as adding loss for multiple outputs. We don't always know for sure if output is scalar at this point
                    if(v.dataType() == initialGrad.dataType()){
                        sameDiff.setGradientForVariableName(v.getVarName(), initialGrad);
                    } else {
                        sameDiff.setGradientForVariableName(v.getVarName(), initialGrad.castTo(v.dataType()));
                    }
                    if(finalOutputs.contains(v)){
                        log.warn("Loss function variable \"{}\" appears multiple times in list of loss variables - using only first instance", s);
                    } else {
                        finalOutputs.add(v);
                    }
                }

                if (log.isTraceEnabled()) {
                    String[] initialOutputsStr = allFunctions.get(allFunctions.size() - 1).getOp().outputVariablesNames();
                    String s = initialOutputsStr == null ? "null" : Arrays.toString(initialOutputsStr);
                    log.trace("Defining backward function: initial outputs {}", s);
                }


                //----- Step 1: Determine FP variables connected to loss -----
                // Find all FP variables that are connected to loss by an FP32 path
                Set<String> allFpVarsConnectedToLoss = new HashSet<>();
                Queue<String> toProcess = new LinkedList<>();
                for(String s : lossVariables){
                    if(!toProcess.contains(s)){
                        toProcess.add(s);
                    }
                }
                while(!toProcess.isEmpty()){
                    String next = toProcess.remove();
                    if(!allFpVarsConnectedToLoss.contains(next)){
                        Variable v = variables.get(next);
                        if(v.getVariable().dataType().isFPType()){
                            allFpVarsConnectedToLoss.add(v.getName());
                            //Work out what op (if any) this is an output of... and add the inputs to that op to be processed
                            if(v.getOutputOfOp() != null){
                                String opName = v.getOutputOfOp();
                                SameDiffOp op = ops.get(opName);
                                List<String> opInputs = op.getInputsToOp();
                                if(opInputs != null){
                                    for(String s : opInputs){
                                        Variable inputVar = variables.get(s);
                                        if(inputVar.getVariable().dataType().isFPType()){
                                            //Add this connected floating point type to the list to be processed
                                            toProcess.add(s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                //----- Step 2: Determine minimal set of FP variables actually required -----
                // Keep removing leaf nodes until only Variable type SDVariables remain
                Set<String> minimalSubgraph = new HashSet<>(allFpVarsConnectedToLoss);
                Queue<String> leafFPVars = new LinkedList<>();
                for(String s : allFpVarsConnectedToLoss){
                    //First: determine if is a FP leaf (Array type SDVariable)
                    Variable v = variables.get(s);
                    if(v.getVariable().getVariableType() == VariableType.ARRAY){
                        String opName = v.getOutputOfOp();  //Always defined for array type
                        SameDiffOp op = ops.get(opName);
                        List<String> inputsToOp = op.getInputsToOp();
                        boolean anyInputsInSubgraph = false;
                        if(inputsToOp != null){
                            for(String s2 : inputsToOp){
                                if(allFpVarsConnectedToLoss.contains(s2)){
                                    //Connection s2 -> s exists... therefore s is not a leaf (yet)
                                    anyInputsInSubgraph = true;
                                    break;
                                }
                            }
                        }
                        if(!anyInputsInSubgraph){
                            //Mark s as a leaf to be removed
                            leafFPVars.add(s);
                        }
                    }
                }

                while(!leafFPVars.isEmpty()){
                    String nextLeaf = leafFPVars.remove();
                    Variable v = variables.get(nextLeaf);
                    minimalSubgraph.remove(nextLeaf);

                    //Now, after removing: check what this variable is input to...
                    //If nextLeaf is input to some op X, then if none of inputs y->X are present in subgraph, then
                    // output variables X->z must now be leafs
                    //Note that any time we remove a variable, the only possible new leafs are those that this one
                    // is connected to.
                    List<String> inputsTo = v.getInputsForOp();
                    if( inputsTo != null && !inputsTo.isEmpty()) {
                        for (String opName : inputsTo) {
                            SameDiffOp op = ops.get(opName);
                            List<String> inputsToOp = op.getInputsToOp();
                            boolean anyPresent = false;
                            for(String s : inputsToOp){
                                if(minimalSubgraph.contains(s)){
                                    anyPresent = true;
                                    break;
                                }
                            }
                            if(!anyPresent){
                                //All inputs to op X are not in subgraph. Therefore outputs of op must be new leaves
                                List<String> outVars = op.getOutputsOfOp();
                                if(outVars != null) {
                                    for (String s : outVars) {
                                        if(!leafFPVars.contains(s)){
                                            //Mark this variable to be processed next
                                            leafFPVars.add(s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                //At this point: we know the set of variables that are connected to the loss - these all (and only) need gradients
                Queue<DifferentialFunction> availableForDiff = new LinkedList<>();
                for(SDVariable lossVar : finalOutputs){
                    Variable v = sameDiff.variables.get(lossVar.getVarName());
                    if(v.getOutputOfOp() != null){
                        String opName = v.getOutputOfOp();
                        availableForDiff.add(sameDiff.ops.get(opName).getOp());
                    }
                }

                Set<String> differentiatedOps = new HashSet<>();
                int numProcessed = 0;
                while(!availableForDiff.isEmpty()){
                    DifferentialFunction df = availableForDiff.remove();

                    //Get the inputs and outputs of the op
                    List<String> inputsToOp;
                    List<String> outputsOfOp;
                    if(df instanceof GradientBackwardsMarker){
                        SameDiffOp op = sameDiff.ops.get(df.getOwnName());
                        inputsToOp = op.getInputsToOp();
                        outputsOfOp = Collections.emptyList();
                    } else {
                        inputsToOp = sameDiff.ops.get(df.getOwnName()).getInputsToOp();
                        outputsOfOp = sameDiff.ops.get(df.getOwnName()).getOutputsOfOp();
                        numProcessed++;
                    }


                    //Get gradients for all output variables:
                    List<SDVariable> grads = new ArrayList<>();
                    for(String s : outputsOfOp){
                        SDVariable v = sameDiff.getVariable(s);
                        SDVariable g = v.gradient();

                        if(g == null){
                            //If no gradient exists at this point, 3 possibilities:
                            // (a) we have a bug
                            // (b) output of this op isn't used in calculating the loss
                            // (c) output isn't a FP type
                            //In the FP case, we should create a zero variable to backprop, because we can't perform backprop
                            // for this op otherwise...
                            if(!v.dataType().isFPType()){
                                grads.add(null);
                            } else {
                                SDVariable gTemp = sameDiff.zerosLike(v);
                                grads.add(gTemp);
                            }
                        } else {
                            grads.add(g);
                        }
                    }

                    //Differentiate:
                    List<SDVariable> currFnGrads = df.diff(grads);
                    differentiatedOps.add(df.getOwnName());
                    System.out.println("Differentiated op: \"" + df.getOwnName() + "\"");
                    numProcessed++;

                    //Check the inputs to this op, see if we can differentiate those ops now (and if so: add to queue)
                    for(String s : inputsToOp){
                        Variable v = sameDiff.variables.get(s);
                        String opName = v.getOutputOfOp();
                        if(opName == null || differentiatedOps.contains(opName)){
                            //Skip placeholder/constant etc; also skip if we've previously differentiated this op
                            continue;
                        }

                        //Next: we've just differentiated OpX
                        //For s -> OpX: we now have gradient for s after df.diff(grads) call earlier
                        //Now, do we also need to differentiate OpY, where OpY -> s?
                        //If any input variables x (x -> OpY) exist, if they are in the minimal subgraph, then we
                        // need to differentiate OpY too
                        //Note that just because we *need to* doesn't mean we *can* yet

                        boolean isRequiredOp = true;
                        SameDiffOp op = ops.get(opName);
                        if(op.getInputsToOp() != null){
                            List<String> opInputs = op.getInputsToOp();
                            boolean anyInputsRequired = false;
                            for(String s2 : opInputs){
                                if(minimalSubgraph.contains(s2)){
                                    anyInputsRequired = true;
                                    break;
                                }
                            }
                            if(anyInputsRequired){
                                if(!differentiatedOps.contains(op.getName())){
                                    isRequiredOp = true;
                                }
                            }
                        }

                        if(!isRequiredOp){
                            System.out.println("Skipped as not required: " + opName);
                            continue;
                        }

                        //Now that we know we need this op - check if we can actually differentiate it...
                        //We can differentiate it if, for all variables that are outputs of this op:
                        //(a) we have gradient already, OR
                        //(b) it's not a FP variable, OR
                        //(c) it's a FP variable but not one that the loss depends on

                        boolean allAvailable = true;
                        SameDiffOp o = sameDiff.ops.get(opName);
                        for(String opOutput : o.getOutputsOfOp()){
                            Variable outVar = variables.get(opOutput);
                            if(outVar.getVariable().dataType().isFPType()){
                                if(minimalSubgraph.contains(outVar.getName())){
                                    //Need gradient for this variable to be available before we can differentiate
                                    if(outVar.getVariable().gradient() == null){
                                        allAvailable = false;
                                        break;
                                    }
                                }
                                //If in't not in the minimal subgraph, loss doesn't depend on it, so we don't care about it
                            }
                        }

                        if(allAvailable){
                            availableForDiff.add(o.getOp());
                            System.out.println("Marked available for diff: " + o.getOp().getOwnName());
                        } else {
                            System.out.println("Not all inputs available: " + o.getName());
                        }
                    }
                }

                //Let's validate we actually differentiated everything correctly:
                for(String s : minimalSubgraph){
                    if(lossVariables.contains(s))
                        continue;
                    SDVariable g = variables.get(s).getVariable().gradient();
                    if(g == null){
                        throw new IllegalStateException("Error encountered during differentiation: no gradient for required variable \"" + s + "\" was calculated");
                    }
                }


                return new SDVariable[]{sameDiff.var("grad", org.nd4j.linalg.api.buffer.DataType.FLOAT, 1)};
            }
        });

        associateSameDiffWithOpsAndVariables();
    }


    /**
     * Set the original shape for a given place holder.<br>
     * This is used to track original shapes of place holder variables.<br>
     * The reason we track original shapes is to validate possible candidate arrays coming in (especially with -1
     * as the expected shapes).
     * <p>
     * Note that if {@link #isPlaceHolder(String)}
     * returns false for the passed in vertex id,
     * a {@link ND4JIllegalStateException} is thrown.
     * <p>
     *
     * @param variableName the vertex id for the original shape
     * @param shape        the shape of the place holder
     */
    public void setOriginalPlaceHolderShape(String variableName, long[] shape) {
        if (!isPlaceHolder(variableName)) {
            throw new ND4JIllegalStateException("Vertex id " + variableName + " does not appear to be a place holder. Did you forget to call addPlaceHolder?");
        }

        if (shape == null) {
            throw new ND4JIllegalStateException("Null and 0 length shape arrays not allowed");
        }


        if (placeHolderOriginalShapes.containsKey(variableName) && !Arrays.equals(placeHolderOriginalShapes.get(variableName), shape)) {
            throw new ND4JIllegalStateException("Unable to add a new shape for vertex id " + variableName);
        }

        //after validation now only set once
        placeHolderOriginalShapes.put(variableName, shape);

    }


    /**
     * Get the original shape for the vertex id if one was set (other wise returns null).<br>
     * This is mainly for use in validating passed in arrays as arguments to {@link #resolveVariablesWith(Map)}
     * usually when executing using {@link #execWithPlaceHolder(Map)}
     *
     * @param varName the vertex id to get the original shape for.
     * @return the set vertex
     */
    @Deprecated
    public long[] getOriginalShapeForPlaceHolder(String varName) {
        return placeHolderOriginalShapes.get(varName);
    }

    /**
     * Returns true if this vertex id is a place holder variable or not<br>
     * A place holder variable is one where the array shape(s) are currently known and can't yet be calculated
     *
     * @param varName the vertex id to test
     * @return True if the variable is a placeholder, false otherwise
     */
    public boolean isPlaceHolder(String varName) {
        Preconditions.checkState(variables.containsKey(varName), "No variable present in SameDiff instance with name \"%s\"", varName);
        return variables.get(varName).getVariable().isPlaceHolder();
    }


    /**
     * Resolve all ndarrays by updating the variables for each array specified in the given map.
     * An {@link IllegalStateException} will be thrown if not all arrays are specified for resolution.
     *
     * @param arrays the arrays to resolve.
     */
    public void resolveVariablesWith(Map<String, INDArray> arrays) {
        for (Map.Entry<String,INDArray> e : arrays.entrySet()) {
            SDVariable varForName = getVariable(e.getKey());
            if (varForName == null) {
                throw new ND4JIllegalStateException("No variable name found for " + e.getKey());
            }

            Variable v = variables.get(e.getKey());
            if(varForName.getVariableType() == VariableType.PLACEHOLDER){
                //Check shape:
                long[] shape = varForName.placeholderShape();
                long[] newShape = e.getValue().shape();
                Preconditions.checkState(shape.length == newShape.length, "Placeholder shape not compatible (mismatched rank): placeholder \"%s\" " +
                        "shape %s, got incompatible shape %s", e.getKey(), shape, newShape);
            }
        }


        for (val entry : arrays.entrySet()) {
            if (!variables.get(entry.getKey()).getVariable().isPlaceHolder()) {
                throw new ND4JIllegalStateException("Illegal variable " + entry.getKey() + " passed in. Variable found not to be a place holder variable");
            }

            val specifiedShape = getOriginalShapeForPlaceHolder(entry.getKey());
            //whole shape was specified: validate whether the input array shape is equal
            if (!Shape.isPlaceholderShape(specifiedShape)) {
                if (!Shape.shapeEquals(specifiedShape, entry.getValue().shape())) {
                    throw new ND4JIllegalStateException("Place holder shape specified was " + Arrays.toString(specifiedShape) + " but array shape was " + Arrays.toString(entry.getValue().shape()));
                }
            }

            associateArrayWithVariable(entry.getValue(), getVariable(entry.getKey()));
            setArrayForVariable(entry.getKey(), entry.getValue());
        }

        //declare resolved
        resolvedVariables = true;
    }

    /**
     * Updates the variable name property on the passed in variable, the reference in samediff, and returns the variable.
     * <p>
     * Note that if null for the new variable is passed in, it will just return the original input variable.
     *
     * @param varToUpdate the variable to update
     * @param newVarName  the new variable name
     * @return the passed in variable
     */
    public SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName) {
        if (varToUpdate == null) {
            throw new NullPointerException("Null input: No variable found for updating!");
        }

        if(newVarName != null && variables.containsKey(newVarName) && varToUpdate != variables.get(newVarName).getVariable()){
            throw new IllegalStateException("Variable name \"" + newVarName + "\" already exists for a different SDVariable");
        }

        if (newVarName == null && variables.containsKey(varToUpdate.getVarName())) {
            //Edge case: suppose we do m1=sd.mean(in), m2=sd.mean(m1) -> both initially have the name
            // "mean" and consequently a new variable name needs to be generated
            newVarName = generateNewVarName(varToUpdate.getVarName(), 0);
        }

        if (newVarName == null || varToUpdate.getVarName().equals(newVarName)) {
            return varToUpdate;
        }

        val oldVarName = varToUpdate.getVarName();
        varToUpdate.setVarName(newVarName);
        updateVariableName(oldVarName, newVarName);
        return varToUpdate;
    }

    @Override
    protected SameDiff sd() {
        //Helper method for SDBaseOps etc
        return this;
    }


    /**
     * Updates the variable name property on the passed in variables, its reference in samediff, and returns the variable.
     *
     * @param variablesToUpdate the variable to update
     * @param newVariableNames  the new variable name
     * @return the updated, passed in variables
     */
    public SDVariable[] updateVariableNamesAndReferences(SDVariable[] variablesToUpdate, String[] newVariableNames) {

        int numVariables = variablesToUpdate.length;
        SDVariable[] updatedVariables = new SDVariable[numVariables];

        for (int i = 0; i < numVariables; i++) {
            SDVariable varToUpdate = variablesToUpdate[i];
            String name = newVariableNames == null ? null : newVariableNames[i];
            updatedVariables[i] = updateVariableNameAndReference(varToUpdate, name);
        }

        return updatedVariables;
    }

    /**
     * Associate the current SameDiff instance with all ops and variables.
     * This is necessary to ensure that when dealing with shared state (usually with a SameDiff function such
     * as "grad" - the backward function) we have the correct SameDiff instance set for all ops/SDVariables.<br>
     * If this is not done, arrays and shapes could be fetched from the incorrect SameDiff instance for some methods
     */
    protected void associateSameDiffWithOpsAndVariables(){
        for(SDVariable var : variableMap().values()){
            var.setSameDiff(this);
        }
//        for(DifferentialFunction df : functionInstancesById.values()){
        for(SameDiffOp op : ops.values()){
            DifferentialFunction df = op.getOp();
            df.setSameDiff(this);

            //TODO: This is ugly but seemingly necessary
            //Finally, also set the SDVariable for each op
            //Otherwise: could have an op pointing to this SameDiff instance, but op's SDVariable's sameDiff field pointing
            // to another SameDiff instance. At which point, they could fetch shapes and arrays from some other instance
            // (i.e., not from this one that is currently executing)
            SDVariable[] args = df.args();
            if(args != null){
                for(SDVariable arg : args){
                    arg.setSameDiff(this);
                }
            }

            SDVariable[] outputs = df.outputVariables();
            if(outputs != null){
                for(SDVariable out : outputs){
                    out.setSameDiff(this);
                }
            }
        }
    }

    public Map<String,INDArray> execAll(Map<String,INDArray> placeholders){
        List<String> allVars = new ArrayList<>();
        for(Variable v : variables.values()){
            allVars.add(v.getName());
        }
        return exec(placeholders, allVars.toArray(new String[allVars.size()]));
    }

    public INDArray execSingle(Map<String,INDArray> placeholders, String output){
        return exec(placeholders, output).get(output);
    }

    public Map<String,INDArray> exec(Map<String,INDArray> placeholders, List<String> outputs){
        return exec(placeholders, outputs.toArray(new String[outputs.size()]));
    }

    public Map<String,INDArray> exec(Map<String,INDArray> placeholders, String... outputs){
        Preconditions.checkState(outputs != null && outputs.length > 0, "No outputs were specified");
        long threadId = Thread.currentThread().getId();
        if(!sessions.containsKey(threadId)){
            log.info("Creating new InferenceSession for thread {}", threadId);
            sessions.put(threadId, new InferenceSession(this));
        }

        List<String> phNames = inputs();
        if(placeholders == null && phNames != null){
            //Maybe user set placeholders before calling exec method?
            placeholders = placeholdersPerThread.get(Thread.currentThread().getId());
        }

        //Check that all placeholders are provided
        if(phNames != null && phNames.size() > 0) {
            Preconditions.checkNotNull(placeholders, "No placeholders were provided. Network has placeholders: %s", phNames);
            for (String s : phNames) {
                Preconditions.checkState(placeholders.containsKey(s), "No placeholder variable was provided for variable \"%s\"." +
                        " Cannot execute without all placeholders set", s);
            }
        }

        InferenceSession is = sessions.get(threadId);
        Map<String,INDArray> ret = is.output(Arrays.asList(outputs), placeholders);
        return ret;
    }


    protected int asFlatNode(String name, @NonNull SameDiff scope, @NonNull FlatBufferBuilder bufferBuilder) {
        int scopeName = bufferBuilder.createString(name);

        int flatNode = FlatNode.createFlatNode(bufferBuilder,
                scopeName,
                scopeName,
                OpType.LOGIC,
                10, // hardcoded value
                0,
                0,
                0,
                (byte) 0,
                0,
                0,
                0,
                0,
                -1,
                0, 0, 0, 0,0, 0);

        return flatNode;
    }

    /**
     * Note: INTENDED FOR DEVELOPER USE<br>
     * This method extract base variable name and output index (if exists) from raw variable name.
     * I.e:
     * - if variable name is "Unstack_2", result will be Pair("Unstack_2", 0)
     * - if variable name is "Unstack_2:12", result will be Pair("Unstack_2", 12)
     *
     * @param varName
     * @return
     */
    public static Pair<String, Integer> parseVariable(@NonNull String varName) {
        if (!varName.contains(":")) {
            return Pair.pairOf(varName, 0);
        } else {
            val split = varName.split(":");
            val index = Integer.valueOf(split[split.length - 1]);
            if (split.length == 2)
                return Pair.pairOf(split[0], index);
            else {
                val builder = new StringBuilder();
                for (int e = 0; e < split.length - 1; e++) {
                    builder.append(split[e]);

                    if (e < split.length - 2)
                        builder.append(":");
                }

                return Pair.pairOf(builder.toString(), index);
            }
        }
    }

    protected int asFlatNode(@NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables,
                             Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter, Integer id) {
        val opName = node.opName();
        val hash = FlatBuffersMapper.getOpNum(node.opName(), node.opType());
        //log.info("Exporting node: [{}:<{}> ; OpType: {}; Hash/opNum: {}]", node.opName(), node.tensorflowName(), node.opType(), hash);

        double[] extras;
        if(node.opType() == Op.Type.CUSTOM){
            CustomOp op = (CustomOp)node;
            extras = op.tArgs();
        } else {
            extras = node.getExtraArgs() != null ? new double[node.getExtraArgs().length] : new double[0];
            for (int e = 0; e < extras.length; e++) {
                extras[e] = ((Number) node.getExtraArgs()[e]).doubleValue();
            }
        }

        boolean[] boolArgs = null;
        long[] extraBits = null;
        if (node.opType() == Op.Type.CUSTOM) {
            DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) node;
            extraBits = dynamicCustomOp.iArgs();
            boolArgs = dynamicCustomOp.bArgs();
        } else if (node instanceof Enter) {
            // in case of Enter node we'll be storing unique frame reference
            val frameName = ((Enter) node).getFrameName();
            if (!framesMap.containsKey(frameName))
                framesMap.put(frameName, idCounter.incrementAndGet());

            extraBits = new long[]{framesMap.get(frameName).intValue()};
        } else
            extraBits = new long[]{};

        if (node.opType() == Op.Type.REDUCE_BOOL || node.opType() == Op.Type.REDUCE_SAME || node.opType() == Op.Type.REDUCE_FLOAT || node.opType() == Op.Type.REDUCE_LONG) {
            val op = (ReduceOp) node;

            boolArgs = new boolean[2];
            boolArgs[0] = op.isKeepDims();
            boolArgs[1] = true; // always new format
        } else if (node.opType() == Op.Type.INDEXREDUCE) {
            val op = (IndexAccumulation) node;

            boolArgs = new boolean[2];
            boolArgs[0] = op.isKeepDims();
            boolArgs[1] = true; // always new format
        }

        val inPaired = new ArrayList<Integer>();

        int[] outputIds = null;
        SDVariable[] outputVertexId = null;

        try {
            outputVertexId = node.outputVariables();
            outputIds = new int[outputVertexId.length];
            for (int i = 0; i < outputIds.length; i++) {
                outputIds[i] = variables.indexOf(outputVertexId[i]);
            }
        } catch (ND4UnresolvedOutputVariables e) {

            outputIds = new int[0];
            outputVertexId = null;
        } catch (Exception e) {
            throw new ND4JIllegalStateException(e);
        }


        SDVariable[] inputs = node.args();
        for (SDVariable input : inputs) {
            String varName = input.getVarName();
            int outIdx;
            if(this.variables.get(varName).getOutputOfOp() != null){
                DifferentialFunction df = ops.get(this.variables.get(varName).getOutputOfOp()).getOp();
                outIdx = ops.get(df.getOwnName()).getOutputsOfOp().indexOf(varName);
            } else {
                outIdx = 0;
            }

            if (!reverseMap.containsKey(varName)) {
                if (varName.contains("NextIteration")) {
                    // forward declaration: Merge node in case of loop will be referring to NextIteration node, which wasn't announced yet
                    int fwdNodeId = idCounter.incrementAndGet();
                    forwardMap.put(varName, fwdNodeId);
                    reverseMap.put(varName, fwdNodeId);
                } else {
                    throw new ND4JIllegalStateException("Unknown variable used in input: [" + varName + "]");
                }
            }

            int nodeId = reverseMap.get(varName);
            inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outIdx));
        }

        log.trace("Own Name: {}", node.getOwnName());
        int ownId = id != null ? id : idCounter.incrementAndGet();  //forwardMap.containsKey(node.getOwnName()) ? forwardMap.get(node.getOwnName()) : idCounter.incrementAndGet();
        String[] outNames = node.outputVariablesNames();
        for(String s : outNames){
            if(!reverseMap.containsKey(s)){
                reverseMap.put(s, ownId);
            }
        }

        int[] dims;
        if(node.opType() == Op.Type.REDUCE_FLOAT || node.opType() == Op.Type.REDUCE_SAME || node.opType() == Op.Type.REDUCE_BOOL || node.opType() == Op.Type.REDUCE_LONG || node.opType() == Op.Type.INDEXREDUCE || node.opType() == Op.Type.REDUCE3){
            dims = node.getDimensions();
            if(dims == null)
                dims = new int[0];
        } else {
            dims = new int[0];
        }
        Map<String,Object> fnProps = node.propertiesForFunction();
        int[] flatProperties = FlatBuffersMapper.mapFunctionPropertiesToFlatProperties(bufferBuilder, fnProps);
        int propIdx = FlatNode.createPropertiesVector(bufferBuilder, flatProperties);

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, outputIds);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int bArgs = FlatNode.createExtraBoolsVector(bufferBuilder, boolArgs != null ? boolArgs : new boolean[0]);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, dims);
        int fname = bufferBuilder.createString(node.getOwnName());
        int scopeName = bufferBuilder.createString("");
        int scalar = 0;
        if(node instanceof ScalarOp){
            ScalarOp sOp = (ScalarOp)node;
            INDArray s = sOp.scalar();
            if(s != null){
                scalar = s.toFlatArray(bufferBuilder);
            }
        }


        if (node.opType() == null)
            log.warn("Null-op node: {}", node);


        List<String> outVarNames = node.getSameDiff().ops.get(node.getOwnName()).getOutputsOfOp();
        int[] outVarNamesStringsOffsets = new int[outVarNames == null ? 0 : outVarNames.size()];
        for( int i=0; i<outVarNamesStringsOffsets.length; i++ ){
            outVarNamesStringsOffsets[i] = bufferBuilder.createString(outVarNames.get(i));
        }
        int outVarNamesOffset = FlatNode.createOutputNamesVector(bufferBuilder, outVarNamesStringsOffsets);

        int opNameOffset = bufferBuilder.createString(opName);

        byte[] outTypes = new byte[outVarNames.size()];
        int i=0;
        for(String s : outVarNames){
            SDVariable v = getVariable(s);
            outTypes[i++] = FlatBuffersMapper.getDataTypeAsByte(v.dataType());
        }
        int outTypesOffset = FlatNode.createOutputTypesVector(bufferBuilder, outTypes);

        int flatNode = FlatNode.createFlatNode(
                bufferBuilder,
                ownId,
                fname,
                FlatBuffersMapper.getFlatOpType(node.opType()),
                hash,
                propIdx,
                nodesIn,
                nodesInPaired,
                nodesOut,
                extraz,
                integerArgs,
                bArgs,
                dimensions,
                -1,     //Device
                0,      //Scope ID
                scopeName,      //Scope name
                outVarNamesOffset,
                opNameOffset,
                outTypesOffset,   //Output types
                scalar
        );

        return flatNode;
    }

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * @param configuration - ExecutorConfiguration to be embedded into serialized graph
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers(@NonNull ExecutorConfiguration configuration) {
        return asFlatBuffers(0, configuration);
    }

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * @param configuration - ExecutorConfiguration to be embedded into serialized graph
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers(long graphId, @NonNull ExecutorConfiguration configuration) {
        Nd4j.getExecutioner().commit();
        val bufferBuilder = new FlatBufferBuilder(1024);
        val idCounter = new AtomicInteger(0);

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();

        // first of all we build VariableSpace dump
        val variableList = new ArrayList<SDVariable>(variables());
        val reverseMap = new LinkedHashMap<String, Integer>();
        val forwardMap = new LinkedHashMap<String, Integer>();
        val framesMap = new LinkedHashMap<String, Integer>();

        int idx = 0;
        val idxForOps = new IdentityHashMap<DifferentialFunction,Integer>();
        List<SDVariable> allVars = variables();
        for (SDVariable variable : allVars) {
            INDArray arr = variable.getArr();
            log.trace("Exporting variable: [{}]", variable.getVarName());

            //If variable is the output of some op - let's use the ONE index for exporting, and properly track the output
            // numbers. For example, unstack(x) -> y0, y1, y2 -> the y's should be say (3,0), (3,1), (3,2) NOT (4,0), (5,0), (6,0)
            String varName = variable.getVarName();
            int varIdx;
            int outputNum;
            if(variables.get(varName).getOutputOfOp() != null){
                //This variable is the output of a node
                DifferentialFunction df = ops.get(variables.get(varName).getOutputOfOp()).getOp();
                if(!idxForOps.containsKey(df)){
                    varIdx = idCounter.incrementAndGet();
                    idxForOps.put(df, varIdx);
                } else {
                    varIdx = idxForOps.get(df);
                }
                String[] outNames = df.outputVariablesNames();
                outputNum = ArrayUtils.indexOf(outNames, varName);
                Preconditions.checkState(outputNum >= 0, "Variable name \"%s\" not found in list of outputs: %s", varName, outNames);
            } else {
                varIdx = idCounter.incrementAndGet();
                outputNum = 0;
            }


            reverseMap.put(variable.getVarName(), varIdx);
            log.trace("Adding [{}] as [{}]", variable.getVarName(), varIdx);

            int shape = 0;
            int name = bufferBuilder.createString(variable.getVarName());
            int array = arr == null ? 0 : arr.toFlatArray(bufferBuilder);
            int id = IntPair.createIntPair(bufferBuilder, varIdx, outputNum);
            byte varType = (byte)variable.getVariableType().ordinal();

            if (variable.getVariableType() == VariableType.PLACEHOLDER) {
                val shp = variable.getShape();
                shape = FlatVariable.createShapeVector(bufferBuilder, shp);
            }

            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name,  FlatBuffersMapper.getDataTypeAsByte(variable.dataType()), shape, array, -1, varType);
            flatVariables.add(flatVariable);
        }

        //add functions
        for(SameDiffOp op : ops.values()){
            DifferentialFunction func = op.getOp();
            Integer fnId = idxForOps.get(func);
            flatNodes.add(asFlatNode(func, bufferBuilder, variableList, reverseMap, forwardMap, framesMap, idCounter, fnId));
        }

        // we're dumping scopes now
        for (Map.Entry<String, SameDiff> scope : sameDiffFunctionInstances.entrySet()) {
            if(scope.getKey().equalsIgnoreCase("grad")){
                //Skip the gradient function for export
                continue;
            }

            flatNodes.add(asFlatNode(scope.getKey(), scope.getValue(), bufferBuilder));
            val currVarList = new ArrayList<SDVariable>(scope.getValue().variables());
            // converting all ops from node
            for (val node : scope.getValue().variables()) {
                INDArray arr = node.getArr();
                if (arr == null) {
                    continue;
                }

                int name = bufferBuilder.createString(node.getVarName());
                int array = arr.toFlatArray(bufferBuilder);
                int id = IntPair.createIntPair(bufferBuilder, ++idx, 0);

                val pair = parseVariable(node.getVarName());
                reverseMap.put(pair.getFirst(), idx);

                log.trace("Adding [{}] as [{}]", pair.getFirst(), idx);

                byte varType = (byte)node.getVariableType().ordinal();
                int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, FlatBuffersMapper.getDataTypeAsByte(arr.dataType()),0, array, -1, varType);
                flatVariables.add(flatVariable);
            }

            //add functions
            for(SameDiffOp op : scope.getValue().ops.values()){
                DifferentialFunction func = op.getOp();
                flatNodes.add(asFlatNode(func, bufferBuilder, currVarList, reverseMap, forwardMap, framesMap, idCounter, null));
            }
        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int numPlaceholders = 0;
        for(SDVariable v : variables()){
            if(v.isPlaceHolder()){
                numPlaceholders++;
            }
        }

        int[] placeholderOffsets = new int[numPlaceholders];
        if(numPlaceholders > 0){
            int i=0;
            for(SDVariable v : variables()){
                if(!v.isPlaceHolder())
                    continue;
                placeholderOffsets[i++] = bufferBuilder.createString(v.getVarName());
            }
        }
        int placeholdersOffset = FlatGraph.createPlaceholdersVector(bufferBuilder, placeholderOffsets);

        List<String> lossVars = getLossVariables();
        int[] lossVarOffsets = new int[lossVars == null ? 0 : lossVars.size()];
        for( int i=0; i<lossVarOffsets.length; i++ ){
            lossVarOffsets[i] = bufferBuilder.createString(lossVars.get(i));
        }
        int lossVarOffset = FlatGraph.createLossVariablesVector(bufferBuilder, lossVarOffsets);

        int fg = FlatGraph.createFlatGraph(bufferBuilder, graphId, variablesOffset, nodesOffset, outputsOffset,
                configuration.getFlatConfiguration(bufferBuilder), placeholdersOffset, lossVarOffset);
        bufferBuilder.finish(fg);

        synchronized (this) {
            for(Map.Entry<String,Integer> e : reverseMap.entrySet()){
                this.variables.get(e.getKey()).setVariableIndex(e.getValue());
            }
        }

        return bufferBuilder.dataBuffer();
    }

    public FlatGraph asFlatGraph() {
        return FlatGraph.getRootAsFlatGraph(this.asFlatBuffers());
    }

    /**
     * This method returns FlatGraph structure
     *
     * @param configuration
     * @return
     */
    public FlatGraph asFlatGraph(long graphId, ExecutorConfiguration configuration) {
        return FlatGraph.getRootAsFlatGraph(asFlatBuffers(graphId, configuration));
    }

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers() {
        val configuration = ExecutorConfiguration.builder()
                .outputMode(OutputMode.VARIABLE_SPACE)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .build();

        return asFlatBuffers(configuration);
    }


    /**
     * Save this samediff instance with its training config.
     * Note that if a training configuration is not defined,
     * an {@link IllegalStateException} is thrown.
     *
     * @param outputStream the output stream to write to
     * @throws IOException
     */
    public void saveWithTrainingConfig(OutputStream outputStream) throws IOException {
        if(this.trainingConfig == null) {
            throw new IllegalStateException("No training configuration found!");
        }

        saveWithTrainingConfig(this.trainingConfig,outputStream);
    }



    /**
     * Save this samediff instance with its training config.
     * Note that if a training configuration is not defined,
     * an {@link IllegalStateException} is thrown.
     *
     * @param outputFile the output stream to write to
     * @throws IOException
     */
    public void saveWithTrainingConfig(File outputFile) throws IOException {
        if(this.trainingConfig == null) {
            throw new IllegalStateException("No training configuration found!");
        }

        try(BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(outputFile))) {
            saveWithTrainingConfig(this.trainingConfig, bufferedOutputStream);
            bufferedOutputStream.flush();
        }

    }


    /**
     * Save this samediff instance as a zip file
     * with the training configuration
     * @param trainingConfig the training configuration to save
     * @param outputStream the output stream to write to
     * @throws IOException
     */
    public void saveWithTrainingConfig(TrainingConfig trainingConfig,OutputStream outputStream) throws  IOException {
        ObjectMapper objectMapper = ObjectMapperHolder.getJsonMapper();
        String configJson = objectMapper.writeValueAsString(trainingConfig);
        ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(outputStream));
        ZipEntry config = new ZipEntry(TRAINING_CONFIG_JSON_ZIP_ENTRY_NAME);
        zipfile.putNextEntry(config);
        zipfile.write(configJson.getBytes());

        ZipEntry sameDiff = new ZipEntry(SAMEDIFF_FILE_ENTRY_NAME);
        zipfile.putNextEntry(sameDiff);

        val fb = asFlatBuffers();
        val offset = fb.position();

        val array = fb.array();

        try (BufferedOutputStream zipFileOutputStream = new BufferedOutputStream(zipfile);
             val dos = new DataOutputStream(zipFileOutputStream)) {
            dos.write(array, offset, array.length - offset);
        }
    }


    /**
     * Restore a {@link SameDiff}
     * instance from a configuration
     * zip file
     * @param file the file to restore from
     * @return the associated samediff instance
     * @throws IOException
     */
    public static SameDiff restoreFromTrainingConfigZip(File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);
        ZipEntry config = zipFile.getEntry(TRAINING_CONFIG_JSON_ZIP_ENTRY_NAME);
        TrainingConfig trainingConfig = null;
        try(InputStream stream = zipFile.getInputStream(config)) {
            byte[] read = IOUtils.toByteArray(stream);
            trainingConfig = ObjectMapperHolder.getJsonMapper().readValue(read,TrainingConfig.class);
        }

        SameDiff ret = null;

        ZipEntry sameDiffFile = zipFile.getEntry(SAMEDIFF_FILE_ENTRY_NAME);
        try(InputStream stream = zipFile.getInputStream(sameDiffFile)) {
            byte[] read = IOUtils.toByteArray(stream);
            ret = SameDiff.fromFlatBuffers(ByteBuffer.wrap(read));
        }


        ret.setTrainingConfig(trainingConfig);
        ret.initializeTraining();
        return ret;
    }

    /**
     * This method converts SameDiff instance to
     * FlatBuffers and saves it to file which
     * can be restored later
     *
     * @param file File to save the FlatBuffers serialized graph (including arrays) to
     */
    public void asFlatFile(@NonNull File file) throws IOException {
        val fb = asFlatBuffers();
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }

    /**
     * This method converts SameDiff instance to FlatBuffers and saves it to file which can be restored later
     *
     * @param file File to save the FlatBuffers serialized graph (including arrays) to
     */
    public void asFlatFile(@NonNull File file, @NonNull ExecutorConfiguration configuration) throws IOException {
        val fb = asFlatBuffers(configuration);
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }


    /**
     * Create a {@link SameDiff}
     * instance from a file.
     * The method to save the file is
     * {@link #asFlatFile(File)}
     * @param file the file to load from
     * @return the loaded same diff instance
     * @throws IOException
     */
    public static SameDiff fromFlatFile(@NonNull File file) throws IOException {
        byte[] bytes;
        try (InputStream is = new BufferedInputStream(new FileInputStream(file))) {
            bytes = IOUtils.toByteArray(is);
        }

        ByteBuffer bbIn = ByteBuffer.wrap(bytes);
        return fromFlatBuffers(bbIn);
    }

    /**
     * Create a {@link SameDiff}
     * instance from a byte buffers
     * instance.
     * @param bbIn the input byte buffer
     * @return the created samediff instance
     * @throws IOException
     */
    public static SameDiff fromFlatBuffers(ByteBuffer bbIn) throws IOException {

        FlatGraph fg = FlatGraph.getRootAsFlatGraph(bbIn);

        int numOps = fg.nodesLength();
        int numVars = fg.variablesLength();
        List<FlatNode> ops = new ArrayList<>(numOps);
        for( int i=0; i<numOps; i++ ){
            ops.add(fg.nodes(i));
        }
        List<FlatVariable> vars = new ArrayList<>(numVars);
        for( int i = 0; i < numVars; i++) {
            vars.add(fg.variables(i));
        }

        FlatConfiguration conf = fg.configuration();

        /* Reconstruct the graph
        We'll do the reconstruction manually here, rather than using sd.var(...), so that we have more control
        over the final result.
         */

        SameDiff sd = SameDiff.create();

        //Reconstruct placeholders
        int numPlaceholders = fg.placeholdersLength();
        Set<String> ph = new LinkedHashSet<>();
        for(int i=0; i<numPlaceholders; i++ ){
            ph.add(fg.placeholders(i));
        }

        //Reconstruct variables:
        Map<Integer,SDVariable> varNodeIds = new HashMap<>();
        Map<Pair<Integer,Integer>, SDVariable> variablesByNodeAndOutNum = new HashMap<>();
        Map<String,List<SDVariable>> variablesByName = new HashMap<>();
        for(FlatVariable v : vars){
            int shapeLength = v.shapeLength();
            long[] shape = new long[shapeLength];
            for( int i = 0; i < shapeLength; i++) {
                shape[i] = v.shape(i);
            }

            String n = v.name();

            byte dtypeByte = v.dtype();
            org.nd4j.linalg.api.buffer.DataType dtype = FlatBuffersMapper.getDataTypeFromByte(dtypeByte);

            //TODO Infer this properly! Could be constant, etc.
            VariableType vt = VariableType.values()[v.variabletype()];
            SDVariable var = new SDVariable(n, vt, sd, shape, dtype, null);
            sd.variables.put(n, Variable.builder().name(n).variable(var).build());
            sd.variableNameToShape.put(n, shape);


            FlatArray fa = v.ndarray();
            if(fa != null && vt != VariableType.ARRAY){
                INDArray arr;
                try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    arr = Nd4j.createFromFlatArray(fa);
                }
                sd.setArrayForVariable(n, arr);
            }

            IntPair id = v.id();    //First value: node (op) id. Second: output number
            variablesByNodeAndOutNum.put(new Pair<>(id.first(), id.second()), var);

            if(!variablesByName.containsKey(n)){
                variablesByName.put(n, new ArrayList<SDVariable>());
            }

            List<SDVariable> list = variablesByName.get(n);
            list.add(var);
        }

        //Reconstruct ops:
        for(FlatNode fn : ops){
            DifferentialFunction df = FlatBuffersMapper.fromFlatNode(fn);
            String name = fn.name();
            df.setSameDiff(sd);
            df.setOwnName(name);
            if(sd.ops.containsKey(name)){
                sd.ops.get(name).setOp(df);
            } else {
                sd.ops.put(name, SameDiffOp.builder().name(name).op(df).build());
            }

            int outLength = fn.outputLength();
            int[] outs = new int[outLength];
            for( int i=0; i<outLength; i++ ){
                outs[i] = fn.output(i);
            }

            int opId = fn.id();

            //Work out inputs and outputs:
            int[] output = new int[fn.outputLength()];
            for (int i = 0; i < output.length; i++) {
                output[i] = fn.output(i);
            }
            int[] input = new int[fn.inputLength()];
            for (int i = 0; i < input.length; i++) {
                input[i] = fn.input(i);
            }
            IntPair[] inputPaired = new IntPair[fn.inputPairedLength()];
            List<Pair<Integer,Integer>> intPairList = new ArrayList<>();
            for (int i = 0; i < inputPaired.length; i++) {
                inputPaired[i] = fn.inputPaired(i);
                intPairList.add(new Pair<>(inputPaired[i].first(), inputPaired[i].second()));
            }

            String[] inputNames = new String[inputPaired.length];
            for(int i=0; i<inputPaired.length; i++ ){
                int nodeId = inputPaired[i].first();
                int nodeOutNum = inputPaired[i].second();
                SDVariable varIn = variablesByNodeAndOutNum.get(new Pair<>(nodeId, nodeOutNum));
                if(varIn == null){
                    //The variable corresponding to this op was not
                }
                inputNames[i] = varIn.getVarName();
            }
            sd.ops.get(df.getOwnName()).setInputsToOp(Arrays.asList(inputNames));

            //Record that input variables are input to this op
            for(String inName : inputNames) {
                Variable v = sd.getVariables().get(inName);
                if(v.getInputsForOp() == null){
                    v.setInputsForOp(new ArrayList<String>());
                }
                if(!v.getInputsForOp().contains(df.getOwnName())){
                    v.getInputsForOp().add(df.getOwnName());
                }
            }

            List<SDVariable> varsForOp = variablesByName.get(name);

            //Can't assume that variables for the op have all been defined. For example, if we export before execution in SameDiff
            //In theory, we can reconstruct the output variables (minus names) if we know the number of op outputs
            //And we can calculate the op outputs - in most cases - after the op has been created and parameters set
            int numOutputs = df.getNumOutputs();
            if(numOutputs <= 0){
                numOutputs = fn.outputLength();
            }

            String[] varNames = null;
            if(varsForOp != null && varsForOp.size() == numOutputs){
                varNames = new String[varsForOp.size()];
                for( int i=0; i<varNames.length; i++ ){
                    varNames[i] = varsForOp.get(i).getVarName();
                    sd.getVariables().get(varNames[i]).setOutputOfOp(df.getOwnName());
                }
                sd.ops.get(df.getOwnName()).setOutputsOfOp(Arrays.asList(varNames));
            } else {
                //We're missing some variables...
                int outputNamesLength = fn.outputNamesLength();
                varNames = new String[outputNamesLength];
                for( int i=0; i<outputNamesLength; i++ ){
                    String n = fn.outputNames(i);
                    varNames[i] = n;
                    if(!sd.variables.containsKey(n)){
                        //Need to create the variable - perhaps it wasn't exported. Note output of node -> can only be VARIABLE type
                        SDVariable var = new SDVariable(n, VariableType.VARIABLE, sd, null, null, null);
                        sd.variables.put(n, Variable.builder().name(n).variable(var).build());
                        variablesByNodeAndOutNum.put(new Pair<>(opId, i), var);
                    }
                    sd.getVariables().get(varNames[i]).setOutputOfOp(df.getOwnName());
                }
                sd.ops.get(df.getOwnName()).setOutputsOfOp(Arrays.asList(varNames));
            }

            //Check the op mapping int he variablesByNodeAndOutputNum
            //For multi-output ops, variables will have their own index, not related to the op index
            for( int i=0; i<varNames.length; i++ ){
                Pair<Integer,Integer> p = new Pair<>(opId, i);
                if(!variablesByNodeAndOutNum.containsKey(p)){
                    variablesByNodeAndOutNum.put(p, sd.getVariable(varNames[i]));
                }
            }
        }

        //Reconstruct loss variables
        if(fg.lossVariablesLength() > 0){
            for(int i=0; i<fg.lossVariablesLength(); i++ ){
                sd.addLossVariable(fg.lossVariables(i));
            }
        }

        return sd;
    }

    /**
     * This method returns a text representation of the "flattened" graph.
     *
     * @return String representation of the graph
     * @see #summary()
     */
    public String asFlatPrint() {
        val sb = new StringBuilder();
        val fb = asFlatBuffers();

        val graph = FlatGraph.getRootAsFlatGraph(fb);

        sb.append("\nExternal variables:\n\n");
        for (int e = 0; e < graph.variablesLength(); e++) {
            val var = graph.variables(e);
            INDArray ndarray = null;
            try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                FlatArray fa = var.ndarray();
                if(fa != null) {
                    ndarray = Nd4j.createFromFlatArray(fa);
                }
            }

            sb.append(var.id().first())
                    .append(":<").append(var.name()).append("> ");
            if(ndarray == null){
                sb.append("<no array>").append("; Values: ").append("<no array>").append(";\n");
            } else {
                sb.append(Arrays.toString(ndarray.shapeInfoDataBuffer().asInt())).append("; Values: ");
                if(ndarray.data() == null){
                    //Empty array
                    sb.append("<empty array>");
                } else if(ndarray.dataType() == DataType.UTF8) {
                    sb.append("<string array>");
                } else {
                    if(ndarray.length() < 50){
                        sb.append(Arrays.toString(ndarray.data().asFloat()).replaceAll(" ",""));
                    } else {
                        //Array is too long - only tak. last few values...
                        sb.append("[");
                        for( int i=0; i<50; i++ ){
                            if(i > 0)
                                sb.append(",");
                            sb.append(ndarray.data().getFloat(i));
                        }
                        sb.append("]");
                    }
                }
                sb.append(";\n");
            }

        }

        val map = Nd4j.getExecutioner().getCustomOperations();


        sb.append("\nOps sequence:\n\n");
        for (int e = 0; e < graph.nodesLength(); e++) {
            val node = graph.nodes(e);

            log.info("{}:<{}>", node.id(), node.name());
            sb.append(node.id())
                    .append(":<").append(node.name()).append("> ").append(FlatBuffersMapper.getTypeFromByte(node.opType()));

            if (FlatBuffersMapper.getTypeFromByte(node.opType()) != Op.Type.CUSTOM)
                sb.append(": ").append(node.opNum());
            else {
                val keys = map.keySet();
                String opName = null;
                for (val k : keys) {
                    val d = map.get(k);
                    if (d.getHash() == node.opNum())
                        opName = k;
                }

                if (opName == null)
                    opName = "unknown";

                sb.append(": ").append(opName);
            }

            sb.append("; Inputs: {");

            for (int i = 0; i < node.inputPairedLength(); i++) {
                val pair = node.inputPaired(i);

                sb.append("[").append(pair.first()).append(":").append(pair.second()).append("]");

                if (i < node.inputPairedLength() - 1)
                    sb.append(", ");
            }

            sb.append("};");
            sb.append(" OpNum: {").append(node.opNum()).append("};");

            sb.append("\n");
        }


        return sb.toString();
    }

    /**
     * Generate and return a String representation of the current SameDiff instance<br>
     * Reports variables, ops, SameDiff function instances, and (where possible) array shapes.<br>
     * For ops, the input and output variables are reported.<br>
     * For variables, the ops that they are inputs to - or outputs of - are also reported
     *
     * @return A String representation of the SameDiff instance
     */
    public String summary() {

        Map<String, SDVariable> varMap = variableMap();
        DifferentialFunction[] functions = functions();


        int countVarsWithArrays = 0;
        for (String s : varMap.keySet()) {
            if (getArrForVarName(s) != null) {
                countVarsWithArrays++;
            }
        }

        StringBuilder sb = new StringBuilder();
        String format = "%-25s%-20s";
        sb.append("--- Summary ---\n");
        sb.append(String.format(format, "Variables:", varMap.size())).append(" (").append(countVarsWithArrays).append(" with arrays)").append("\n")
                .append(String.format(format, "Functions:", functions.length)).append("\n")
                .append(String.format(format, "SameDiff Function Defs:", sameDiffFunctionInstances.size()))
                .append("\n\n");

        sb.append("--- Variables ---\n");
        //Work out which function - if any - this arg is an output of...
        Map<String, String> outputOfFn = new HashMap<>();
        int maxLengthOutputOf = 22;     //Length of "- Output Of Function -"
        int maxLengthOfName = 8;       //Length of "- Name -"
        for (String s : varMap.keySet()) {
            String outputOf = null;
            for(SameDiffOp op : ops.values()){
                List<String> outputsOfOp = op.getOutputsOfOp();
                if (outputsOfOp != null && outputsOfOp.contains(s)) {
                    outputOf = op.getName();
                    break;
                }
            }

            if (outputOf == null) {
                outputOf = "<none>";
            } else {
                DifferentialFunction d = getFunctionById(outputOf);
                outputOf = d.getOwnName() + "(" + d.opName() + ")";
            }
            outputOfFn.put(s, outputOf);
            maxLengthOutputOf = Math.max(maxLengthOutputOf, outputOf.length());
            maxLengthOfName = Math.max(maxLengthOfName, s.length());
        }
        maxLengthOutputOf += 2;
        maxLengthOfName += 2;

        //Create the output for values:
        format = "%-" + maxLengthOfName + "s%-20s%-20s%-20s%-" + maxLengthOutputOf + "s%-20s";
        sb.append(String.format(format, "- Name -", "- Array Shape -", "- Variable Type -", "- Data Type-", "- Output Of Function -", "- Inputs To Functions -")).append("\n");
        for (String s : varMap.keySet()) {
            INDArray arr = getArrForVarName(s);
            String arrayShape = "-";
            if (arr != null) {
                arrayShape = Arrays.toString(arr.shape());
            }
            String varType = getVariable(s).getVariableType().toString();
            String dtype = getVariable(s).dataType().toString();

            List<String> argNames = variables.get(s).getInputsForOp();
            String dfArrStr = "";
            if (argNames != null) {
                dfArrStr = argNames.toString();
            }

            String outputOfStr = outputOfFn.get(s);

            sb.append(String.format(format, s, arrayShape, varType, dtype, outputOfStr, dfArrStr)).append("\n");
        }

        sb.append("\n\n--- Functions ---\n");

        //First: work out the amount of space we need for inputs and outputs...
        List<String> dfInputStr = new ArrayList<>();
        List<String> dfOutputStr = new ArrayList<>();
        int maxInLength = 10;       //Length of "- Inputs -"
        int maxOutLength = 11;      //Length of "- Outputs -"
        int maxOpNameLength = 17;   //Default to min of 17 - length of "- Function Name -"
        int maxDfClassNameLength = 10;  //Default to min of 10
        for (DifferentialFunction df : functions) {
            String[] argNames = df.argNames();
            String[] outNames = df.outputVariablesNames();

            String argStr = Arrays.toString(argNames);
            String outStr = Arrays.toString(outNames);

            maxInLength = Math.max(maxInLength, argStr.length());
            maxOutLength = Math.max(maxOutLength, outStr.length());

            dfInputStr.add(argStr);
            dfOutputStr.add(outStr);

            String name = df.getOwnName() == null ? df.opName() : df.getOwnName();
            maxOpNameLength = Math.max(maxOpNameLength, name.length());
            maxDfClassNameLength = Math.max(maxDfClassNameLength, df.getClass().getSimpleName().length());
        }
        //Extra padding space
        maxInLength += 2;
        maxOutLength += 2;
        maxOpNameLength += 2;
        maxDfClassNameLength += 2;


        format = "%-5s%-" + maxOpNameLength + "s%-" + maxDfClassNameLength + "s%-" + maxInLength + "s%-" + maxOutLength + "s";
        sb.append(String.format(format, "", "- Function Name -", "- Op -", "- Inputs -", "- Outputs -")).append("\n");
        for (int i = 0; i < functions.length; i++) {
            DifferentialFunction df = functions[i];
            String fnName = df.getOwnName() == null ? df.opName() : df.getOwnName();

            sb.append(String.format(format, String.valueOf(i), fnName, df.getClass().getSimpleName(), dfInputStr.get(i), dfOutputStr.get(i))).append("\n");
        }

        if (sameDiffFunctionInstances.size() > 0) {
            sb.append("\n\n--- SameDiff Defined Functions ---\n");
            format = "%-20s%-15s%-15s%-15s";
            sb.append(String.format(format, "- Name -", "- Variables -", "- Functions -", "- Fn Defs -")).append("\n");
            for (Map.Entry<String, SameDiff> e : sameDiffFunctionInstances.entrySet()) {
                SameDiff sd = e.getValue();
                int vars = sd.variableMap().size();
                int fns = (sd.functions() == null ? 0 : sd.functions().length);
                int defFns = sd.definedFunctionNames().size();

                sb.append(String.format(format, e.getKey(), String.valueOf(vars), String.valueOf(fns), String.valueOf(defFns))).append("\n");
            }
        }

        return sb.toString();
    }


    public Map<String,org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(){
        List<String> allVars = new ArrayList<>(variables.keySet());
        DataTypesSession session = new DataTypesSession(this);
        Map<String,org.nd4j.linalg.api.buffer.DataType> phValues = new HashMap<>();
        for(Variable v : variables.values()){
            if(v.getVariable().isPlaceHolder()){
                org.nd4j.linalg.api.buffer.DataType dt = v.getVariable().dataType();
                Preconditions.checkNotNull(dt, "Placeholder variable %s has null datatype", v.getName());
                phValues.put(v.getName(), dt);
            }
        }
        Map<String, org.nd4j.linalg.api.buffer.DataType> out = session.output(allVars, phValues);
        return out;
    }
}
