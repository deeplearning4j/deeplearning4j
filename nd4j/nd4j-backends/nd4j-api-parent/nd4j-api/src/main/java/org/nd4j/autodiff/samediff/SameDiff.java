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

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.listeners.*;
import org.nd4j.autodiff.listeners.impl.HistoryListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.api.OutAndGrad;
import org.nd4j.autodiff.samediff.array.SingleThreadArrayHolder;
import org.nd4j.autodiff.samediff.array.ThreadSafeArrayHolder;
import org.nd4j.autodiff.samediff.config.BatchOutputConfig;
import org.nd4j.autodiff.samediff.config.EvaluationConfig;
import org.nd4j.autodiff.samediff.config.FitConfig;
import org.nd4j.autodiff.samediff.config.OutputConfig;
import org.nd4j.autodiff.samediff.internal.*;
import org.nd4j.autodiff.samediff.ops.*;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.graph.*;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.dataset.AsyncMultiDataSetIterator;
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
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.DeviceLocalNDArray;
import org.nd4j.linalg.util.ND4JFileUtils;
import org.nd4j.shade.guava.base.Predicates;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Maps;
import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.GraphDef;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.nd4j.autodiff.util.TrainingUtils.stackOutputs;

/**
 * SameDiff is the entrypoint for ND4J's automatic differentiation functionality.
 * <p>
 * You define a graph symbolically.
 * <p>
 * That graph accumulates operations.
 * <p>
 * In order to execute the graph, you run one of the execution methods, such as {@link #output(Map, String...)}
 */
@AllArgsConstructor
@Builder
@Slf4j
public class SameDiff extends SDBaseOps {
    protected static final String GRAD_FN_KEY = "grad";

    //Fields for graph structure and execution
    @Getter
    private final Map<String, Variable> variables = new LinkedHashMap<>();         //Use linked hash map to guarantee iteration order based on order they were added. Used in inputs() and flatbuffers serde
    @Getter
    private final Map<String, SameDiffOp> ops = new LinkedHashMap<>();
    @Getter
    private final Map<Long, InferenceSession> sessions = new ConcurrentHashMap<>();      //Key: thread ID

    private ArrayHolder constantArrays = new ThreadSafeArrayHolder(true);
    private ArrayHolder variablesArrays = new ThreadSafeArrayHolder(true);
    private final Map<Long, Map<String, INDArray>> placeholdersPerThread = new ConcurrentHashMap<>(); //Placeholders for each thread - if the user sets them

    private final List<String> lossVariables = new ArrayList<>();

    private final List<Listener> listeners = new ArrayList<>();

    private final List<NameScope> nameScopes = new ArrayList<>();  //Used as a stack

    private List<String> outputs;       //Names of the output variables, set by the user.

    ///////////////////////////////////////
    //Fields related to training
    @Getter
    private TrainingConfig trainingConfig;                          //Configuration for training. Must be set for training/evaluation, but not for other operations
    @Getter
    private boolean initializedTraining;                            //True if training setup has been done
    @Getter
    private Map<String, GradientUpdater> updaterMap;                 //GradientUpdater instance for each trainable parameter

    ////////////////////////////////////////

    private DifferentialFunctionFactory functionFactory;

    // counter for auto-naming variables
    private int variableId = 0;

    ////////////////////////////////////////

    /**
     * Op creator object for math operations
     */
    public final SDMath math = new SDMath(this);
    /**
     * Op creator object for random number generation operations
     */
    public final SDRandom random = new SDRandom(this);
    /**
     * Op creator object for general neural network operations
     */
    public final SDNN nn = new SDNN(this);
    /**
     * Op creator object for convolutional neural network operations
     */
    public final SDCNN cnn = new SDCNN(this);
    /**
     * Op creator object for recurrent neural network operations
     */
    public final SDRNN rnn = new SDRNN(this);
    /**
     * Op creator object for loss function operations
     */
    public final SDLoss loss = new SDLoss(this);
    /**
     * Op creator object for image operations
     */
    public final SDImage image = new SDImage(this);

    /**
     * Op creator object for bitwise operations
     */
    public final SDBitwise bitwise = new SDBitwise(this);

    /**
     * Op creator object for math operations
     */
    public SDMath math() {
        return math;
    }

    /**
     * Op creator object for random number generation operations
     */
    public SDRandom random() {
        return random;
    }

    /**
     * Op creator object for general neural network operations
     */
    public SDNN nn() {
        return nn;
    }

    /**
     * Op creator object for convolutional neural network operations
     */
    public SDCNN cnn() {
        return cnn;
    }

    /**
     * Op creator object for recurrent neural network operations
     */
    public SDRNN rnn() {
        return rnn;
    }

    /**
     * Op creator object for loss function operations
     */
    public SDLoss loss() {
        return loss;
    }

    /**
     * Op creator object for image operations
     */
    public SDImage image() {
        return image;
    }

    /**
     * Op creator object for bitwise operations
     */
    public SDBitwise bitwise(){
        return bitwise;
    }

    private Map<String, SameDiff> sameDiffFunctionInstances;

    private Table<String, String, String> fieldVariableResolutionMapping;

    // flag, shows if graph was already registered with libnd4j
    private transient AtomicBoolean wasRegistered = new AtomicBoolean(false);


    //debug mode variables
    @Getter
    private boolean debugMode;

    @Getter
    private Stack<ArgumentInterceptor> argumentInterceptors = new Stack<>();
    @Getter
    private Set<ArgumentInterceptor> pausedArgumentInterceptors = new HashSet<>();

    private Set<String> blockNames = new HashSet<>();

    @Getter
    @Setter
    boolean logExecution = true;

    @Getter
    private SameDiff parent;

    @Getter
    private SameDiff child;


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
     * @return DifferentialFunctionFactory
     */
    public DifferentialFunctionFactory f() {
        return functionFactory;
    }

    /**
     * Set the current SameDiff-wide {@link Listener} instances.
     *
     * Note that this will overwrite the current listener list.
     * If you want to use additional listeners for a single operation,
     * use the listener arguments in those methods (e.g. {@link #fit()} and {@link FitConfig#listeners(Listener...)}).
     *
     * @param listeners Listeners
     */
    public void setListeners(Listener... listeners) {
        this.listeners.clear();
        addListeners(listeners);
    }

    /**
     * See {@link #setListeners(Listener...)}.
     */
    public void setListeners(Collection<? extends Listener> listeners) {
        this.listeners.clear();
        addListeners(listeners);
    }


    /**
     * Add SameDiff-wide {@link Listener} instances.
     *
     * If you want to use additional listeners for a single operation,
     * use the listener arguments in those methods (e.g. {@link #fit()} and {@link FitConfig#listeners(Listener...)}).
     *
     * @param listeners Listeners
     */
    public void addListeners(Listener... listeners) {
        addListeners(Arrays.asList(listeners));
    }

    /**
     * See {@link #addListeners(Listener...)}.
     */
    public void addListeners(Collection<? extends Listener> listeners) {
        this.listeners.addAll(listeners);
    }

    /**
     * Gets the current SameDiff-wide listeners.
     */
    public List<Listener> getListeners() {
        return listeners;
    }

    public void setArrayHolders(@NonNull ArrayHolder variableArrayHolder, @NonNull ArrayHolder constantArrayHolder, boolean initialize){
        if(initialize){
            variableArrayHolder.initFrom(this.variablesArrays);
            constantArrayHolder.initFrom(this.constantArrays);
        }
        this.variablesArrays = variableArrayHolder;
        this.constantArrays = constantArrayHolder;
    }

    /**
     * @return The current name scope, if any (null otherwise). See {@link #withNameScope(String)} for more details.
     */
    public String currentNameScope() {
        if (nameScopes.isEmpty())
            return null;

        //Would use String.join but that is Java 8+
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (NameScope ns : nameScopes) {
            if (!first) {
                sb.append("/");
            }
            sb.append(ns.getName());
            first = false;
        }
        return sb.toString();
    }

    /**
     * @return The name with the current name scope (if any) appended. See {@link #withNameScope(String)}
     */
    protected String nameWithScope(String name) {
        String scope = currentNameScope();
        if (scope == null) {
            return name;
        }
        if (!name.startsWith(scope + "/"))
            return scope + "/" + name;
        else
            return name;
    }

    //Intentionally package private
    void addNameScope(NameScope nameScope) {
        nameScopes.add(nameScope);
    }

    //Intentionally package private
    void closeNameScope(NameScope nameScope) {
        //Check that the name scope is closed correctly/in order
        Preconditions.checkState(!nameScopes.isEmpty(), "Cannot close name scope: no name scopes are currently defined");
        Preconditions.checkState(nameScopes.get(nameScopes.size() - 1).equals(nameScope),
                "Cannot close name scope %s: Name scopes must be closed in order. Current name scopes: \"%s\"", nameScope, currentNameScope());

        nameScopes.remove(nameScopes.size() - 1);
    }

    /**
     * Create a name scope. Name scopes append a prefix to the names of any variables and ops created while they are open.
     * <pre>
     *  {@code
     *  SameDiff sd = SameDiff.create();
     *  SDVariable x = sd.var("x", DataType.FLOAT, 5);
     *  SDVariable y;
     *  try(NameScope ns = sd.withNameScope("myScope"){
     *      y = sd.var("y", DataType.FLOAT, 5);
     *  }
     *  SDVariable z = sd.var("z", DataType.FLOAT, 5);
     *
     *  String xName = x.name();      //RESULT: "x"
     *  String yName = y.name();      //RESULT: "myScope/y"
     *  String zName = z.name();      //RESULT: "z"
     *  }
     * </pre>
     * <p>
     * Note that name scopes can also be nested:
     * <pre>
     *  {@code
     *  SameDiff sd = SameDiff.create();
     *  SDVariable x;
     *  try(NameScope ns = sd.withNameScope("first"){
     *      try(NameScope ns2 = sd.withNameScope("second"){
     *          x = sd.var("x", DataType.FLOAT, 5);
     *      }
     *  }
     *  String xName = x.name();      //RESULT: "first/second/x"
     *  }
     * </pre>
     *
     * @param nameScope Name of the name scope to open/create
     * @return The NameScope object
     */
    public NameScope withNameScope(String nameScope) {
        NameScope ns = new NameScope(this, nameScope);
        addNameScope(ns);
        return ns;
    }


    /**
     * Gets all operations in a given name scope.
     */
    public List<SameDiffOp> getOpsInScope(NameScope scope) {
        ArrayList<SameDiffOp> ops = new ArrayList<>();
        for (SameDiffOp v : this.ops.values()) {
            if (v.getName().startsWith(scope.getName()))
                ops.add(v);
        }
        return ops;
    }

    /**
     * See {@link #getOpsInScope(NameScope)}.
     */
    public List<SameDiffOp> getOpsInScope(String scope){
        return getOpsInScope(new NameScope(this, scope));
    }

    /**
     * Gets all variables in a given name scope.
     */
    public List<SDVariable> getVariablesInScope(NameScope scope) {
        ArrayList<SDVariable> vars = new ArrayList<>();
        for (SDVariable v : variables()) {
            if (v.name().startsWith(scope.getName()))
                vars.add(v);
        }
        return vars;
    }

    /**
     * See {@link #getVariablesInScope(NameScope)}.
     */
    public List<SDVariable> getVariablesInScope(String scope){
        return getVariablesInScope(new NameScope(this, scope));
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
            SDVariable clone = var.clone(this);
            SDVariable newVar = sameDiff.var(clone);
            if (var.getVariableType() != VariableType.ARRAY && var.getArr() != null ) {      //ARRAY type = "activations" - are overwritten anyway
                sameDiff.associateArrayWithVariable(var.getArr(), newVar);
            }


            thisVertexIdToNew.put(idx, idx);
            clone.setSameDiff(sameDiff);
            idx++;

        }

        Map<String,Integer> reverseMap = new HashMap<>();
        int count = 0;
        for( Variable v : variables.values()){
            reverseMap.put(v.getName(), count++);
        }

        val newFunctions = new LinkedHashMap<String, DifferentialFunction>();
        for (SameDiffOp op : ops.values()) {
            DifferentialFunction function = op.getOp();

            //Clone the op
            DifferentialFunction clone = FlatBuffersMapper.cloneViaSerialize(this, function, reverseMap);

            clone.setSameDiff(sameDiff);
            clone.setOwnName(function.getOwnName());
            if (sameDiff.opExists(function.getOwnName()))
                sameDiff.putOpForId(function.getOwnName(), function);
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
    public boolean opExists(String id) {
        return ops.containsKey(id);
    }

    /**
     * Get the differential function (if any) that this variable is the output for
     *
     * @param variableName Name of the variable
     * @return The differential function that this variable is an output of, or null if it is not the output of a function
     */
    public DifferentialFunction getVariableOutputOp(String variableName) {
        Preconditions.checkState(variables.containsKey(variableName), "No variable with name \"%s\" found in graph", variableName);
        if (variables.get(variableName).getOutputOfOp() == null)
            return null;
        return ops.get(variables.get(variableName).getOutputOfOp()).getOp();
    }

    /**
     * Get the function by the {@link DifferentialFunction#getOwnName()}
     *
     * @param id the id of the function
     * @return the function for the given id if it exists
     */
    public DifferentialFunction getOpById(@NonNull String id) {
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
    public void putOpForId(String id, DifferentialFunction function) {
        if (ops.containsKey(id) && ops.get(id).getOp() == null) {
            throw new ND4JIllegalStateException("Function by id already exists!");
        }

        if (!ops.containsKey(id)) {
            ops.put(id, SameDiffOp.builder().name(id).op(function).build());
        }
    }


    /**
     * Returns the name(s) of the inputs for the given function
     *
     * @param function the function to get the inputs for
     * @return the input ids for a given function
     */
    public String[] getInputsForOp(@NonNull DifferentialFunction function) {
        if (!ops.containsKey(function.getOwnName()))
            throw new ND4JIllegalStateException("Unknown function instance id found: \"" + function.getOwnName() + "\"");
        List<String> inputs = ops.get(function.getOwnName()).getInputsToOp();
        return inputs == null ? null : inputs.toArray(new String[inputs.size()]);
    }

    /**
     * Returns the name(s) of the outputs for the given function
     *
     * @param function the function to get the outputs for
     * @return the outputs ids for a given function
     */
    public String[] getOutputsForOp(DifferentialFunction function) {
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
    public SDVariable[] getOutputVariablesForOp(DifferentialFunction function) {
        val inputs = getOutputsForOp(function);
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
    public SDVariable[] getInputVariablesForOp(DifferentialFunction function) {
        val inputs = getInputsForOp(function);
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


    /**
     * Set the stored {@link INDArray} for a variable.  Only works if the variable is of type
     * {@link VariableType#CONSTANT}, {@link VariableType#PLACEHOLDER}, or {@link VariableType#VARIABLE}.
     */
    public void setArrayForVariable(@NonNull String varName, @NonNull INDArray arr) {
        Preconditions.checkState(variables.containsKey(varName), "No variable with name \"%s\" exists", varName);

        SDVariable v = getVariable(varName);
        if (v.isConstant()) {
            constantArrays.setArray(varName, arr);
        } else if (v.getVariableType() == VariableType.VARIABLE) {
            variablesArrays.setArray(varName, arr);
        } else if (v.isPlaceHolder()) {
            long tid = Thread.currentThread().getId();
            if (!placeholdersPerThread.containsKey(tid)) {
                placeholdersPerThread.put(tid, new HashMap<String, INDArray>());
            }
            placeholdersPerThread.get(tid).put(varName, arr);
        } else {
            throw new UnsupportedOperationException("Cannot set variable of type " + v.getVariableType() + " using this method");
        }
    }


    /**
     * Returns true if the given vertex id and {@link INDArray} already exist.
     *
     * @param varName the vertex id
     * @return true if a vertex with the given INDArray exists, and it has an INDArray associated with it
     */
    public boolean arrayAlreadyExistsForVarName(String varName) {
        SDVariable var = getVariable(varName);
        switch (var.getVariableType()) {
            case VARIABLE:
                return variablesArrays.hasArray(varName);
            case ARRAY:
                long tid = Thread.currentThread().getId();
                return sessions.containsKey(tid) && sessions.get(tid).contains(varName, InferenceSession.OUTER_FRAME, 0, null);
            case CONSTANT:
                return constantArrays.hasArray(varName);
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
        switch (v.getVariableType()) {
            case VARIABLE:
                return variablesArrays.getArray(varName);
            case CONSTANT:
                if (!constantArrays.hasArray(varName))
                    return null;
                return constantArrays.getArray(varName);
            case ARRAY:
                //Only stored in inference session...
                InferenceSession s = sessions.get(Thread.currentThread().getId());
                if (s == null)
                    return null;
                return s.get(varName, InferenceSession.OUTER_FRAME, 0, null, false);
            case PLACEHOLDER:
                long tid = Thread.currentThread().getId();
                if (placeholdersPerThread.get(tid) == null || !placeholdersPerThread.get(tid).containsKey(varName))
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
                variable.name(), variable.dataType(), arr.dataType());

        if (sessions.get(Thread.currentThread().getId()) == null) {
            sessions.put(Thread.currentThread().getId(), new InferenceSession(this));
        }

        if (arr.isAttached()) {
            arr = arr.detach();
        }

        switch (variable.getVariableType()) {
            case VARIABLE:
                variablesArrays.setArray(variable.name(), arr);
                break;
            case CONSTANT:
                constantArrays.setArray(variable.name(), arr);
                break;
            case ARRAY:
                throw new UnsupportedOperationException("Cannot associate array with SDVariable of type ARRAY - arrays for" +
                        " this type of variable is calculated ");
            case PLACEHOLDER:
                //Validate placeholder shapes:
                long[] phShape = variable.placeholderShape();
                Preconditions.checkState(phShape == null || Shape.shapeMatchesPlaceholder(phShape, arr.shape()),
                        "Invalid array shape: cannot associate an array with shape %ndShape with a placeholder of shape %s:" +
                                "shape is wrong rank or does not match on one or more dimensions", arr, phShape);


                long tid = Thread.currentThread().getId();
                if (!placeholdersPerThread.containsKey(tid)) {
                    placeholdersPerThread.put(tid, new HashMap<String, INDArray>());
                }
                placeholdersPerThread.get(tid).put(variable.name(), arr);
                break;
            default:
                throw new IllegalStateException("Unknown variable type: " + variable.getVariableType());
        }

        //putOrUpdateShapeForVarName(variable.name(), arr.shape(), true);

        //Also update nested SameDiff instances (such as gradient function)
        if (sameDiffFunctionInstances != null && sameDiffFunctionInstances.size() > 0) {
            for (Map.Entry<String, SameDiff> e : sameDiffFunctionInstances.entrySet()) {
                SameDiff sd = e.getValue();
                SDVariable v = sd.getVariable(variable.name());
                if (v != null) {
                    sd.associateArrayWithVariable(arr, v);
                }
            }
        }
    }

    /**
     * Update the constant or variable type SDVariable with the values from the specified
     * array. Note that unlike {@link #associateArrayWithVariable(INDArray, String)} this method will take the
     * values from the argument array and assign it to the current array.
     * The actual array (INDArray object) will not be stored or otherwise used within the SameDiff instance.
     * @param arr      Array values to set
     * @param variable Variable to update the array of. Must be CONSTANT or VARIBLE type SDVariable
     */
    public void assignArray(@NonNull INDArray arr, @NonNull SDVariable variable){
        Preconditions.checkState(variable.getVariableType() == VariableType.VARIABLE || variable.getVariableType() == VariableType.CONSTANT,
                "assignArray method can only be used with VARIBLE or CONSTANT type SDVariables, variable \"%s\" has type %s", variable.name(), variable.getVariableType());

        //DeviceLocal doesn't work with views
        if(arr.isView())
            arr = arr.dup();

        if(variable.getVariableType() == VariableType.VARIABLE ){
            variablesArrays.setArray(variable.name(), arr);
        } else {
            constantArrays.setArray(variable.name(), arr);
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
        Map<String, SDVariable> ret = new LinkedHashMap<>();
        for (Variable v : variables.values()) {
            ret.put(v.getName(), v.getVariable());
        }
        return ret;
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

    private SameDiff() {
        functionFactory = new DifferentialFunctionFactory(this);
        sameDiffFunctionInstances = new LinkedHashMap<>();
        fieldVariableResolutionMapping = HashBasedTable.create();
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
     * @param function  Differential function
     */
    public void addOutgoingFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            varNames[i] = variables[i].name();
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
     * Add a new argument interceptor to the interceptor stack
     * <p>
     * For internal use only.
     * <p>
     * When a op is added with arguments, most recent argument interceptor is called on it.
     * If ops are added in that interceptor, the next most recent will be called on their args, and so on.
     *
     * @param interceptor the argument interceptor to add
     */
    public void addArgumentInterceptor(@NonNull ArgumentInterceptor interceptor) {
        argumentInterceptors.push(interceptor);
    }

    private boolean isArgumentInterceptorPaused(@NonNull ArgumentInterceptor interceptor) {
        return pausedArgumentInterceptors.contains(interceptor);
    }

    private ArgumentInterceptor getArgumentInterceptorToUse() {

        if (argumentInterceptors.isEmpty())
            return null;

        ArgumentInterceptor use = argumentInterceptors.peek();
        int i = 1;
        while (isArgumentInterceptorPaused(use)) {
            if (argumentInterceptors.size() - i < 0)
                return null;

            use = argumentInterceptors.elementAt(argumentInterceptors.size() - i);
            i++;
        }

        return use;
    }

    /**
     * Remote the top (most recently added) argument interceptor
     * <p>
     * For internal use only.
     */
    public void removeArgumentInterceptor() {
        if (!argumentInterceptors.isEmpty())
            argumentInterceptors.pop();
    }

    /**
     * Pause the top (most recently added) argument interceptor
     * <p>
     * For internal use only.
     */
    public void pauseArgumentInterceptor() {
        pausedArgumentInterceptors.add(argumentInterceptors.peek());
    }

    /**
     * Pause the given argument interceptor
     * <p>
     * For internal use only.
     *
     * @param interceptor the argument interceptor to pause
     */
    public void pauseArgumentInterceptor(@NonNull ArgumentInterceptor interceptor) {
        pausedArgumentInterceptors.add(interceptor);
    }

    /**
     * Unpause the top (most recently added) argument interceptor
     * <p>
     * For internal use only.
     */
    public void unpauseArgumentInterceptor() {
        pausedArgumentInterceptors.remove(argumentInterceptors.peek());
    }

    /**
     * Unpause the top given argument interceptor
     * <p>
     * For internal use only.
     *
     * @param interceptor the argument interceptor to unpause
     */
    public void unpauseArgumentInterceptor(@NonNull ArgumentInterceptor interceptor) {
        pausedArgumentInterceptors.remove(interceptor);
    }

    /**
     * Adds incoming arguments for the specified differential function to the graph
     *
     * @param variables Name of the variables that are arguments (inputs) to the specified function
     * @param function  Function
     */
    public void addArgsFor(String[] variables, DifferentialFunction function) {

        ArgumentInterceptor interceptor = getArgumentInterceptorToUse();

        if (interceptor != null) {
            pauseArgumentInterceptor(interceptor);
            for (int i = 0; i < variables.length; i++) {
                variables[i] = interceptor.intercept(getVariable(variables[i])).name();
            }
            unpauseArgumentInterceptor(interceptor);
        }

        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        //Add function if it doesn't exist
        //TODO could "not existing" be a bug sometimes?
        if (!ops.containsKey(function.getOwnName())) {
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
            if (!funcs.contains(function.getOwnName()))  //Avoid duplicates for function names.
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
            varNames[i] = variables[i].name();
        }
        addArgsFor(varNames, function);
    }

    /**
     * Replaces the argument at i with newArg for function
     * Does not use (or remove) ArgumentInterceptor stuff
     */
    public void replaceArgFor(int i, @NonNull SDVariable newArg, @NonNull DifferentialFunction function) {

        Preconditions.checkArgument(i < function.args().length, "Index out of range: function " +
                function.getOwnName() + " only has " + function.args().length + " args but you are trying" +
                "to replace the argument at " + i);

        String oldName = function.arg(i).name();
        String newName = newArg.name();

        List<String> oldArgs = ops.get(function.getOwnName()).getInputsToOp();
        oldArgs = new ArrayList<>(oldArgs);
        oldArgs.set(i, newName);
        ops.get(function.getOwnName()).setInputsToOp(oldArgs);

        List<String> funcs = this.variables.get(newName).getInputsForOp();

        if (funcs == null) {
            funcs = new ArrayList<>();
            this.variables.get(newName).setInputsForOp(funcs);
        }
        if (!funcs.contains(function.getOwnName()))  //Avoid duplicates for function names.
            funcs.add(function.getOwnName());

        List<String> oldFuncs = this.variables.get(oldName).getInputsForOp();
        if (oldFuncs != null) {
            if (!ArrayUtils.contains(function.argNames(), oldName))
                oldFuncs.remove(function.getOwnName());
        }

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
     * Clear the placeholder arrays from the SameDiff instance
     *
     * @param allThreads If true: clear the placeholders for all threads. False: clear only for current thread
     */
    public void clearPlaceholders(boolean allThreads) {
        if (allThreads) {
            this.placeholdersPerThread.clear();
        } else {
            long tid = Thread.currentThread().getId();
            this.placeholdersPerThread.remove(tid);
        }
        for (SameDiff sd : this.sameDiffFunctionInstances.values()) {
            sd.clearPlaceholders(allThreads);
        }
    }

    /**
     * Clear the input arrays to each op.
     * This is usually not required, under normal SameDiff use
     */
    public void clearOpInputs() {
        for (SameDiffOp op : ops.values()) {
            if (op.getOp() instanceof Op) {
                Op o = ((Op) op.getOp());
                o.setX(null);
                if (o.y() != null) {
                    o.setY(null);
                }
            } else if (op.getOp() instanceof DynamicCustomOp) {
                DynamicCustomOp o = (DynamicCustomOp) op.getOp();
                o.setInputArguments((INDArray[]) null);
            }
        }
        for (SameDiff sd : this.sameDiffFunctionInstances.values()) {
            sd.clearOpInputs();
        }
    }

    /**
     * Get an array of differential functions that have been defined for this SameDiff instance
     *
     * @return Array of differential functions
     */
    public DifferentialFunction[] ops() {
        List<DifferentialFunction> out = new ArrayList<>(ops.size());
        for (SameDiffOp op : ops.values()) {
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
        return sameDiffFunctionInstances != null ? sameDiffFunctionInstances.equals(sameDiff.sameDiffFunctionInstances) : sameDiff.sameDiffFunctionInstances == null;
    }

    /**
     * Create a new (empty) SameDiff instance without any functions or variables
     *
     * @return New SameDiff instance
     */
    public static SameDiff create() {
        return new SameDiff();
    }

    /**
     * Clone/duplicate the SameDiff instance, including arrays etc. The returned SameDiff instance should have no
     * shared state with the original instance
     *
     * @return The cloned SameDiff instance
     */
    public SameDiff dup() {
        ByteBuffer bb = asFlatBuffers(true);
        try {
            return fromFlatBuffers(bb);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }


    /**
     * Count the number of elements in all arrays, according to {@link SDVariable#getShape()}
     *
     * @return Number of array elements for all variables
     */
    public long numElements() {
        long ret = 0;
        for (SDVariable variable : variables()) {
            long[] shape = variable.getShape();
            if (shape != null) {
                ret += ArrayUtil.prod(shape);
            }
        }
        return ret;
    }

    /**
     * Returns the inputs (placeholders) for the SameDiff graph
     *
     * @return the inputs for this graph
     */
    public List<String> inputs() {
        List<String> out = new ArrayList<>();
        for (String s : variables.keySet()) {
            if (isPlaceHolder(s))
                out.add(s);
        }
        return out;
    }

    /**
     * Outputs are the names of the predictions of the network.
     * Note that the outputs must be set using {@link #setOutputs(List)} first
     *
     * @return The outputs of the SameDiff instance, or null if no outputs have been set
     */
    public List<String> outputs() {
        return this.outputs;
    }

    /**
     * See {@link #setOutputs(List)}
     */
    public void setOutputs(String... outputs){
        setOutputs(outputs == null ? null : Arrays.asList(outputs));
    }


    /**
     * Set the outputs of the SameDiff instance.
     * Outputs are the names of the variables that are the predictions of the neural network.
     * Note that this is merely a convenience, and does not impact execution at all. Outputs can be retrieved (after
     * setting here) using {@link #outputs()}
     * @param outputs Outputs to set. Must be valid variable names in this SameDiff instance
     */
    public void setOutputs(List<String> outputs){
        if(outputs != null){
            for(String s : outputs){
                Preconditions.checkArgument(variables.containsKey(s), "Cannot set variable \"%s\" as an output: SameDiff instance does not contain a variable with this name");
            }
        }
        this.outputs = outputs;
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
    public List<String> getLossVariables() {
        return Collections.unmodifiableList(this.lossVariables);
    }

    /**
     * Clear/remove any existing loss variables, and set the loss variables to the specified variable names.<br>
     * See {@link #addLossVariable(String)} for more details
     *
     * @param lossVariableNames Names of variables to be loss function variables
     */
    public void setLossVariables(@NonNull String... lossVariableNames) {
        this.lossVariables.clear();
        for (String s : lossVariableNames) {
            addLossVariable(s);
        }
        //After changing loss function variables, we (probably) need to recreate gradient function - as gradient
        // function is defined with respect to specific loss function variables
        sameDiffFunctionInstances.remove("grad");
    }

    /**
     * See {@link #setLossVariables(String...)}
     */
    public void setLossVariables(@NonNull SDVariable... lossVariables) {
        String[] varNames = new String[lossVariables.length];
        for (int i = 0; i < lossVariables.length; i++)
            varNames[i] = lossVariables[i].name();

        setLossVariables(varNames);
    }

    /**
     * Mark the specified variable as a loss function variable. This means that this variable will be minimized via backprop during training.<br>
     * This will add the variable as a loss to any others - i.e., if multiple variables are marked as losses, their values will be summed
     * to give the total network loss.<br>
     * Note that only floating point (Float16/32/64) variables may be marked as a loss.<br>
     * Note also that only ARRAY type SDVariables can be marked as losses to be minimized. That is, we cannot mark the value
     * of a constant, variable or placeholder to be minimized as doing so would not make sense.<br>
     */
    public void addLossVariable(@NonNull String variableName) {
        Preconditions.checkState(hasVariable(variableName), "No variable with name \"%s\" exists", variableName);
        SDVariable v = getVariable(variableName);
        Preconditions.checkState(v.dataType().isFPType(), "Only floating point type variables can be marked as losses to be minimized." +
                " SDVariable \"%s\" has datatype %s", variableName, v.dataType());
        Preconditions.checkState(v.getVariableType() == VariableType.ARRAY, "Only ARRAY type SDVariables can be marked as losses to be minimized." +
                " SDVariable \"%s\" has variable type %s", variableName, v.getVariableType());
        if (!lossVariables.contains(variableName)) {
            lossVariables.add(variableName);
        }
    }

    /**
     * See {@link #addLossVariable(String)}
     */
    public void addLossVariable(@NonNull SDVariable variable) {
        addLossVariable(variable.name());
    }

    /**
     * Set the training configuration ({@link TrainingConfig}) for the SameDiff instance.
     * A TrainingConfig must be set before the SameDiff instance can be trained via the fit methods
     *
     * @param trainingConfig Training configuration
     */
    public void setTrainingConfig(TrainingConfig trainingConfig) {
        this.trainingConfig = trainingConfig;
    }

    /**
     * Fit the SameDiff instance based on a single DataSet (i.e., a single minibatch for one iteration).<br>
     * This method can only be used for singe input, single output SameDiff instances as DataSet only supports a
     * single input and a single output.<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param dataSet   The DataSet (single minibatch) to peform training on
     * @param listeners Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull DataSet dataSet, @NonNull Listener... listeners) {
        return fit(new SingletonMultiDataSetIterator(dataSet.toMultiDataSet()), 1, false,
                null, 1, listeners);
    }

    /**
     * Fit the SameDiff instance based on a single MultiDataSet (i.e., a single minibatch for one iteration).<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     *
     * @param dataSet   The MultiDataSet (single minibatch) to peform training on
     * @param listeners Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull MultiDataSet dataSet, @NonNull Listener... listeners) {
        return fit(new SingletonMultiDataSetIterator(dataSet), 1, false,
                null, 1, listeners);
    }

    /**
     * Fit the SameDiff instance based on DataSetIterator for the specified number of epochs.<br>
     * This method can only be used for singe input, single output SameDiff instances as DataSet only supports a
     * single input and a single output.<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     * <p>
     * A special case of {@link #fit()}.
     *
     * @param iter                The iterator to train the SameDiff instance with
     * @param numEpochs           The number of epochs for training. Must be > 0
     * @param validationIter      The DataSetIterator to use for validation (null to skip validation)
     * @param validationFrequency The frequency with which to run validation.  1 is every epoch, 2 is every other, etc.
     * @param listeners           Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull DataSetIterator iter, int numEpochs, DataSetIterator validationIter, int validationFrequency, @NonNull Listener... listeners) {
        return fit().train(iter, numEpochs).validate(validationIter, validationFrequency).listeners(listeners).exec();
    }

    /**
     * See {@link #fit(DataSetIterator, int, DataSetIterator, int, Listener...)}, does not preform validation.
     * <p>
     * A special case of {@link #fit()}.
     *
     * @param iter      The iterator to train the SameDiff instance with
     * @param numEpochs The number of epochs for training. Must be > 0
     * @param listeners Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull DataSetIterator iter, int numEpochs, @NonNull Listener... listeners) {
        return fit().train(iter, numEpochs).listeners(listeners).exec();
    }

    /**
     * Fit the SameDiff instance based on MultiDataSetIterator for the specified number of epochs.<br>
     * This method can both singe input, single output and multi-input, multi-output SameDiff instances<br>
     * Note that a {@link TrainingConfig} must be set via {@link #setTrainingConfig(TrainingConfig)} before training can
     * be performed.
     * <p>
     * A special case of {@link #fit()}.
     *
     * @param iter                The iterator to train the SameDiff instance with
     * @param numEpochs           The number of epochs for training. Must be > 0
     * @param validationIter      The MultiDataSetIterator to use for validation (null to skip validation)
     * @param validationFrequency The frequency with which to run validation.  1 is every epoch, 2 is every other, etc.
     * @param listeners           Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull MultiDataSetIterator iter, int numEpochs, MultiDataSetIterator validationIter, int validationFrequency, @NonNull Listener... listeners) {
        return fit(iter, numEpochs, true, validationIter, validationFrequency, listeners);
    }

    /**
     * See {@link #fit(MultiDataSetIterator, int, MultiDataSetIterator, int, Listener...)}, does not preform validation.
     * <p>
     * A special case of {@link #fit()}.
     *
     * @param iter      The iterator to train the SameDiff instance with
     * @param numEpochs The number of epochs for training. Must be > 0
     * @param listeners Additional listeners to use during this operation
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History fit(@NonNull MultiDataSetIterator iter, int numEpochs, @NonNull Listener... listeners) {
        return fit().train(iter, numEpochs).listeners(listeners).exec();
    }

    /**
     * Set up for a fit operation using {@link FitConfig}.
     * <p>
     * Supports the setting of training data ({@link MultiDataSetIterator} or {@link DataSetIterator}), number of epochs,
     * validation data ({@link MultiDataSetIterator} or {@link DataSetIterator}), validation frequency, and additional listeners.
     * <br><br>
     * Example: train on data for 5 epochs, validating on valData every 2nd epoch
     * <pre>
     *     {@code
     *     SameDiff sd = ...;
     *     MultiDataSet data = ...;
     *     MultiDataSet valData = ...;
     *
     *     History hist = sd.fit()
     *         .train(data, 5)
     *         .validate(valData, 2)
     *         .exec();
     *     }
     * </pre>
     */
    public FitConfig fit() {
        return new FitConfig(this);
    }

    //Synchronized for thread safety
    protected synchronized History fit(@NonNull MultiDataSetIterator iter, int numEpochs, boolean incrementEpochCount,
                                       MultiDataSetIterator validationData, int validationFrequency, @NonNull Listener... listeners) {
        boolean async = iter.asyncSupported();

        boolean validationAsync = false;
        if (validationData != null)
            validationAsync = validationData.asyncSupported();

        if (async) {
            iter = new AsyncMultiDataSetIterator(iter, 3, true);
        }

        if (validationAsync) {
            validationData = new AsyncMultiDataSetIterator(validationData, 3, true);
        }

        try {
            return fitHelper(iter, numEpochs, incrementEpochCount, validationData, validationFrequency, Arrays.asList(listeners));
        } finally {
            if (async) {
                ((AsyncMultiDataSetIterator) iter).shutdown();
            }
            if (validationAsync) {
                ((AsyncMultiDataSetIterator) validationData).shutdown();
            }
        }
    }

    //fitHelper should only be called from fit method above
    protected synchronized History fitHelper(@NonNull MultiDataSetIterator iter, int numEpochs, boolean incrementEpochCount,
                                             MultiDataSetIterator validationData, int validationFrequency, @NonNull List<Listener> listeners) {
        Preconditions.checkNotNull(iter, "Iterator must not be null");
        Preconditions.checkState(numEpochs > 0, "Number of training epochs must be a positive number. Got: %s", numEpochs);
        Preconditions.checkState(trainingConfig != null, "No training configuration has been set. A training configuration must " +
                "be set before training. Use setTrainingConfig(TrainingConfig)");
        Preconditions.checkState(numEpochs == 1 || iter.resetSupported(), "Cannot train for multiple epochs on an iterator that" +
                " does not support resetting");

        HistoryListener history = new HistoryListener(trainingConfig);

        List<Listener> activeListeners = new ArrayList<>();

        activeListeners.add(history);

        for (Listener l : this.listeners)
            if (l.isActive(Operation.TRAINING))
                activeListeners.add(l);

        for (Listener l : listeners)
            if (l.isActive(Operation.TRAINING))
                activeListeners.add(l);

        validateListenerActivations(activeListeners, Operation.TRAINING);
        validateListenerActivations(activeListeners, Operation.TRAINING_VALIDATION);

        if (!iter.hasNext() && iter.resetSupported())
            iter.reset();

        boolean performedValidation = false;

        int trainThreadNum = 0;
        long jThreadId = Thread.currentThread().getId();
        boolean hasListeners = !activeListeners.isEmpty();
        At at = At.builder()
                .epoch(trainingConfig.getEpochCount())
                .iteration(trainingConfig.getIterationCount())
                .trainingThreadNum(trainThreadNum)
                .javaThreadNum(jThreadId)
                .operation(Operation.TRAINING)
                .build();

        LossCurve lossCurve = null;


        Set<String> requiredVars = new HashSet<>();
        for (Listener l : activeListeners) {
            ListenerVariables lv = l.requiredVariables(this);
            if(lv != null) {
                Set<String> s = lv.trainingVariables();
                if(s != null) {
                    requiredVars.addAll(s);
                }
            }
        }

        List<Listener> listenersWitHistory = new ArrayList<>(listeners);
        for(Listener l : this.listeners){
            if(!listenersWitHistory.contains(l))
                listenersWitHistory.add(l);
        }
        listenersWitHistory.add(history);


        SameDiff gradInstance = getFunction("grad");
        if(gradInstance == null){
            createGradFunction();
            gradInstance = getFunction("grad");
        }
        TrainingSession ts = new TrainingSession(gradInstance);
        gradInstance.setTrainingConfig(trainingConfig);     //In case any listeners want to use it

        for(Listener l : activeListeners){
            l.operationStart(gradInstance, Operation.TRAINING);
        }

        Set<String> paramsToTrain = new LinkedHashSet<>();
        for(Variable v : variables.values()){
            if(v.getVariable().getVariableType() == VariableType.VARIABLE){
                //TODO not all variable type are needed - i.e., variable that doesn't impact loss should be skipped
                paramsToTrain.add(v.getName());
            }
        }

        Loss lastLoss = null;
        for (int i = 0; i < numEpochs; i++) {
            if (incrementEpochCount && hasListeners) {
                at.setEpoch(trainingConfig.getEpochCount());
                for (Listener l : activeListeners) {
                    l.epochStart(this, at);
                }
            }
            long epochStartTime = System.currentTimeMillis();

            double[] lossSums = null;
            List<String> lossNames = null;
            int lossCount = 0;

            while (iter.hasNext()) {
                long dataStart = hasListeners ? System.currentTimeMillis() : 0;
                org.nd4j.linalg.dataset.api.MultiDataSet ds = iter.next();

                long dataEnd = hasListeners ? System.currentTimeMillis() : 0;
                if (!performedValidation) {
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

                if (hasListeners) {
                    at.setIteration(trainingConfig.getIterationCount());
                    for (Listener l : activeListeners) {
                        l.iterationStart(this, at, ds, (dataEnd - dataStart));
                    }
                }

                //Create placeholder variable map
                Map<String, INDArray> placeholders = toPlaceholderMap(ds);

                Preconditions.checkState(placeholders.size() > 0, "No placeholder variables were set for training");

                //Call TrainingSession to perform training
                if (!initializedTraining)
                    initializeTraining();

                lastLoss = ts.trainingIteration(
                        trainingConfig,
                        placeholders,
                        paramsToTrain,
                        updaterMap,
                        ds,
                        getLossVariables(),
                        listenersWitHistory,
                        at);


                if (lossSums == null) {
                    lossSums = lastLoss.getLosses().clone();
                } else {
                    for (int j = 0; j < lossSums.length; j++) {
                        lossSums[j] += lastLoss.getLosses()[j];
                    }
                }
                lossCount++;

                trainingConfig.incrementIterationCount();
            }

            long epochTime = System.currentTimeMillis() - epochStartTime;

            if (incrementEpochCount) {
                lossNames = lastLoss.getLossNames();

                for (int j = 0; j < lossSums.length; j++)
                    lossSums[j] /= lossCount;

                if (lossCurve != null)
                    lossCurve = lossCurve.addLossAndCopy(lossSums, lossNames);
                else
                    lossCurve = new LossCurve(lossSums, lossNames);
            }


            if (incrementEpochCount) {
                if (hasListeners) {
                    boolean doStop = false;
                    Listener stopped = null;

                    for (Listener l : activeListeners) {
                        ListenerResponse res = l.epochEnd(this, at, lossCurve, epochTime);

                        if (res == ListenerResponse.STOP && (i < numEpochs - 1)) {
                            doStop = true;
                            stopped = l;
                        }
                    }

                    if (doStop) {
                        log.info("Stopping training early.  Listener " + stopped + " gave a STOP signal at epoch " + at.epoch() + " and iteration " + at.iteration());

                        for (Listener l1 : activeListeners)
                            l1.operationEnd(this, Operation.TRAINING);

                        if (i < numEpochs - 1) {
                            iter.reset();
                        }

                        if (incrementEpochCount)
                            trainingConfig.incrementEpochCount();
                        return history.getReport();
                    }


                    //validation evaluation
                    if (validationData != null && (validationFrequency <= 0 || i % validationFrequency == 0)) {

                        long validationStart = System.currentTimeMillis();
                        outputHelper(validationData, new At(at.epoch(), 0, 0, 0, Operation.TRAINING_VALIDATION),
                                listenersWitHistory);

                        long validationTime = System.currentTimeMillis() - validationStart;

                        boolean doStopV = false;
                        Listener stoppedV = null;
                        for (Listener l : activeListeners) {

                            ListenerResponse res = l.validationDone(this, at, validationTime);

                            if (res == ListenerResponse.STOP && (i < numEpochs - 1)) {
                                doStopV = true;
                                stoppedV = l;
                            }
                        }

                        if (doStopV) {
                            log.info("Stopping training early from validation.  Listener " + stoppedV + " gave a STOP signal at epoch " + at.epoch() + " and iteration " + at.iteration());

                            for (Listener l1 : activeListeners)
                                l1.operationEnd(this, Operation.TRAINING);

                            if (i < numEpochs - 1) {
                                iter.reset();
                            }

                            if (incrementEpochCount)
                                trainingConfig.incrementEpochCount();

                            return history.getReport();
                        }

                    }

                }

                trainingConfig.incrementEpochCount();
            }
            if (i < numEpochs - 1) {
                iter.reset();
            }
        }

        for (Listener l1 : activeListeners)
            l1.operationEnd(this, Operation.TRAINING);

        return history.getReport();
    }

    /**
     * Ensure the specified listeners do not request any activations that aren't present for the given operation
     */
    private void validateListenerActivations(List<Listener> listeners, Operation op) {
        for (Listener l : listeners) {
            ListenerVariables lv = l.requiredVariables(this);
            if(lv != null) {
                for (String s : lv.requiredVariables(op)) {
                    if (!variables.containsKey(s)) {
                        Preconditions.checkState(false, "Listener %s requested variable %s that is not defined in this SameDiff graph", l, s);
                    }
                }
            }
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

        if (trainingConfig.getRegularization() == null || trainingConfig.getRegularization().isEmpty()) {
            return 0.0;
        }

        List<Regularization> l = trainingConfig.getRegularization();
        double loss = 0.0;
        for (Variable v : variables.values()) {
            SDVariable sdv = v.getVariable();
            if (sdv.getVariableType() != VariableType.VARIABLE || !sdv.dataType().isFPType()) {
                //Only trainable parameters (FP and variable type vars) contribute to regularization score
                continue;
            }
            for (Regularization r : l) {
                INDArray arr = sdv.getArr();
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
    protected void initializeTraining() {
        if (!initializedTraining) {
            if (trainingConfig == null) {
                throw new ND4JIllegalStateException("Please specify a training config with setTrainingConfig");
            }
            updaterMap = new HashMap<>();
            for (Variable v : variables.values()) {
                if (v.getVariable().getVariableType() != VariableType.VARIABLE || !v.getVariable().dataType().isFPType()) {
                    //Skip non-trainable parameters
                    continue;
                }

                INDArray arr = v.getVariable().getArr();
                long stateSize = trainingConfig.getUpdater().stateSize(arr.length());
                INDArray view = stateSize == 0 ? null : Nd4j.createUninitialized(arr.dataType(), 1, stateSize);
                GradientUpdater gu = trainingConfig.getUpdater().instantiate(view, false);
                gu.setStateViewArray(view, arr.shape(), arr.ordering(), true);
                updaterMap.put(v.getName(), gu);
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
    private Map<String, INDArray> toPlaceholderMap(org.nd4j.linalg.dataset.api.MultiDataSet ds) {
        Map<String, INDArray> placeholders = new HashMap<>();
        int count = 0;
        for (String s : trainingConfig.getDataSetFeatureMapping()) {
            placeholders.put(s, ds.getFeatures(count++));
        }
        count = 0;
        if (trainingConfig.getDataSetLabelMapping() != null) {
            //Labels may be null in some models (unsupervised etc)
            for (String s : trainingConfig.getDataSetLabelMapping()) {
                placeholders.put(s, ds.getLabels(count++));
            }
        }

        if (trainingConfig.getDataSetFeatureMaskMapping() != null && trainingConfig.getDataSetFeatureMaskMapping().size() > 0) {
            count = 0;
            for (String s : trainingConfig.getDataSetFeatureMaskMapping()) {
                if (s == null) {
                    count++;
                    continue;
                }
                placeholders.put(s, ds.getFeaturesMaskArray(count++));
            }
        }

        if (trainingConfig.getDataSetLabelMaskMapping() != null && trainingConfig.getDataSetLabelMaskMapping().size() > 0) {
            count = 0;
            for (String s : trainingConfig.getDataSetLabelMaskMapping()) {
                if (s == null) {
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
     * <p>
     * A special case of {@link #evaluate()}.
     *
     * @param iterator       Iterator as source of data to evaluate
     * @param outputVariable The variable to evaluate
     * @param listeners      Additional listeners to use during this operation.
     * @param evaluations    The evaluations to perform
     */
    public void evaluate(@NonNull DataSetIterator iterator, @NonNull String outputVariable, @NonNull List<Listener> listeners, @NonNull IEvaluation... evaluations) {
        Preconditions.checkArgument(evaluations != null && evaluations.length > 0, "No evaluations were passed to the evaluate method");

        evaluate().data(iterator).evaluate(outputVariable, evaluations).listeners(listeners.toArray(new Listener[0])).exec();
    }

    /**
     * See {@link #evaluate(DataSetIterator, String, List, IEvaluation[])}.
     * <p>
     * A special case of {@link #evaluate()}.
     */
    public void evaluate(@NonNull DataSetIterator iterator, @NonNull String outputVariable, @NonNull IEvaluation... evaluations) {
        evaluate().data(iterator).evaluate(outputVariable, evaluations).exec();
    }

    /**
     * Evaluation for multiple-output networks.<br>
     * See {@link #evaluate(MultiDataSetIterator, Map, Map, Listener[])}.
     * <p>
     * A special case of {@link #evaluate()}.
     */
    public void evaluate(@NonNull DataSetIterator iterator, @NonNull Map<String, IEvaluation> variableEvals, @NonNull Listener... listeners) {
        Map<String, Integer> map = new HashMap<>();
        Map<String, List<IEvaluation>> variableEvalsList = new HashMap<>();
        for (String s : variableEvals.keySet()) {
            map.put(s, 0);  //Only 1 possible output here with DataSetIterator
            variableEvalsList.put(s, Collections.singletonList(variableEvals.get(s)));
        }
        evaluate(new MultiDataSetIteratorAdapter(iterator), variableEvalsList, map, listeners);
    }

    /**
     * Evaluation for multiple output networks - one or more.
     * See {@link #evaluate(MultiDataSetIterator, Map, Map, Listener[])}.
     * <p>
     * A special case of {@link #evaluate()}.
     */
    public void evaluateMultiple(DataSetIterator iterator, Map<String, List<IEvaluation>> variableEvals, @NonNull Listener... listeners) {
        Map<String, Integer> map = new HashMap<>();
        for (String s : variableEvals.keySet()) {
            map.put(s, 0);  //Only 1 possible output here with DataSetIterator
        }
        evaluate(new MultiDataSetIteratorAdapter(iterator), variableEvals, map, listeners);
    }

    /**
     * Evaluate the performance of a single variable's prediction.<br>
     * For example, if the variable to evaluatate was called "softmax" you would use:
     * <pre>
     * {@code Evaluation e = new Evaluation();
     * sameDiff.evaluate(iterator, "softmax", e);}
     * </pre>
     * <p>
     * A special case of {@link #evaluate()}.
     *
     * @param iterator       Iterator as source of data to evaluate
     * @param outputVariable The variable to evaluate
     * @param labelIndex     The index of the target variable's labels in the iterator
     * @param listeners      Additional listeners to use during this operation.
     * @param evaluations    The evaluations to perform
     */
    public void evaluate(@NonNull MultiDataSetIterator iterator, @NonNull String outputVariable, int labelIndex,
                         @NonNull List<Listener> listeners, @NonNull IEvaluation... evaluations) {
        Preconditions.checkArgument(evaluations != null && evaluations.length > 0, "No evaluations were passed to the evaluate method");

        evaluate().data(iterator).evaluate(outputVariable, labelIndex, evaluations).listeners(listeners.toArray(new Listener[0])).exec();
    }

    /**
     * See {@link #evaluate(MultiDataSetIterator, String, int, List, IEvaluation[])}.
     * <p>
     * A special case of {@link #evaluate()}.
     */
    public void evaluate(@NonNull MultiDataSetIterator iterator, @NonNull String outputVariable, int labelIndex, @NonNull IEvaluation... evaluations) {
        evaluate().data(iterator).evaluate(outputVariable, labelIndex, evaluations).exec();
    }

    /**
     * Perform evaluation using classes such as {@link Evaluation} for classifier outputs
     * and {@link org.nd4j.evaluation.regression.RegressionEvaluation} for regression outputs.<br>
     * <br>
     * <b>Example: classifier evaluation</b><br>
     * Predictions variable name: "softmaxOutput"<br>
     * Evaluations to perform: {@link Evaluation}<br>
     * Data: single input, single output MultiDataSets<br>
     * Code:<br>
     * <pre>
     * {@code
     * MultiDataSetIterator data = ...
     * Map<String,List<IEvaluation>> evals = Collections.singletonMap("softmaxOutput",Collections.singletonList(new Evaluation()));
     * Map<String,Integer> labelMapping = Collections.singletonMap("softmaxOutput",0);  //Compare: "softmaxOutput" vs. MultiDataSet.getLabels(0)
     * }
     * </pre>
     * <p>
     * A special case of {@link #evaluate()}.
     *
     * @param iterator               The iterator - the source of the data for evaluation
     * @param variableEvals          The evaluations to perform. Key: the name of the variable. Value: the evaluations to perform
     * @param predictionLabelMapping The output/label mapping. Key: the name of the variable.
     * @param listeners              Additional listeners to use during this operation.
     */
    public void evaluate(MultiDataSetIterator iterator, Map<String, List<IEvaluation>> variableEvals, Map<String, Integer> predictionLabelMapping, Listener... listeners) {
        evaluateHelper(iterator, variableEvals, predictionLabelMapping, At.defaultAt(Operation.EVALUATION), listeners);
    }


    /**
     * Set up for a evaluation operation using EvaluationConfig.
     * <p>
     * Supports the setting of the data ({@link MultiDataSetIterator} or {@link DataSetIterator}),
     * adding evaluations for variables (with optional label index setting), setting label indices,
     * and setting additional listeners.
     * Does not require setting label indices when using a {@link DataSetIterator}.
     * <p>
     * Also supports using {@link SDVariable} instances instead of variable names.
     *
     * <br><br>
     * Example: evaluate "pred" with {@link Evaluation} and {@link ROC}, using label 0.
     * <pre>
     *      {@code
     *     SameDiff sd = ...;
     *     MultiDataSetIterator data = ...;
     *
     *     EvaluationRecord results = sd.evaluate()
     *         .data(data)
     *         .evaluate("pred", 0, new Evaluation(), new ROC()),
     *         .exec();
     *      }
     *  </pre>
     * Example: evaluate "pred" with {@link Evaluation}, using the only label from a DataSetIterator.
     * <pre>
     *      {@code
     *     SameDiff sd = ...;
     *     DataSetIterator singleData = ...;
     *
     *     EvaluationRecord results = sd.evaluate()
     *         .data(singleData)
     *         .evaluate("pred", new Evaluation()),
     *         .exec();
     *      }
     *  </pre>
     */
    public EvaluationConfig evaluate() {
        return new EvaluationConfig(this);
    }

    /**
     * Helper method for evaluations.  Should only be called from the above evaluate method
     */
    private void evaluateHelper(MultiDataSetIterator iterator,
                                Map<String, List<IEvaluation>> variableEvals, Map<String, Integer> predictionLabelMapping, At at, @NonNull Listener... listeners) {
        Preconditions.checkState(trainingConfig != null, "Training config has not been set");

        Preconditions.checkState(variableEvals.keySet().equals(predictionLabelMapping.keySet()), "Keysets for variable evaluations" +
                " and for the prediction label mapping must be equal. Keys for variables to evaluate: %s vs. keys for label mapping: %s", variableEvals.keySet(), predictionLabelMapping.keySet());

        List<Listener> activeListeners = new ArrayList<>();

        for (Listener l : listeners)
            if (l.isActive(at.operation()))
                activeListeners.add(l);

        for (Listener l : this.listeners)
            if (l.isActive(at.operation()))
                activeListeners.add(l);

        validateListenerActivations(activeListeners, at.operation());

        for (Listener l : activeListeners)
            l.operationStart(this, at.operation());

        boolean hasListeners = !activeListeners.isEmpty();

        if (!iterator.hasNext() && iterator.resetSupported())
            iterator.reset();
        Set<String> requiredVars = new HashSet<>(variableEvals.keySet());

        if (hasListeners) {
            for (Listener l : activeListeners) {
                ListenerVariables v = l.requiredVariables(this);
                if(v != null) {
                    requiredVars.addAll(v.evaluationVariables());
                }
            }
        }

        String[] requiredVarsArr = requiredVars.toArray(new String[0]);

        while (iterator.hasNext()) {
            MultiDataSet ds = iterator.next();
            Map<String, INDArray> placeholderMap = toPlaceholderMap(ds);

            Map<String, INDArray> m = directExecHelper(placeholderMap, at, ds, Collections.<String>emptyList(), activeListeners, requiredVarsArr);

            for (Map.Entry<String, List<IEvaluation>> e : variableEvals.entrySet()) {
                INDArray prediction = m.get(e.getKey());
                for (IEvaluation eval : e.getValue()) {
                    //TODO time series, etc

                    INDArray label = ds.getLabels(predictionLabelMapping.get(e.getKey()));
                    INDArray mask = ds.getLabelsMaskArray(predictionLabelMapping.get(e.getKey()));
                    eval.eval(label, prediction, mask);
                }
            }

            at.setIteration(at.iteration() + 1);
        }


        for (Listener l : activeListeners)
            l.operationEnd(this, at.operation());
    }

    /**
     * Do a single batch inference on a network with a single input.<br>
     * For example, if the variable to infer was called "softmax" you would use:
     * <pre>
     * {@code
     * sameDiff.output(iterator, "softmax");}
     * </pre>
     *
     * @param dataSet The data to evaluate
     * @param outputs The variables to evaluate
     */
    public Map<String, INDArray> output(@NonNull DataSet dataSet, @NonNull String... outputs) {
        return outputBatches(new SingletonMultiDataSetIterator(dataSet.toMultiDataSet()), outputs).get(0);
    }

    /**
     * Do a single batch inference on a network.<br>
     * For example, if the variable to infer was called "softmax" you would use:
     * <pre>
     * {@code
     * sameDiff.output(iterator, "softmax");}
     * </pre>
     *
     * @param dataSet The data to evaluate
     * @param outputs The variables to evaluate
     */
    public Map<String, INDArray> output(@NonNull MultiDataSet dataSet, @NonNull String... outputs) {
        return outputBatches(new SingletonMultiDataSetIterator(dataSet), outputs).get(0);
    }

    /**
     * Do inference on a network with a single input.<br>
     * For example, if the variable to infer was called "softmax" you would use:
     * <pre>
     * {@code
     * sameDiff.output(iterator, "softmax");}
     * </pre>
     * <p>
     * Uses concatenation on the outputs of {@link #outputBatches(DataSetIterator, String...)} which may cause issues with some inputs.
     * RNNs with variable time series length and CNNs with variable image sizes will most likely have issues.
     * <p>
     * Special case of {@link #output()}.
     *
     * @param iterator  Iterator as source of data to evaluate
     * @param listeners Additional listeners to use during this operation.
     * @param outputs   The variables to evaluate
     */
    public Map<String, INDArray> output(@NonNull DataSetIterator iterator, @NonNull List<Listener> listeners, @NonNull String... outputs) {
        return output().data(iterator).output(outputs).listeners(listeners.toArray(new Listener[0])).exec();
    }

    /**
     * See {@link #output(DataSetIterator, List, String...)}.  No additional listeners.
     * <p>
     * Special case of {@link #output()}.
     */
    public Map<String, INDArray> output(@NonNull DataSetIterator dataSet, @NonNull String... outputs) {
        return output().data(dataSet).output(outputs).exec();
    }


    /**
     * See {@link #output(DataSetIterator, List, String...)}, but without the concatenation of batches.
     * <p>
     * Special case of {@link #output()}.
     */
    public List<Map<String, INDArray>> outputBatches(DataSetIterator iterator, List<Listener> listeners, String... outputs) {
        return output().data(iterator).output(outputs).listeners(listeners.toArray(new Listener[0])).execBatches();
    }


    /**
     * See {@link #output(DataSetIterator, String...)}, but without the concatenation of batches.
     * <p>
     * Special case of {@link #output()}.
     */
    public List<Map<String, INDArray>> outputBatches(DataSetIterator iterator, String... outputs) {
        return output().data(iterator).output(outputs).execBatches();
    }

    /**
     * Perform inference.<br>
     * <br>
     * <b>Example: classifier inference</b><br>
     * Predictions variable name: "softmaxOutput"<br>
     * Evaluations to perform: {@link Evaluation}<br>
     * Data: single output MultiDataSets<br>
     * Code:<br>
     * <pre>
     * {@code
     * MultiDataSetIterator data = ...
     * sameDiff.output(iterator, "softmaxOutput);
     * }
     * </pre>
     * <p>
     * Special case of {@link #output()}.
     *
     * @param iterator  The iterator - the source of the data for inference
     * @param listeners Additional listeners to use during this operation.
     * @param outputs   The set of outputs to report.  If null, defaults to all outputs of this SameDiff.
     */
    public Map<String, INDArray> output(@NonNull MultiDataSetIterator iterator, @NonNull List<Listener> listeners, @NonNull String... outputs) {
        return stackOutputs(outputHelper(iterator, At.defaultAt(Operation.INFERENCE), listeners, outputs));
    }

    /**
     * See {@link #output(MultiDataSetIterator, List, String...)}.  No additional listeners.
     * <p>
     * Special case of {@link #output()}.
     */
    public Map<String, INDArray> output(@NonNull MultiDataSetIterator dataSet, @NonNull String... outputs) {
        return output().data(dataSet).output(outputs).exec();
    }

    /**
     * Perform inference.<br>
     * <br>
     * <b>Example: classifier inference</b><br>
     * Predictions variable name: "softmaxOutput"<br>
     * Evaluations to perform: {@link Evaluation}<br>
     * Data: single output MultiDataSets<br>
     * Code:<br>
     * <pre>
     * {@code
     * MultiDataSetIterator data = ...
     * sameDiff.output(iterator, "softmaxOutput);
     * }
     * </pre>
     * <p>
     * Uses concatenation on the outputs of {@link #outputBatches(MultiDataSetIterator, List, String...)} which may cause issues with some inputs.
     * RNNs with variable time series length and CNNs with variable image sizes will most likely have issues.
     * <p>
     * Special case of {@link #output()}.
     *
     * @param iterator  The iterator - the source of the data for inference
     * @param listeners Additional listeners to use during this operation.
     * @param outputs   The set of outputs to report.  If null, defaults to all outputs of this SameDiff.
     */
    public List<Map<String, INDArray>> outputBatches(MultiDataSetIterator iterator, List<Listener> listeners, String... outputs) {
        return outputHelper(iterator, At.defaultAt(Operation.INFERENCE), listeners, outputs);
    }

    /**
     * See {@link #outputBatches(MultiDataSetIterator, List, String...)}.  No additional listeners.
     * <p>
     * Special case of {@link #output()}.
     */
    public List<Map<String, INDArray>> outputBatches(MultiDataSetIterator iterator, String... outputs) {
        return output().data(iterator).output(outputs).execBatches();
    }

    /**
     * Set up for an inference operation using OutputConfig.
     * Supports the setting of variables to output, the input data ({@link MultiDataSetIterator} or {@link DataSetIterator}),
     * and additional listeners.
     * Has exec methods to get results in batches or concatenated, or to get results when there is only
     * a single output (again in batches or concatenated).
     * <p>
     * Also supports using {@link SDVariable} instances instead of variable names.
     *
     * <br><br>
     * Example: get the output of pred, with batches concatenated together
     * <pre>
     *     {@code
     *     SameDiff sd = ...;
     *     MultiDataSet data = ...;
     *
     *     INDArray out = sd.output()
     *         .data(data)
     *         .output("pred")
     *         .outputSingle();
     *     }
     * </pre>
     */
    public OutputConfig output() {
        return new OutputConfig(this);
    }

    /**
     * Helper method to run inference.  Also used for validation
     */
    private List<Map<String, INDArray>> outputHelper(MultiDataSetIterator iterator, At at, @NonNull List<Listener> listeners, @NonNull String... outputs) {
        Preconditions.checkState(trainingConfig != null, "Training config has not been set");

        List<Listener> activeListeners = new ArrayList<>();

        for (Listener l : listeners)
            if (l.isActive(at.operation()))
                activeListeners.add(l);

        for (Listener l : this.listeners)
            if (l.isActive(at.operation()))
                activeListeners.add(l);

        validateListenerActivations(activeListeners, at.operation());

        for (Listener l : activeListeners)
            l.operationStart(this, at.operation());

        boolean hasListeners = !activeListeners.isEmpty();

        List<String> neededOutputs;

        if (outputs != null && outputs.length != 0) {
            neededOutputs = Arrays.asList(outputs);
        } else {
            neededOutputs = getLossVariables();
        }

        String[] neededOutputsArr = neededOutputs.toArray(new String[0]);

        List<Map<String, INDArray>> predictions = new ArrayList<>();

        if (!iterator.hasNext() && iterator.resetSupported())
            iterator.reset();

        Set<String> requiredVars = new HashSet<>();

        for (Listener l : activeListeners) {
            if (at.operation() == Operation.TRAINING_VALIDATION)
                requiredVars.addAll(l.requiredVariables(this).validationVariables());
            else
                requiredVars.addAll(l.requiredVariables(this).inferenceVariables());
        }

        while (iterator.hasNext()) {
            long dataStart = hasListeners ? System.currentTimeMillis() : 0;
            MultiDataSet ds = iterator.next();
            long dataEnd = hasListeners ? System.currentTimeMillis() : 0;
            Map<String, INDArray> placeholderMap = toPlaceholderMap(ds);

            if (hasListeners) {

                for (Listener l : activeListeners) {
                    l.iterationStart(this, at, ds, (dataEnd - dataStart));
                }

                Map<String, INDArray> outs = directExecHelper(placeholderMap, at, ds, requiredVars, activeListeners, neededOutputsArr);

                for (Listener l : activeListeners) {
                    l.iterationDone(this, at, ds, null);
                }

                predictions.add(outs);
            } else {
                predictions.add(directExecHelper(placeholderMap, at, ds, requiredVars, activeListeners, neededOutputsArr));
            }
            at.setIteration(at.iteration() + 1);
        }


        for (Listener l : activeListeners)
            l.operationEnd(this, at.operation());

        return predictions;
    }

    /**
     * Set up for a single batch inference operation using OutputConfig.
     * Supports the setting of placeholder inputs, outputs, and additional listeners.
     * Has exec methods to get the single output if only one is requested, or all requested outputs.
     * <p>
     * Also supports using {@link SDVariable} instances instead of variable names.
     * <p>
     * Example: get the value of "out" with placeholders x and y
     * <pre>
     *     {@code
     *     SameDiff sd = ...;
     *     INDArray xValue = ...;
     *     INDArray yValue = ...;
     *     SDVariable y = ...;
     *
     *     INDArray outValue = sd.batchOutput()
     *         .output("out")
     *         .input("x", xValue)
     *         .input(y, yValue)
     *         .outputSingle();
     *     }
     * </pre>
     */
    public BatchOutputConfig batchOutput() {
        return new BatchOutputConfig(this);
    }

    /**
     * Do inference for all variables for a single batch.
     * <p>
     * See {@link #output(Map, List, String...)}.
     * <p>
     * Special case of {@link #batchOutput()}.
     */
    public Map<String, INDArray> outputAll(Map<String, INDArray> placeholders) {
        return batchOutput().outputAll().inputs(placeholders).exec();
    }
    /**
     * Do inference for a single variable for a single batch.
     * <p>
     * See {@link #output(Map, List, String...)}.
     * <p>
     * Special case of {@link #batchOutput()}.
     */
    public INDArray outputSingle(Map<String, INDArray> placeholders, String output) {
        return batchOutput().output(output).inputs(placeholders).execSingle();
    }

    /**
     * Do inference for the given variables for a single batch.
     * <p>
     * See {@link #output(Map, List, String...)}.
     * <p>
     * Special case of {@link #batchOutput()}.
     */
    public Map<String, INDArray> output(Map<String, INDArray> placeholders, @NonNull List<String> outputs) {
        return batchOutput().output(outputs.toArray(new String[0])).inputs(placeholders).output();
    }

    /**
     * Do inference for the given variables for a single batch.
     * <p>
     * See {@link #output(Map, List, String...)}.
     * <p>
     * Special case of {@link #batchOutput()}.
     */
    public Map<String, INDArray> output(Map<String, INDArray> placeholders, String... outputs) {
        return batchOutput().output(outputs).inputs(placeholders).output();
    }


    /**
     * Do inference for the given variables for a single batch.
     * <p>
     * Special case of {@link #batchOutput()}.
     *
     * @param placeholders The values to use for placeholders.
     * @param listeners    Additional listeners to use during this operation.
     * @param outputs      The variables to output and return.
     */
    public Map<String, INDArray> output(Map<String, INDArray> placeholders, List<Listener> listeners, String... outputs) {
        return batchOutputHelper(placeholders, listeners, Operation.INFERENCE, outputs);
    }

    protected Map<String, INDArray> batchOutputHelper(Map<String, INDArray> placeholders, List<Listener> listeners, Operation operation, String... outputs) {
        List<Listener> activeListeners = new ArrayList<>();

        if(operation == null)
            operation = Operation.INFERENCE;

        for (Listener l : this.listeners)
            if (l.isActive(operation))
                activeListeners.add(l);

        if(listeners != null) {
            for (Listener l : listeners)
                if (l.isActive(operation))
                    activeListeners.add(l);
        }

        for (Listener l : activeListeners) {
            l.operationStart(this, operation);
        }

        validateListenerActivations(activeListeners, operation);

        Map<String, INDArray> ret = directExecHelper(placeholders, At.defaultAt(operation), null, Collections.<String>emptyList(), activeListeners, outputs);

        for (Listener l : activeListeners) {
            l.operationEnd(this, operation);
        }
        return ret;
    }

    /**
     * Do inference for the given variables for a single batch, with training information
     */
    protected Map<String, INDArray> directExecHelper(Map<String, INDArray> placeholders, At at, MultiDataSet batch,
                                                     Collection<String> requiredActivations, List<Listener> activeListeners, String... outputs) {
        if (at == null)
            at = At.defaultAt();

        Preconditions.checkState(outputs != null && outputs.length > 0, "No outputs were specified");
        long threadId = Thread.currentThread().getId();
        if (!sessions.containsKey(threadId)) {
            log.info("Creating new InferenceSession for thread {}", threadId);
            sessions.put(threadId, new InferenceSession(this));
        }

        List<String> phNames = inputs();
        if (placeholders == null && phNames != null) {
            //Maybe user set placeholders before calling exec method?
            placeholders = placeholdersPerThread.get(Thread.currentThread().getId());
        }

        //Placeholder validation is performed in InferenceSession

        InferenceSession is = sessions.get(threadId);
        return is.output(outputs == null ? Collections.<String>emptyList() : Arrays.asList(outputs),
                placeholders, batch, requiredActivations, activeListeners, at);
    }

    /**
     * See {@link #one(String, DataType, int...)}.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     * Uses the DataType of the Nd4j default floating point type ({@link Nd4j#defaultFloatingPointType()}).
     */
    public SDVariable one(String name, int... shape) {
        return one(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * See {@link #one(String, DataType, long...)}.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     * Uses the DataType of the Nd4j default floating point type ({@link Nd4j#defaultFloatingPointType()}).
     */
    public SDVariable one(String name, long... shape) {
        return one(name, Nd4j.defaultFloatingPointType(), shape);
    }


    /**
     * Create a new variable with the specified shape, with all values initialized to 1.0.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return one(name, dataType, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 1.0.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return constant(name, Nd4j.ones(dataType, shape));
    }

    /**
     * See {@link #zero(String, DataType, long...)}.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     * Uses the DataType of the Nd4j default floating point type ({@link Nd4j#defaultFloatingPointType()}).
     */
    public SDVariable zero(String name, long... shape) {
        return zero(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * See {@link #zero(String, DataType, int...)}.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     * Uses the DataType of the Nd4j default floating point type ({@link Nd4j#defaultFloatingPointType()}).
     */
    public SDVariable zero(String name, int... shape) {
        return zero(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 0.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        return constant(name, Nd4j.zeros(dataType, shape));
    }

    /**
     * Create a new variable with the specified shape, with all values initialized to 0.
     * Creates a constant - i.e., CONSTANT type SDVariable.
     *
     * @param name  the name of the variable to create
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return zero(name, dataType, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create an SDVariable with a fixed/constant value, with a generated name<br>
     * Constants are not modified by training/backprop. See {@link VariableType} for more details.
     *
     * @param constant Value for the constant SDVariable
     * @return The created variable
     */
    public SDVariable constant(@NonNull INDArray constant) {
        return constant(getNewVarName(), constant);
    }

    /**
     * Create an SDVariable with a fixed/constant value<br>
     * Constants are not modified by training/backprop. See {@link VariableType} for more details.
     *
     * @param name     Name of the constant SDVariable
     * @param constant Value for the constant SDVariable
     * @return The created variable
     */
    public SDVariable constant(String name, @NonNull INDArray constant) {
        Preconditions.checkState(!variables.containsKey(name), "Variable with name \"%s\" already exists", name);
        if (name == null || name.length() < 1)
            name = getNewVarName();
        if(constant.isView()) {
            try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()){
                constant = constant.dup();
            }
        }

        SDVariable v = new SDVariable(name, VariableType.CONSTANT, this, constant.shape(), constant.dataType());
        name = v.name();
        variables.put(name, Variable.builder().name(name).variable(v).build());
        constantArrays.setArray(name, constant);
        return v;
    }

    /**
     * Create a a placeholder variable. Placeholders are variables that expect an array to be provided during training
     * and inference.<br>
     * For example, the SDVariables for your input/features and labels should be placeholders.<br>
     * See also: {@link VariableType}
     *
     * @param name     the name of the variable
     * @param dataType Data type of the new placeholder
     * @param shape    the shape of the variable if any
     * @return SDVariable placeholder
     */
    public SDVariable placeHolder(@NonNull String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        Preconditions.checkState(!variables.containsKey(name), "Variable already exists with name %s", name);
        SDVariable ret = new SDVariable(name, VariableType.PLACEHOLDER, this, shape, dataType);
        variables.put(name, Variable.builder().name(name).variable(ret).build());
        return ret;
    }

    /**
     * Variable initialization with a specified {@link WeightInitScheme}
     * This method creates VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name             the name of the variable
     * @param shape            the shape of the array to be created
     * @param weightInitScheme the weight initialization scheme
     * @return the created variable
     */
    public SDVariable var(@NonNull String name, @NonNull WeightInitScheme weightInitScheme, @NonNull org.nd4j.linalg.api.buffer.DataType dataType, @NonNull long... shape) {
        return var(name, VariableType.VARIABLE, weightInitScheme, dataType, shape);
    }

    /**
     * Variable initialization with a specified {@link WeightInitScheme}
     * This method creates VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name             the name of the variable
     * @param variableType     the SameDiff variable type of the variable (e.g. CONSTANT, PLACEHOLDER, etc.)
     * @param weightInitScheme the weight initialization scheme
     * @param dataType         the data type of the variable (float, int, etc)
     * @param shape            the shape of the array to be created
     * @return the created variable
     */
    public SDVariable var(@NonNull String name, @NonNull VariableType variableType, WeightInitScheme weightInitScheme,
                          org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        if(shape != null) {
            for (long l : shape) {
                Preconditions.checkArgument(l != 0, "Cannot create variable with a shape that contains zeros (empty array shape) - got shape %s", shape);
            }
        }

        if (name == null || name.length() < 1)
            name = getNewVarName();
        else
            name = generateNewVarName(name, 0);

        if (variables.containsKey(name)) {
            if (nameScopes.isEmpty()) {
                throw new IllegalArgumentException("Another variable with the name " + name + " already exists (current name scope: \""
                        + currentNameScope() + "\"");
            } else {
                throw new IllegalArgumentException("Another variable with the name " + name + " already exists.");
            }
        }

        Preconditions.checkState(variableType != VariableType.VARIABLE || weightInitScheme != null, "A weight initalization scheme must be provided" +
                " when creating a VARIABLE type SDVariables - variable name: \"%s\"", name);

        SDVariable ret = new SDVariable(name, variableType, this, shape, dataType);
        addVariable(ret);

        if(variableType == VariableType.VARIABLE){
            try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                INDArray vArr = weightInitScheme.create(dataType, shape);
                variablesArrays.setArray(name, vArr);
            }
        }

        return ret;
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * The underlying array will be initialized using the specified weight initilization scheme<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name             the name of the variable
     * @param shape            the shape of the variable
     * @param weightInitScheme Weight initialization scheme to use to initialize the underlying array
     * @return the created variable
     */
    public SDVariable var(@NonNull String name, @NonNull LongShapeDescriptor shape, WeightInitScheme weightInitScheme) {
        return var(name, weightInitScheme, shape.dataType(), shape.getShape());
    }


    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, org.nd4j.linalg.api.buffer.DataType dataType, long... shape) {
        Preconditions.checkNotNull(shape != null, "Invalid shape: shape may not be null");
        if (Shape.isPlaceholderShape(shape)) {
            return placeHolder(name, dataType, shape);
        }
        return var(name, new ZeroInitScheme(), dataType, shape);
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name      the name of the variable
     * @param shapeDesc the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, LongShapeDescriptor shapeDesc) {
        Preconditions.checkNotNull(shapeDesc != null, "Invalid shape: shape may not be null");
        return var(name, shapeDesc, new ZeroInitScheme());
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values. Data type will be given by {@link Nd4j#defaultFloatingPointType()}<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, int... shape) {
        return var(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values. Data type will be given by {@link Nd4j#defaultFloatingPointType()}<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, long... shape) {
        return var(name, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Variable initialization with a specified {@link WeightInitScheme}. Data type will be given by {@link Nd4j#defaultFloatingPointType()}<br>
     * This method creates VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param name             the name of the variable
     * @param shape            the shape of the array to be created
     * @param weightInitScheme the weight initialization scheme
     * @return the created variable
     */
    public SDVariable var(@NonNull String name, @NonNull WeightInitScheme weightInitScheme, @NonNull long... shape) {
        return var(name, weightInitScheme, Nd4j.defaultFloatingPointType(), shape);
    }

    /**
     * Creates a {@link SDVariable} with the given shape and name<br>
     * Any array will be generated with all zeros for the values<br>
     *
     * @param name  the name of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        Preconditions.checkNotNull(shape, "Invalid shape: shape may not be null");
        if (Shape.isPlaceholderShape(shape)) {
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
        if (variables.containsKey(v.name()) && variables.get(v.name()).getVariable().getArr() != null)
            return variables.get(v.name()).getVariable();

        if (v.name() == null || v.name().length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        VariableType vt = v.getVariableType();
        NDArraySupplierInitScheme s = null;
        switch (vt) {
            case VARIABLE:
                SDVariable r = new SDVariable(v.name(), v.getVariableType(), this, v.getShape(), v.dataType());
                addVariable(r);
                try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
                    variablesArrays.setArray(v.name(), v.getArr().dup());
                }
                return r;
            case ARRAY:
                SDVariable ret = new SDVariable(v.name(), v.getVariableType(), this, v.getShape(), v.dataType());
                return addVariable(ret);
            case CONSTANT:
                return constant(v.name(), v.getArr());
            case PLACEHOLDER:
                return placeHolder(v.name(), v.dataType(), v.placeholderShape());
            default:
                throw new RuntimeException("Unknown/not supported variable type: " + vt);
        }
    }

    private String getNewVarName() {
        return generateNewVarName("sd_var", 0, false);
    }

    /**
     * Creates a {@link SDVariable} with the specified shape and a generated name<br>
     * Any array will be generated with all zeros for the values<br>
     * This method creates a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(org.nd4j.linalg.api.buffer.DataType dataType, int... shape) {
        return var(getNewVarName(), dataType, shape);
    }

    /**
     * Creates a {@link SDVariable} with the specified shape and a generated name<br>
     * Any array will be generated with all zeros for the values<br>
     * This method creates a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
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
     * Create an {@link SDVariable} with a generated name, and assocate the specified array with it.<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param arr Array to associate with the new variable
     * @return New SDVariable
     * @see #var(String, INDArray)
     */
    public SDVariable var(INDArray arr) {
        return var(getNewVarName(), arr);
    }

    /**
     * Create an {@link SDVariable} with the specified name, and associate the specified array with it<br>
     * This is a VARIABLE type SDVariable - i.e., must be floating point, and is a trainable parameter. See {@link VariableType} for more details.
     *
     * @param arr Array to associate with the new variable
     * @return New SDVariable with the specified name and array
     */
    public SDVariable var(String name, @NonNull INDArray arr) {
        if (variables.containsKey(name) && variables.get(name).getVariable().getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name + " already exists.");
        Preconditions.checkState(arr.dataType().isFPType(), "Cannot create variable with non-floating point type:" +
                " provided array has datatype %s. Variables must be floating point type to be trainable by backpropagation.\n" +
                "For non floating point types, these should be created as placeholders or constants instead.", arr.dataType());
        Preconditions.checkArgument(!arr.isEmpty(), "Empty arrays cannot be used when creating variables. Array shape: %ndShape", arr);

        if (name == null || name.length() < 1)
            name = getNewVarName();

        boolean duped = false;
        if (arr.isAttached()) {
            arr = arr.detach();
            duped = true;
        }

        if (!duped) {
            for (String s : variablesArrays.arrayNames()) {
                if (variablesArrays.getArray(s) == arr) {    //Check for exact same object, to avoid array reuse (can result in unexpected behaviour)
                    arr = arr.dup();
                    break;
                }
            }
        }


        SDVariable ret = new SDVariable(name, VariableType.VARIABLE, this, arr.shape(), arr.dataType());
        associateArrayWithVariable(arr, ret);

        addVariable(ret);
        return ret;
    }

    /**
     * Convert the specified variable to a constant. This is equivalent to "freezing" a variable so that it's value
     * won't be changed by further training.<br>
     * This can only be done for variables and placeholders, not ARRAY type variables (which are usually network activations).
     * As a constant, this variable will no longer be modified by any subsequent training.<br>
     * See also: {@link VariableType}
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
     * As constants, these variables will no longer be modified by any subsequent training.<br>
     * See also: {@link VariableType}
     *
     * @param variables Variables to convert to constants
     * @return The (now constant) SDVariables
     */
    public void convertToConstants(List<SDVariable> variables) {
        if (variables.size() == 0)
            return;
        boolean allConst = true;
        for (SDVariable variable : variables) {
            if (variable.getVariableType() != VariableType.CONSTANT) {
                allConst = false;
                Preconditions.checkState(variable.getVariableType() != VariableType.ARRAY, "Cannot convert variable of type ARRAY to a constant: %s", variable);
            }
        }
        if (allConst) {
            return; //No op
        }

        //Remove all sessions in case they have any cached arrays/state
        sessions.clear();

        //If gradient function has been defined, remove it (so it will be recreated later)
        sameDiffFunctionInstances.remove(GRAD_FN_KEY);

        for (SDVariable variable : variables) {
            String n = variable.name();
            INDArray arr = variable.getArr();
            Preconditions.checkNotNull(arr, "Could not get array for variable %s: if this is a placeholder, use SDVariable.setArray before converting", variable);

            constantArrays.setArray(n, arr);   //DeviceLocal with delayed initialization, in case we don't actually need multiple threads
            variablesArrays.removeArray(n);
            if (!placeholdersPerThread.isEmpty()) {
                for (Map<String, INDArray> m : placeholdersPerThread.values()) {
                    m.remove(n);
                }
            }

            variable.setVariableType(VariableType.CONSTANT);
        }


        if (trainingConfig != null && initializedTraining) {
            //Remove updater state for now constant variables
            for (SDVariable v : variables) {
                GradientUpdater gu = updaterMap.remove(v.name());
                Map<String, INDArray> m = gu == null ? null : gu.getState();
                if (m != null) {
                    for (INDArray arr : m.values()) {
                        if (arr.closeable())
                            arr.close();
                    }
                }

                //Also check dataset feature/label mapping -  remove any placeholders here...
                if (trainingConfig.getDataSetFeatureMapping() != null && trainingConfig.getDataSetFeatureMapping().contains(v.name())) {
                    List<String> newFM = new ArrayList<>(trainingConfig.getDataSetFeatureMapping());    //New list in case of immutable list
                    newFM.remove(v.name());
                    trainingConfig.setDataSetFeatureMapping(newFM);
                }

                if (trainingConfig.getDataSetLabelMapping() != null && trainingConfig.getDataSetLabelMapping().contains(v.name())) {
                    List<String> newLM = new ArrayList<>(trainingConfig.getDataSetLabelMapping());
                    newLM.remove(v.name());
                    trainingConfig.setDataSetLabelMapping(newLM);
                }

                if (trainingConfig.getDataSetFeatureMaskMapping() != null && trainingConfig.getDataSetFeatureMaskMapping().contains(v.name())) {
                    List<String> newFMM = new ArrayList<>(trainingConfig.getDataSetFeatureMaskMapping());
                    newFMM.remove(v.name());
                    trainingConfig.setDataSetFeatureMaskMapping(newFMM);
                }

                if (trainingConfig.getDataSetLabelMaskMapping() != null && trainingConfig.getDataSetLabelMaskMapping().contains(v.name())) {
                    List<String> newLMM = new ArrayList<>(trainingConfig.getDataSetLabelMaskMapping());
                    newLMM.remove(v.name());
                    trainingConfig.setDataSetLabelMaskMapping(newLMM);
                }
            }
        }
    }

    /**
     * Convert the specified variable to a VARIABLE type SDVariable.<br>
     * This can only be done for constants and placeholders, not ARRAY type variables (which are usually network activations).
     * As a variable, this variable will modified during any subsequent training.<br>
     * See also: {@link VariableType}
     *
     * @return This variable (now a variable type SDVariable)
     */
    public SDVariable convertToVariable(@NonNull SDVariable constant) {
        Preconditions.checkState(constant.dataType().isFPType(), "Only floating point SDVariables can be converted to variables," +
                " datatype of %s is %s", constant.name(), constant.dataType());
        convertToVariables(Collections.singletonList(constant));
        return constant;
    }

    /**
     * Convert the specified variables to VARIABLE type SDVariables.<br>
     * This can only be done for constants and placeholders, not ARRAY type variables (which are usually network activations).
     * As variables, this variable will modified during any subsequent training.<br>
     * See also: {@link VariableType}
     */
    public void convertToVariables(@NonNull List<SDVariable> constants) {
        if (constants.size() == 0)
            return;
        boolean allConst = true;
        for (SDVariable variable : constants) {
            if (variable.getVariableType() != VariableType.VARIABLE) {
                allConst = false;
            }
            Preconditions.checkState(variable.getVariableType() != VariableType.ARRAY, "Cannot convert variable of type ARRAY to a variable: %s", variable);
        }
        if (allConst) {
            return; //No op
        }

        //Remove all sessions in case they have any cached arrays/state
        sessions.clear();

        //If gradient function has been defined, remove it (so it will be recreated later)
        sameDiffFunctionInstances.remove(GRAD_FN_KEY);

        for (SDVariable variable : constants) {
            String n = variable.name();
            INDArray arr = variable.getArr();
            Preconditions.checkNotNull(arr, "Could not get array for variable %s: if this is a placeholder, use SDVariable.setArray before converting", variable);

            variablesArrays.setArray(n, arr);  //DeviceLocal with delayed initialization, in case we don't actually need multiple threads
            constantArrays.removeArray(n);
            if (!placeholdersPerThread.isEmpty()) {
                for (Map<String, INDArray> m : placeholdersPerThread.values()) {
                    m.remove(n);
                }
            }

            variable.setVariableType(VariableType.VARIABLE);
        }


        //For training: need to add new updater state
        if (trainingConfig != null && initializedTraining) {
            //Add updater state for this variable: updaterState, updaterViews, updaterMap
            for (SDVariable v : constants) {
                if (!updaterMap.containsKey(v.name())) {
                    //Create new updater state
                    INDArray arr = v.getArr();
                    long thisSize = trainingConfig.getUpdater().stateSize(arr.length());
                    if (thisSize > 0) {
                        INDArray stateArr = Nd4j.create(arr.dataType(), 1, thisSize);
                        GradientUpdater u = trainingConfig.getUpdater().instantiate(stateArr, false);
                        u.setStateViewArray(stateArr, arr.shape(), arr.ordering(), true);                       //TODO eventually this should be 1 call...
                        updaterMap.put(v.name(), u);
                    } else {
                        GradientUpdater u = trainingConfig.getUpdater().instantiate((INDArray) null, true);
                        updaterMap.put(v.name(), u);
                    }
                }
            }
        }
    }

    /**
     * Convert the datatypes of the specified constants, placeholders and variables.<br>
     * After conversion, the downstream datatypes are changed.
     * For example, {@code z(float) = x(float)+y(float)}, changing both x and y to double results in {@code z(double) = x(double)+y(double)}
     * without doing anything to change z's datatype directly (z datatype is inferred from x + y + add op).<br>
     * ARRAY type SDVariables cannot be converted directly, as their datatypes are determined by the function +
     * input datatypes.<b>
     * Note that this method should be used with caution: incorrect datatype modifications may leave your network
     * in an incorrect state. For example, {@code op(x(float),y(float)) -> op(x(double),y(float))} may not be
     * supported by all ops.
     *
     * @param dataTypeMap Map of SDVariables to change the datatype for. Key = SDVariable name, Value = new datatype
     */
    public void convertDataTypes(@NonNull Map<String, DataType> dataTypeMap) {
        if (dataTypeMap.isEmpty())
            return;

        //First: check these are all either constants, variables or placeholders.
        for (Map.Entry<String, DataType> e : dataTypeMap.entrySet()) {
            String s = e.getKey();
            Preconditions.checkState(variables.containsKey(s), "Cannot change datatype of variable \"%s\": No variable with this name exists", s);
            SDVariable v = variables.get(s).getVariable();
            Preconditions.checkState(v.getVariableType() != VariableType.ARRAY, "Cannot change datatype of ARRAY type variable \"%s\": " +
                    "datatype of ARRAY type variables is determined by the datatypes of their inputs plus corresponding ");
            if (v.getVariableType() != VariableType.PLACEHOLDER) {
                //Can't convert constant or variable between numerical and non-numerical type (not possible to cast)
                Preconditions.checkState(v.dataType().isNumerical() == e.getValue().isNumerical(), "Cannot convert variables between numerical " +
                        "and non-numerical types: attempting to convert variable \"%s\" from %s to %s", e.getKey(), v.dataType(), e.getValue());
            }
        }

        boolean anyChanged = false;
        for (Map.Entry<String, DataType> e : dataTypeMap.entrySet()) {
            String s = e.getKey();
            DataType d = e.getValue();
            SDVariable v = variables.get(s).getVariable();
            if (v.dataType() == d)
                continue;   //No-op

            v.setDataType(d);

            switch (v.getVariableType()) {
                case VARIABLE:
                    INDArray arr = variablesArrays.removeArray(e.getKey());
                    INDArray newArr = arr.castTo(d);
                    variablesArrays.setArray(e.getKey(), newArr);  //DeviceLocal with delayed initialization, in case we don't actually need multiple threads
                    break;
                case CONSTANT:
                    INDArray arr2 = constantArrays.removeArray(e.getKey());
                    INDArray newArr2 = arr2.castTo(d);
                    constantArrays.setArray(e.getKey(), newArr2);  //DeviceLocal with delayed initialization, in case we don't actually need multiple threads
                    break;
                case PLACEHOLDER:
                    Map<String, INDArray> m = placeholdersPerThread.get(Thread.currentThread().getId());
                    if (m != null && m.containsKey(e.getKey())) {
                        m.put(e.getKey(), m.get(e.getKey()).castTo(d));
                    }
                    break;
                case ARRAY:
                default:
                    throw new IllegalStateException("Cannot convert array type variable");  //Should never happen
            }


            anyChanged = true;
        }

        if (anyChanged) {
            sessions.clear();

            //Recalculate datatypes of outputs, and dynamically update them
            Set<String> allSeenOps = new HashSet<>();
            Queue<String> queueOps = new LinkedList<>();

            for(String s : dataTypeMap.keySet()){
                Variable v = variables.get(s);
                v.getVariable().setDataType(dataTypeMap.get(s));
                List<String> inToOp = v.getInputsForOp();
                if(inToOp != null){
                    for(String op : inToOp) {
                        if (!allSeenOps.contains(op)) {
                            allSeenOps.add(op);
                            queueOps.add(op);
                        }
                    }
                }
            }

            while(!queueOps.isEmpty()){
                String op = queueOps.remove();
                SameDiffOp o = ops.get(op);
                List<String> inVars = o.getInputsToOp();
                List<DataType> inDTypes = new ArrayList<>();
                if(inVars != null) {
                    for (String s : inVars) {
                        SDVariable v = variables.get(s).getVariable();
                        inDTypes.add(v.dataType());
                    }
                }
                List<DataType> outDtypes = o.getOp().calculateOutputDataTypes(inDTypes);
                List<String> outVars = o.getOutputsOfOp();
                for( int i=0; i<outVars.size(); i++ ){
                    String varName = outVars.get(i);
                    Variable var = variables.get(varName);
                    SDVariable v = var.getVariable();
                    v.setDataType(outDtypes.get(i));

                    //Also update queue
                    if(var.getInputsForOp() != null){
                        for(String opName : var.getInputsForOp()){
                            if(!allSeenOps.contains(opName)){
                                allSeenOps.add(opName);
                                queueOps.add(opName);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Rename the specified variable to the new name.
     *
     * @param from The variable to rename - this variable must exist
     * @param to   The new name for the variable - no variable with this name must already exist
     */
    public void renameVariable(String from, String to) {
        Preconditions.checkState(variables.containsKey(from), "Cannot rename variable \"%s\": no variable with this name exists", from);
        Preconditions.checkState(!variables.containsKey(to), "Cannot rename variable \"%s\" to name \"%s\": a variable with name \"%s\" already exists", from, to, to);

        Variable v = variables.get(from);
        v.setName(to);
        v.getVariable().setVarName(to);
        if (v.getInputsForOp() != null) {
            for (String opName : v.getInputsForOp()) {
                SameDiffOp op = ops.get(opName);
                List<String> newInputs = new ArrayList<>(op.getInputsToOp());
                while (newInputs.contains(from)) {
                    newInputs.set(newInputs.indexOf(from), to);
                }
                op.setInputsToOp(newInputs);
            }
        }

        if (v.getControlDepsForOp() != null) {
            for (String opName : v.getControlDepsForOp()) {
                SameDiffOp op = ops.get(opName);
                List<String> newCDs = new ArrayList<>(op.getControlDeps());
                while (newCDs.contains(from)) {
                    newCDs.set(newCDs.indexOf(from), to);
                }
                op.setControlDeps(newCDs);
            }
        }

        if (v.getControlDepsForVar() != null) {
            for (String varName : v.getControlDepsForVar()) {
                Variable var = variables.get(varName);
                List<String> newCDs = new ArrayList<>(var.getControlDeps());
                while (newCDs.contains(from)) {
                    newCDs.set(newCDs.indexOf(from), to);
                }
                var.setControlDeps(newCDs);
            }
        }

        if (v.getControlDeps() != null) {
            for (String varName : v.getControlDeps()) {
                Variable var = variables.get(varName);
                List<String> newCDsFor = new ArrayList<>(var.getControlDepsForVar());
                while (newCDsFor.contains(from)) {
                    newCDsFor.set(newCDsFor.indexOf(from), to);
                }
                var.setControlDepsForVar(newCDsFor);
            }
        }

        if (v.getOutputOfOp() != null) {
            SameDiffOp op = ops.get(v.getOutputOfOp());
            List<String> newOuts = new ArrayList<>(op.getOutputsOfOp());
            while (newOuts.contains(from)) {
                newOuts.set(newOuts.indexOf(from), to);
            }
            op.setOutputsOfOp(newOuts);
        }

        variables.remove(from);
        variables.put(to, v);

        if(v.getVariable().getVariableType() == VariableType.CONSTANT && constantArrays.hasArray(from)){
            constantArrays.rename(from, to);
        }

        if(v.getVariable().getVariableType() == VariableType.VARIABLE && variablesArrays.hasArray(from)){
            variablesArrays.rename(from, to);
        }

        if(v.getVariable().getVariableType() == VariableType.PLACEHOLDER ){
            for(Map<String,INDArray> e : placeholdersPerThread.values()){
                //Not really thread safe - but renaming variables during execution in other threads can never be thread safe :)
                if(e != null && e.containsKey(from)){
                    INDArray arr = e.remove(from);
                    e.put(to, arr);
                }
            }
        }

        if (trainingConfig != null) {
            if (trainingConfig.getDataSetFeatureMapping() != null && trainingConfig.getDataSetFeatureMapping().contains(from)) {
                List<String> l = new ArrayList<>(trainingConfig.getDataSetFeatureMapping());
                while (l.contains(from)) {
                    l.set(l.indexOf(from), to);
                }
                trainingConfig.setDataSetFeatureMapping(l);
            }

            if (trainingConfig.getDataSetLabelMapping() != null && trainingConfig.getDataSetLabelMapping().contains(from)) {
                List<String> l = new ArrayList<>(trainingConfig.getDataSetLabelMapping());
                while (l.contains(from)) {
                    l.set(l.indexOf(from), to);
                }
                trainingConfig.setDataSetLabelMapping(l);
            }

            if (trainingConfig.getDataSetFeatureMaskMapping() != null && trainingConfig.getDataSetFeatureMaskMapping().contains(from)) {
                List<String> l = new ArrayList<>(trainingConfig.getDataSetFeatureMaskMapping());
                while (l.contains(from)) {
                    l.set(l.indexOf(from), to);
                }
                trainingConfig.setDataSetFeatureMaskMapping(l);
            }

            if (trainingConfig.getDataSetLabelMaskMapping() != null && trainingConfig.getDataSetLabelMaskMapping().contains(from)) {
                List<String> l = new ArrayList<>(trainingConfig.getDataSetLabelMaskMapping());
                while (l.contains(from)) {
                    l.set(l.indexOf(from), to);
                }
                trainingConfig.setDataSetLabelMaskMapping(l);
            }

            if (trainingConfig.getLossVariables() != null && trainingConfig.getLossVariables().contains(from)) {
                List<String> l = new ArrayList<>(trainingConfig.getLossVariables());
                while (l.contains(from)) {
                    l.set(l.indexOf(from), to);
                }
                trainingConfig.setLossVariables(l);
            }
        }

        for (SameDiff sd : sameDiffFunctionInstances.values()) {
            if (sd.hasVariable(from)) {
                sd.renameVariable(from, to);
            }
        }
    }


    /**
     * Remove an argument for a function. Note that if this function does not contain the argument, it will just be a no op.
     *
     * @param varName  the variable name to remove
     * @param function the function to remove the argument from
     */
    public void removeArgFromOp(String varName, DifferentialFunction function) {
        val args = function.args();

        for (int i = 0; i < args.length; i++) {
            if (args[i].name().equals(varName)) {
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

        variables.get(varName).getInputsForOp().remove(function.getOwnName());
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

    public boolean hasVariable(String name) {
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
        } else if (sameDiffFunctionInstances.containsKey(GRAD_FN_KEY) && sameDiffFunctionInstances.get(GRAD_FN_KEY).variables.containsKey(varName)) {
            return sameDiffFunctionInstances.get(GRAD_FN_KEY).variables.get(varName).getGradient();
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
    public boolean variableHasGradient(String varName) {
        Preconditions.checkState(variables.containsKey(varName), "No variable with name \"%s\" exists", varName);
        SDVariable v = getVariable(varName);
        if (!v.dataType().isFPType() || v.isConstant())
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
     * Get the gradient for the variable with the specified variable name.
     * Note that in order to run this function, {@link #execBackwards(Map, Operation, MultiDataSet, Collection, List)} must be executed first.
     * All gradient functions are obtained from the results of the execBackwards call.
     *
     * @param varName the variable name to get the gradient variable for.
     * @return The gradient variable for the specified variable
     */
    public SDVariable grad(String varName) {
        if (!sameDiffFunctionInstances.containsKey(GRAD_FN_KEY)) {
            createGradFunction();
        }

        SameDiff grad = getFunction(GRAD_FN_KEY);
        SDVariable var = grad.getVariable(varName);
        return getFunction(GRAD_FN_KEY).getGradForVariable(var.name());
    }


    /**
     * Create a new double scalar (rank 0) SDVariable with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, double value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new float scalar (rank 0) SDVariable with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, float value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new integer scalar (rank 0) SDVariable with the specified value
     *
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
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the variable with
     * @return SDVariable
     */
    public SDVariable scalar(String name, long value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
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
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return var(name, Nd4j.scalar(dataType, value));
        }
    }

    /**
     * Create a new double scalar constant (rank 0) with the specified value.<br>
     * Constants are not modified by training/backprop. See {@link VariableType} for more details.
     *
     * @param value Value to initialize the constant with
     * @return SDVariable
     */
    public SDVariable constant(double value) {
        return constant(null, value);
    }

    /**
     * Create a new double scalar constant (rank 0) with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the constant with
     * @return SDVariable
     */
    public SDVariable constant(String name, double value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return constant(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new float scalar constant (rank 0) with the specified value<br>
     * Constants are not modified by training/backprop. See {@link VariableType} for more details.
     *
     * @param value Value to initialize the constant with
     * @return SDVariable
     */
    public SDVariable constant(float value) {
        return constant(null, value);
    }

    /**
     * Create a new float scalar constant (rank 0) with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the constant with
     * @return SDVariable
     */
    public SDVariable constant(String name, float value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return constant(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new integer scalar constant (rank 0) with the specified value
     *
     * @param value Value to initialize the constant with
     */
    public SDVariable constant(int value) {
        return constant(null, value);
    }

    /**
     * Create a new integer scalar constant (rank 0) with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the constant with
     * @return SDVariable
     */
    public SDVariable constant(String name, int value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return constant(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new long scalar constant (rank 0) with the specified value
     *
     * @param value Value to initialize the constant with
     */
    public SDVariable constant(long value) {
        return constant(null, value);
    }

    /**
     * Create a new long scalar constant (rank 0) with the specified value
     *
     * @param name  Name of the SDVariable
     * @param value Value to initialize the constant with
     */
    public SDVariable constant(String name, long value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return constant(name, Nd4j.scalar(value));
        }
    }

    /**
     * Create a new scalar constant (rank 0) with the specified value and datatype
     *
     * @param name     Name of the SDVariable
     * @param dataType Data type of the scalar constant
     * @param value    Value to initialize the constant with
     */
    public SDVariable constant(String name, DataType dataType, Number value) {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return constant(name, Nd4j.scalar(dataType, value));
        }
    }

    /**
     * Add the specified variable to this SameDiff instance
     *
     * @param variable Variable to add
     */
    public SDVariable addVariable(SDVariable variable) {
        Preconditions.checkState(variable.getSameDiff() == this, "Samediff instance must be the same.");

        if (variables.containsKey(variable.name()) && !variables.get(variable.name()).getVariable().equals(variable)) {
            throw new IllegalArgumentException("Variable with name \"" + variable.name() + "\" already exists");
        }

        Preconditions.checkState(variable.getSameDiff() == this, "Same diff instance for variable must be the same!");
        variables.put(variable.name(), Variable.builder().name(variable.name()).variable(variable).build());
        return variable;
    }


    /**
     * Generate the variables based on the given input op and return the output variable names.
     *
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function, String baseName, boolean isImport) {
        if (baseName == null)
            baseName = function.getOwnName();

        if (baseName == null)
            baseName = function.opName();

        //First: calculate output data types. We can always calculate output data types, even if the input arrays
        //are not available - *except for sometimes during import, until all ops/variables have been added*
        List<org.nd4j.linalg.api.buffer.DataType> outputDataTypes = null;

        if (!isImport) {
            List<org.nd4j.linalg.api.buffer.DataType> inputDataTypes = new ArrayList<>();
            List<String> fnInputs = ops.get(function.getOwnName()).getInputsToOp();
            if (fnInputs != null) {
                for (String var : fnInputs) {
                    inputDataTypes.add(variables.get(var).getVariable().dataType());
                }
            }
            outputDataTypes = function.calculateOutputDataTypes(inputDataTypes);
        }

        //Determine number of output variables
        if (function instanceof CustomOp) {
            CustomOp customOp = (CustomOp) function;
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
            SDVariable[] ret = new SDVariable[num_outputs];

            //Infer the output types: we can always determine datatype but not always shapes
            Preconditions.checkState(isImport || (outputDataTypes != null && outputDataTypes.size() == num_outputs),
                    "Incorrect number of output datatypes: got %s but expected datatypes for %s outputs - %s (op: %s)",
                    (outputDataTypes == null ? null : outputDataTypes.size()), num_outputs, outputDataTypes, function.getClass().getSimpleName());

            //dynamic shapes
            //When importing from TF: convention is "unstack", "unstack:1", "unstack:2", ...
            for (int i = 0; i < ret.length; i++) {
                SDVariable var = (i == 0 ? getVariable(baseName) : getVariable(baseName + ":" + i));
                if (var == null) {
                    //Generate new variable name if one with the specified name doesn't exist
                    //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme

                    org.nd4j.linalg.api.buffer.DataType dataType = isImport ? null : outputDataTypes.get(i);
                    var = var(generateNewVarName(baseName, i), VariableType.ARRAY, null, dataType, (long[]) null);
                }
                var.setCreator(function);
                ret[i] = var;
            }

            //Update the internal state: outgoing variables for function
            if (getOutputsForOp(function) == null)
                addOutgoingFor(ret, function);

            return ret;
        }

        //this is for unresolved shapes, we know xyz is always 1 output
        else if (function instanceof BaseOp) {
            SDVariable[] ret = new SDVariable[1];
            SDVariable checkGet = getVariable(baseName);
            SDVariable[] args = function.args();
            if (checkGet == null) {
                //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme
                org.nd4j.linalg.api.buffer.DataType dataType = outputDataTypes.get(0);
                checkGet = var(baseName, VariableType.ARRAY, null, dataType, (long[]) null);
            }

            if (checkGet == null) {
                //Note: output of an op is ARRAY type - activations, not a trainable parameter. Thus has no weight init scheme
                org.nd4j.linalg.api.buffer.DataType dataType = outputDataTypes.get(0);
                checkGet = var(baseName, VariableType.ARRAY, null, dataType, (long[]) null);
            }

            checkGet.setCreator(function);
            ret[0] = checkGet;


            //Update the internal state: outgoing variables for function
            if (getOutputsForOp(function) == null)
                addOutgoingFor(ret, function);

            return ret;
        } else {
            throw new RuntimeException("Unknown op type: " + function.getClass());
        }
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
        return generateOutputVariableForOp(function,
                function.getOwnName() != null ? function.getOwnName() : function.opName(), false);
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
     * Create a new TensorArray.
     */
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

    /**
     * See {@link #calculateGradients(Map, Collection)}
     */
    public Map<String, INDArray> calculateGradients(Map<String, INDArray> placeholderVals, @NonNull String... variables) {
        Preconditions.checkArgument(variables.length > 0, "No variables were specified");
        return calculateGradients(placeholderVals, Arrays.asList(variables));
    }

    /**
     * Calculate and return the gradients for the specified variables
     *
     * @param placeholderVals Placeholders. May be null
     * @param variables       Names of the variables that you want the gradient arrays for
     * @return Gradients as a map, keyed by the variable name
     */
    public Map<String, INDArray> calculateGradients(Map<String, INDArray> placeholderVals, @NonNull Collection<String> variables) {
        Preconditions.checkArgument(!variables.isEmpty(), "No variables were specified");
        OutAndGrad oag = calculateGradientsAndOutputs(placeholderVals, null, variables);
        return oag.getGradients();
    }

    /**
     * Calculate the activations and the gradients for the specified variables, in one execution call.
     * This is equivalent to calling {@link #output(Map, List)} and {@link #calculateGradients(Map, Collection)}, but
     * is more efficient than calling both separately.
     *
     * @param placeholderVals Placeholders. May be null
     * @param outputVars      Names of the variables that you want the activations/outputs for. May be null
     * @param gradientVars    Names of the variables that you want the gradient arrays for. May be null
     * @return Activations and gradients, keyed by variable name
     */
    public OutAndGrad calculateGradientsAndOutputs(Map<String,INDArray> placeholderVals, Collection<String> outputVars, Collection<String> gradientVars){
        Preconditions.checkArgument((outputVars != null && !outputVars.isEmpty()) || (gradientVars != null && !gradientVars.isEmpty()),
                "No variables were specified for either output or gradients");
        if (getFunction(GRAD_FN_KEY) == null) {
            createGradFunction();
        }

        List<String> varNames = new ArrayList<>();
        if(outputVars != null){
            varNames.addAll(outputVars);
        }
        if(gradientVars != null) {
            for (String s : gradientVars) {
                Preconditions.checkState(this.variables.containsKey(s), "No variable with name \"%s\" exists in the SameDiff instance", s);
                SDVariable v = getVariable(s).getGradient();
                if (v != null) {
                    //In a few cases (like loss not depending on trainable parameters) we won't have gradient array for parameter variable
                    varNames.add(v.name());
                }
            }
        }

        //Key is gradient variable name
        SameDiff gradFn = getFunction(GRAD_FN_KEY);
        gradFn.setListeners(listeners);
        Map<String, INDArray> grads = gradFn.batchOutputHelper(placeholderVals, null, Operation.TRAINING, varNames.toArray(new String[0]));

        Map<String, INDArray> outOutputs = outputVars == null ? null : new HashMap<String,INDArray>();
        Map<String, INDArray> outGrads = gradientVars == null ? null : new HashMap<String,INDArray>();
        if(outputVars != null){
            for(String s : outputVars){
                outOutputs.put(s, grads.get(s));
            }
        }
        if(gradientVars != null) {
            for (String s : gradientVars) {
                if (getVariable(s).getGradient() != null) {
                    String gradVar = getVariable(s).getGradient().name();
                    outGrads.put(s, grads.get(gradVar));
                }
            }
        }

        return new OutAndGrad(outOutputs, outGrads);
    }

    /**
     * Returns true if the gradient function has been created - i.e., {@link #createGradFunction()} or {@link #createGradFunction(String...)}
     * has been called at all
     *
     * @return True if gradient (backprop) function exists
     */
    public boolean hasGradientFunction() {
        return sameDiffFunctionInstances.containsKey(GRAD_FN_KEY);
    }

    /**
     * Create the gradient function (for calculating gradients via {@link #calculateGradients(Map, Collection)}) if it is not already defined.
     * Users do not usually need to call this function manually, as it is called as required in the aforementioned method.
     * <br><br>
     * If the gradient function already exists, this method is a no-op.<br>
     * After this method returns, the SameDiff function instance for the gradient can be accessed using {@link #getFunction(String)}
     * with name "grad" as the argument.<br>
     * Note that the gradient array (after execBackwards has been called) can be accessed via {@code SDVariable.gradient().getArr()}
     */
    public void createGradFunction() {
        createGradFunction((String[]) null);
    }

    /**
     * As per {@link #createGradFunction()}, but this method allows a set of variables requiring gradients to be specified.
     * By default, only parameter gradients will be calculated; placeholder gradients may not be defined (unless they happen
     * to be calculated in the same op as calculating a parameter gradient.
     * This method allows you to override this behaviour by passing the name of the placeholder you want the gradients for.
     * The specified gradient variables still need to be floating point variables.
     *
     * @param variablesRequiringGradients May be null. If non-null: the gradients for the variables with these names will
     *                                    be calculated and available after backprop has been done
     */
    public void createGradFunction(final String... variablesRequiringGradients) {
        if (lossVariables.isEmpty()) {
            if (trainingConfig != null && trainingConfig.getLossVariables() != null && !trainingConfig.getLossVariables().isEmpty()) {
                lossVariables.addAll(trainingConfig.getLossVariables());
            } else {
                List<String> lossInferred = bestGuessLossVariables();
                if (lossInferred.size() == 1) {
                    String outName = lossInferred.get(0);
                    String opName = variables.get(outName).getOutputOfOp();
                    if (opName == null || !(ops.get(opName).getOp() instanceof ExternalErrorsFunction)) {
                        log.info("Inferring output \"{}\" as loss variable as none were previously set." +
                                "Use SameDiff.setLossVariables() or SDVariable.markAsLoss() to override", lossInferred.get(0));
                    }
                    lossVariables.add(lossInferred.get(0));
                } else if(lossInferred.isEmpty()){
                    //Check for external errors function
                    for(SameDiffOp o : ops.values()){
                        if(o.getOp() instanceof ExternalErrorsFunction){
                            List<String> l = o.getOutputsOfOp();
                            lossVariables.add(l.get(0));
                        }
                    }
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

        if (variablesRequiringGradients != null && variablesRequiringGradients.length > 0) {
            //Check that they are FP variables...
            for (String s : variablesRequiringGradients) {
                Preconditions.checkArgument(variables.containsKey(s), "Cannot ensure gradient exists for variable: no variable with name \"%s\" exists", s);
                DataType dt = variables.get(s).getVariable().dataType();
                Preconditions.checkState(dt.isFPType(), "Cannot ensure gradient exists for variable \"%s\": variable is not a floating point SDVariable." +
                        " Only floating point SDVariables have gradients defined - variable has type %s", s, dt);
            }
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
        Note that the user can also specify variables that they need gradients for (like placeholders) that normally wouldn't get gradients.

        Step 3: Differentiate ops in minimal subgraph
        The only major issue here is with multiple output ops, where only one of the outputs lead to the loss.
        For example, X -> slice -> (A,B); B -> loss, with A being unused (in that it doesn't contribute to the loss function)
        But to do split op backprop, we need gradient variables/arrays for both outputs (A and B).
        We know the shape and type of dL/dA must be exactly the same as A; we also know that the loss function doesn't depend on A. Hence, dL/dA == zerosLike(A)

         */


        final SameDiff outer = this;
        defineFunction(GRAD_FN_KEY, new SameDiffFunctionDefinition() {

            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.setArrayHolders(new SingleThreadArrayHolder(), new SingleThreadArrayHolder(), false);      //Training isn't thread safe, no need to use DeviceLocal, even with lazy init

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
                for (String s : lossVariables) {
                    Preconditions.checkNotNull(s, "Encountered null value in loss variables. Null loss variables are not allowed." +
                            " Use SameDiff.setLossVariables with non-null array names to fix");
                    Preconditions.checkState(variables.containsKey(s), "Specified loss function variable \"%s\" does not exist", s);
                    SDVariable v = variables.get(s).getVariable();
                    Preconditions.checkState(v.dataType().isFPType(), "Specified loss function variable \"%s\" is not a floating" +
                            "point variable (datatype: %s). Only floating point variables may be used as loss function variable", s, v.dataType());
                    v = v.sum();    //If output is not a scalar: we'll use loss = v.sum(), same as adding loss for multiple outputs. We don't always know for sure if output is scalar at this point
                    if (v.dataType() == initialGrad.dataType()) {
                        sameDiff.setGradientForVariableName(v.name(), initialGrad);
                    } else {
                        sameDiff.setGradientForVariableName(v.name(), initialGrad.castTo(v.dataType()));
                    }
                    if (finalOutputs.contains(v)) {
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
                // Find all FP variables that are connected to loss by an floating point (FP16/32/64) path
                Set<String> allFpVarsConnectedToLoss = new HashSet<>();
                Queue<String> toProcess = new LinkedList<>();
                for (String s : lossVariables) {
                    if (!toProcess.contains(s)) {
                        toProcess.add(s);
                    }
                }
                while (!toProcess.isEmpty()) {
                    String next = toProcess.remove();
                    if (!allFpVarsConnectedToLoss.contains(next)) {
                        Variable v = variables.get(next);
                        if (v.getVariable().dataType().isFPType()) {
                            allFpVarsConnectedToLoss.add(v.getName());
                            //Work out what op (if any) this is an output of... and add the inputs to that op to be processed
                            if (v.getOutputOfOp() != null) {
                                String opName = v.getOutputOfOp();
                                SameDiffOp op = ops.get(opName);
                                List<String> opInputs = op.getInputsToOp();
                                if (opInputs != null) {
                                    for (String s : opInputs) {
                                        Variable inputVar = variables.get(s);
                                        if (inputVar.getVariable().dataType().isFPType()) {
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
                Set<String> minimalSubgraphVars = new HashSet<>(allFpVarsConnectedToLoss);
                Queue<String> leafFPVars = new LinkedList<>();
                for (String s : allFpVarsConnectedToLoss) {
                    //First: determine if is a FP leaf (Array type SDVariable)
                    Variable v = variables.get(s);
                    if (v.getVariable().getVariableType() == VariableType.ARRAY) {
                        String opName = v.getOutputOfOp();  //Always defined for array type
                        SameDiffOp op = ops.get(opName);
                        List<String> inputsToOp = op.getInputsToOp();
                        boolean anyInputsInSubgraph = false;
                        if (inputsToOp != null) {
                            for (String s2 : inputsToOp) {
                                if (allFpVarsConnectedToLoss.contains(s2)) {
                                    //Connection s2 -> s exists... therefore s is not a leaf (yet)
                                    anyInputsInSubgraph = true;
                                    break;
                                }
                            }
                        }
                        if (!anyInputsInSubgraph) {
                            //Mark s as a leaf to be removed
                            leafFPVars.add(s);
                        }
                    }
                    VariableType vt = v.getVariable().getVariableType();
                    boolean isUserRequested = variablesRequiringGradients != null && ArrayUtils.contains(variablesRequiringGradients, s);
                    if ((vt == VariableType.CONSTANT || vt == VariableType.PLACEHOLDER) && !isUserRequested) {
                        leafFPVars.add(s);
                    }
                }

                while (!leafFPVars.isEmpty()) {
                    String nextLeaf = leafFPVars.remove();
                    Variable v = variables.get(nextLeaf);
                    minimalSubgraphVars.remove(nextLeaf);

                    //Now, after removing: check what this variable is input to...
                    //If nextLeaf is input to some op X, then if none of inputs y->X are present in subgraph, then
                    // output variables X->z must now be leafs
                    //Note that any time we remove a variable, the only possible new leafs are those that this one
                    // is connected to.
                    List<String> inputsTo = v.getInputsForOp();
                    if (inputsTo != null && !inputsTo.isEmpty()) {
                        for (String opName : inputsTo) {
                            SameDiffOp op = ops.get(opName);
                            List<String> inputsToOp = op.getInputsToOp();
                            boolean anyPresent = false;
                            for (String s : inputsToOp) {
                                if (minimalSubgraphVars.contains(s) || (variablesRequiringGradients != null && ArrayUtils.contains(variablesRequiringGradients, s))) {
                                    //Note second condition: means user explicitly specified that they want gradients for that input variable... hence we need to diff this op
                                    anyPresent = true;
                                    break;
                                }
                            }
                            if (!anyPresent) {
                                //All inputs to op X are not in subgraph. Therefore outputs of op must be new leaves
                                List<String> outVars = op.getOutputsOfOp();
                                if (outVars != null) {
                                    for (String s : outVars) {
                                        if (!leafFPVars.contains(s)) {
                                            //Mark this variable to be processed next
                                            leafFPVars.add(s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Preconditions.checkState(!minimalSubgraphVars.isEmpty(), "Cannot differentiate graph relative to the specified loss function variables %s:" +
                        " graph does not contain any trainable SDVariables (floating point VARIABLE type SDVariables) that the loss function depend on.", lossVariables);

                //At this point: we know the set of variables that are connected to the loss - these all (and only) need gradients
                Queue<String> availableForDiff = new LinkedList<>();
                for (SDVariable lossVar : finalOutputs) {
                    Variable v = sameDiff.variables.get(lossVar.name());
                    if (v.getOutputOfOp() != null) {
                        String opName = v.getOutputOfOp();
                        availableForDiff.add(opName);
                    }
                }

                // Collect all the the ops that have to be traversed before we can conclude that the gradient for
                // a variable is fully available
                //For example, if we have  X -> op -> Y, and Y -> (A,B) we need gradient contribution from BOTH
                // Y->A and Y->B connections before we can do differentiation of op "op"
                final HashMap<String, List<String>> prerequisites = new HashMap<>();    //Key: variable name. Value: list of op names
                for (String var : minimalSubgraphVars) {
                    Variable variable = variables.get(var);
                    // Copy the collection, as the original one will be modified during backprop
                    final List<String> inputsForOp = variable.getInputsForOp();
                    if (inputsForOp != null) {
                        List<String> req = new ArrayList<>();
                        for (String opName : inputsForOp) {
                            //Need to filter ops here
                            //For example, if we have: var -> Op1, and var -> Op2
                            //we might not need to differentiate Op2 if output of Op2 doesn't impact loss function
                            SameDiffOp o = ops.get(opName);
                            List<String> opOutputs = o.getOutputsOfOp();
                            boolean anyOpOutputsRequired = false;
                            if (opOutputs != null) {
                                for (String s : opOutputs) {
                                    if (minimalSubgraphVars.contains(s)) {
                                        anyOpOutputsRequired = true;
                                        break;
                                    }
                                }
                            }
                            if (anyOpOutputsRequired) {
                                req.add(opName);
                            }
                        }
                        prerequisites.put(variable.getName(), req);
                    }
                }

                Set<String> differentiatedOps = new HashSet<>();
                while (!availableForDiff.isEmpty()) {
                    String dfName = availableForDiff.remove();
                    DifferentialFunction df = sameDiff.ops.get(dfName).getOp();

                    //Get the inputs and outputs of the op
                    List<String> inputsToOp;
                    List<String> outputsOfOp;
                    if (df instanceof GradientBackwardsMarker) {
                        SameDiffOp op = sameDiff.ops.get(df.getOwnName());
                        inputsToOp = op.getInputsToOp();
                        outputsOfOp = Collections.emptyList();
                    } else {
                        inputsToOp = sameDiff.ops.get(df.getOwnName()).getInputsToOp();
                        outputsOfOp = sameDiff.ops.get(df.getOwnName()).getOutputsOfOp();
                    }


                    //Get gradients for all output variables:
                    List<SDVariable> grads = new ArrayList<>();
                    for (String s : outputsOfOp) {
                        SDVariable v = sameDiff.getVariable(s);
                        SDVariable g = v.hasGradient() ? v.gradient() : null;

                        if (g == null) {
                            //If no gradient exists at this point, 3 possibilities:
                            // (a) we have a bug
                            // (b) output of this op isn't used in calculating the loss
                            // (c) output isn't a FP type
                            //In the FP case, we should create a zero variable to backprop, because we can't perform backprop
                            // for this op otherwise...
                            if (!v.dataType().isFPType()) {
                                grads.add(null);
                            } else {
                                //See "Step 3: Differentiate ops in minimal subgraph" above for explanation on why this should be zerosLike here...
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

                    //Check the inputs to this op, see if we can differentiate those ops now (and if so: add to queue)
                    for (String s : inputsToOp) {
                        Variable v = sameDiff.variables.get(s);
                        String opName = v.getOutputOfOp();
                        if (opName == null || differentiatedOps.contains(opName)) {
                            //Skip placeholder/constant etc; also skip if we've previously differentiated this op
                            continue;
                        }

                        //Next: we've just differentiated OpX
                        //For s -> OpX: we now have gradient for s after df.diff(grads) call earlier
                        //Now, do we also need to differentiate OpY, where OpY -> s?
                        //If any input variables x (x -> OpY) exist, if they are in the minimal subgraph, then we
                        // need to differentiate OpY too
                        //Note that just because we *need to* doesn't mean we *can* yet

                        boolean isRequiredOp = false;
                        SameDiffOp op = ops.get(opName);
                        if (op.getInputsToOp() != null) {
                            List<String> opInputs = op.getInputsToOp();
                            boolean anyInputsRequired = false;
                            for (String s2 : opInputs) {
                                if (minimalSubgraphVars.contains(s2)) {
                                    anyInputsRequired = true;
                                    break;
                                }
                            }
                            if (anyInputsRequired) {
                                if (!differentiatedOps.contains(op.getName())) {
                                    isRequiredOp = true;
                                }
                            }
                        }

                        if (!isRequiredOp) {
                            continue;
                        }

                        //Now that we know we need this op - check if we can actually differentiate it...
                        //We can differentiate it if, for all variables that are outputs of this op:
                        //(a) we have gradient already, OR
                        //(b) it's not a FP variable, OR
                        //(c) it's a FP variable but not one that the loss depends on
                        //Note that for "output array is used multiple times" case (i.e., X->opY->Y, X->opZ->Z) we need all gradient
                        // contributions - i.e., we need to have differentiated both opY and opZ

                        boolean allAvailable = true;
                        SameDiffOp o = sameDiff.ops.get(opName);
                        for (String opOutput : o.getOutputsOfOp()) {
                            Variable outVar = variables.get(opOutput);
                            if (outVar.getVariable().dataType().isFPType()) {
                                if (minimalSubgraphVars.contains(outVar.getName())) {
                                    //Need gradient for this variable to be available before we can differentiate
                                    if (outVar.getVariable().gradient() == null) {
                                        allAvailable = false;
                                        break;
                                    }
                                    //However, when a variable is used multiple times, we need ALL gradient contributions available:
                                    List<String> prereqs = prerequisites.get(outVar.getName());
                                    if (prereqs != null) {
                                        allAvailable &= differentiatedOps.containsAll(prereqs);
                                        if (!allAvailable)
                                            break;
                                    }
                                }
                                //If in't not in the minimal subgraph, loss doesn't depend on it, so we don't care about it
                            }
                        }

                        if (allAvailable && !availableForDiff.contains(o.getOp().getOwnName())) {
                            availableForDiff.add(o.getOp().getOwnName());
                        }
                    }
                }

                //Let's validate we actually differentiated everything correctly:
                for (String s : minimalSubgraphVars) {
                    if (lossVariables.contains(s))
                        continue;
                    SDVariable v = variables.get(s).getVariable();
                    SDVariable g = v.gradient();
                    if (g == null) {
                        throw new IllegalStateException("Error encountered during differentiation: no gradient for required variable \"" + s + "\" was calculated");
                    }
                }


                return new SDVariable[]{sameDiff.var(GRAD_FN_KEY, org.nd4j.linalg.api.buffer.DataType.FLOAT, 1)};
            }
        });

        associateSameDiffWithOpsAndVariables();
    }

    /**
     * Try to infer the loss variable/s (usually loss variables). Note that this is not reliable in general.
     */
    protected List<String> bestGuessLossVariables() {
        List<String> out = new ArrayList<>();
        for (Variable v : variables.values()) {
            if (v.getVariable().isConstant() || v.getVariable().isPlaceHolder() ||                   //Exclude constants and placeholders
                    (v.getInputsForOp() != null && !v.getInputsForOp().isEmpty()) ||                //Exclude variables that are inputs to ops
                    (v.getControlDepsForOp() != null && !v.getControlDepsForOp().isEmpty()) ||      //Exclude variables that are control dependency inputs to ops
                    (v.getControlDepsForVar() != null && !v.getControlDepsForVar().isEmpty())) {    //Exclude variables that are control dependency inputs to other variables (mainly for import of cond etc ops)
                continue;
            }

            //Also exclude assert etc ops - doesn't make sense to return these "outputs" to user
            if (v.getOutputOfOp() != null) {
                String opName = v.getOutputOfOp();
                SameDiffOp o = ops.get(opName);
                if (o.getOp() instanceof Assert) {
                    continue;
                }

                //A bit of a hack for TF import: some TF graphs have Switch ops, where the output of one branch isn't consumed
                // by any ops. Consequently, during execution this "output" might never be available. So we'll exclude the output of execution here
                // This applies to SameDiff while loops as well
                if (o.getOp() instanceof Switch) {
                    continue;
                }
            }


            out.add(v.getName());
        }
        return out;
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

        if (newVarName != null) {
            String nameScope = currentNameScope();
            if (nameScope != null) {
                if (!newVarName.startsWith(nameScope + "/")) {
                    newVarName = nameScope + "/" + newVarName;
                }
            }
        }

        if (newVarName != null && variables.containsKey(newVarName) && varToUpdate != variables.get(newVarName).getVariable()) {
            throw new IllegalStateException("Variable name \"" + newVarName + "\" already exists for a different SDVariable");
        }

        if (newVarName == null && variables.containsKey(varToUpdate.name())
                && variables.get(varToUpdate.name()).getVariable() != varToUpdate) {
            //Edge case: suppose we do m1=sd.mean(in), m2=sd.mean(m1) -> both initially have the name
            // "mean" and consequently a new variable name needs to be generated
            newVarName = generateNewVarName(varToUpdate.name(), 0);
        }

        if (newVarName == null || varToUpdate.name().equals(newVarName)) {
            return varToUpdate;
        }

        val oldVarName = varToUpdate.name();
        varToUpdate.setVarName(newVarName);
        renameVariable(oldVarName, newVarName);
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
    protected void associateSameDiffWithOpsAndVariables() {
        for (SDVariable var : variableMap().values()) {
            var.setSameDiff(this);
        }
//        for(DifferentialFunction df : functionInstancesById.values()){
        for (SameDiffOp op : ops.values()) {
            DifferentialFunction df = op.getOp();
            df.setSameDiff(this);

            //TODO: This is ugly but seemingly necessary
            //Finally, also set the SDVariable for each op
            //Otherwise: could have an op pointing to this SameDiff instance, but op's SDVariable's sameDiff field pointing
            // to another SameDiff instance. At which point, they could fetch shapes and arrays from some other instance
            // (i.e., not from this one that is currently executing)
            SDVariable[] args = df.args();
            if (args != null) {
                for (SDVariable arg : args) {
                    arg.setSameDiff(this);
                }
            }

            SDVariable[] outputs = df.outputVariables();
            if (outputs != null) {
                for (SDVariable out : outputs) {
                    out.setSameDiff(this);
                }
            }
        }
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
                0, 0, 0, 0, 0, 0, 0, 0, 0);

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

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * @param configuration       - ExecutorConfiguration to be embedded into serialized graph
     * @param includeUpdaterState If true: include the updater state (state for updaters such as Adam, Nesterov, AdaGrad etc)
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers(@NonNull ExecutorConfiguration configuration, boolean includeUpdaterState) {
        return asFlatBuffers(0, configuration, includeUpdaterState);
    }

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * @param configuration       - ExecutorConfiguration to be embedded into serialized graph
     * @param includeUpdaterState If true: include the updater state (state for updaters such as Adam, Nesterov, AdaGrad etc)
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers(long graphId, @NonNull ExecutorConfiguration configuration, boolean includeUpdaterState) {
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
        val idxForOps = new IdentityHashMap<DifferentialFunction, Integer>();
        List<SDVariable> allVars = variables();
        for (SDVariable variable : allVars) {
            INDArray arr = variable.getVariableType() == VariableType.ARRAY ? null : variable.getArr();
            log.trace("Exporting variable: [{}]", variable.name());

            //If variable is the output of some op - let's use the ONE index for exporting, and properly track the output
            // numbers. For example, unstack(x) -> y0, y1, y2 -> the y's should be say (3,0), (3,1), (3,2) NOT (4,0), (5,0), (6,0)
            String varName = variable.name();
            int varIdx;
            int outputNum;
            if (variables.get(varName).getOutputOfOp() != null) {
                //This variable is the output of a node
                DifferentialFunction df = ops.get(variables.get(varName).getOutputOfOp()).getOp();
                if (!idxForOps.containsKey(df)) {
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


            reverseMap.put(variable.name(), varIdx);

            log.trace("Adding [{}] as [{}]", variable.name(), varIdx);
            int shape = 0;
            int name = bufferBuilder.createString(variable.name());
            int array = 0;
            int id = IntPair.createIntPair(bufferBuilder, varIdx, outputNum);
            byte varType = (byte) variable.getVariableType().ordinal();
            if (variable.isConstant() || variable.isPlaceHolder() || variable.getVariableType() == VariableType.VARIABLE) {
                //Don't export array type (i.e., activations), these are always replaced/re-calculated on each step
                array = arr == null ? 0 : arr.toFlatArray(bufferBuilder);
            }

            if (variable.getVariableType() == VariableType.PLACEHOLDER) {
                val shp = variable.getShape();
                if(shp != null) {
                    //Some models may have no shape defined, not ever a placeholder type shape
                    shape = FlatVariable.createShapeVector(bufferBuilder, shp);
                }
            }

            int controlDeps = 0;
            int controlDepsForOp = 0;
            int controlDepsForVar = 0;
            Variable v = variables.get(varName);

            int[] cds = FlatBuffersMapper.mapOrNull(v.getControlDeps(), bufferBuilder);
            if(cds != null)
                controlDeps = FlatVariable.createControlDepsVector(bufferBuilder, cds);

            int[] cdsForOp = FlatBuffersMapper.mapOrNull(v.getControlDepsForOp(), bufferBuilder);
            if(cdsForOp != null)
                controlDepsForOp = FlatVariable.createControlDepForOpVector(bufferBuilder, cdsForOp);

            int[] cdsForVar = FlatBuffersMapper.mapOrNull(v.getControlDepsForVar(), bufferBuilder);
            if(cdsForVar != null)
                controlDepsForVar = FlatVariable.createControlDepsForVarVector(bufferBuilder, cdsForVar);


            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, FlatBuffersMapper.getDataTypeAsByte(variable.dataType()), shape,
                    array, -1, varType, controlDeps, controlDepsForOp, controlDepsForVar);
            flatVariables.add(flatVariable);
        }

        //add functions
        for (SameDiffOp op : ops.values()) {
            DifferentialFunction func = op.getOp();
            Integer fnId = idxForOps.get(func);
            flatNodes.add(FlatBuffersMapper.asFlatNode(this, func, bufferBuilder, variableList, reverseMap, forwardMap, framesMap, idCounter, fnId));
        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int numPlaceholders = 0;
        for (SDVariable v : variables()) {
            if (v.isPlaceHolder()) {
                numPlaceholders++;
            }
        }

        int[] placeholderOffsets = new int[numPlaceholders];
        if (numPlaceholders > 0) {
            int i = 0;
            for (SDVariable v : variables()) {
                if (!v.isPlaceHolder())
                    continue;
                placeholderOffsets[i++] = bufferBuilder.createString(v.name());
            }
        }
        int placeholdersOffset = FlatGraph.createPlaceholdersVector(bufferBuilder, placeholderOffsets);

        List<String> lossVars = getLossVariables();
        int[] lossVarOffsets = new int[lossVars == null ? 0 : lossVars.size()];
        for (int i = 0; i < lossVarOffsets.length; i++) {
            lossVarOffsets[i] = bufferBuilder.createString(lossVars.get(i));
        }
        int lossVarOffset = FlatGraph.createLossVariablesVector(bufferBuilder, lossVarOffsets);

        int trainingConfigOffset = 0;
        int updaterStateOffset = 0;
        if (trainingConfig != null) {
            String json = trainingConfig.toJson();
            trainingConfigOffset = bufferBuilder.createString(json);
        }
        if (includeUpdaterState && updaterMap != null && !updaterMap.isEmpty()) {
            int[] updaterOffsets = new int[updaterMap.size()];
            int updaterNum = 0;
            for (Map.Entry<String, GradientUpdater> g : updaterMap.entrySet()) {
                int paramNameOffset = bufferBuilder.createString(g.getKey());
                int stateKeyOffset = 0;
                int stateValuesOffset = 0;
                Map<String, INDArray> state = g.getValue().getState();
                if (state != null && !state.isEmpty()) {
                    int[] keysOffsets = new int[state.size()];
                    int[] valuesOffsets = new int[state.size()];
                    int i = 0;
                    for (Map.Entry<String, INDArray> e : state.entrySet()) {
                        keysOffsets[i] = bufferBuilder.createString(e.getKey());
                        valuesOffsets[i] = e.getValue().toFlatArray(bufferBuilder);
                        i++;
                    }

                    stateKeyOffset = UpdaterState.createUpdaterStateKeysVector(bufferBuilder, keysOffsets);
                    stateValuesOffset = UpdaterState.createUpdaterStateValuesVector(bufferBuilder, valuesOffsets);
                }
                updaterOffsets[updaterNum++] = UpdaterState.createUpdaterState(bufferBuilder, paramNameOffset, stateKeyOffset, stateValuesOffset);
            }

            updaterStateOffset = FlatGraph.createUpdaterStateVector(bufferBuilder, updaterOffsets);
        }

        int fg = FlatGraph.createFlatGraph(bufferBuilder, graphId, variablesOffset, nodesOffset, outputsOffset,
                configuration.getFlatConfiguration(bufferBuilder), placeholdersOffset, lossVarOffset, trainingConfigOffset, updaterStateOffset);
        bufferBuilder.finish(fg);

        synchronized (this) {
            for (Map.Entry<String, Integer> e : reverseMap.entrySet()) {
                this.variables.get(e.getKey()).setVariableIndex(e.getValue());
            }
        }
        return bufferBuilder.dataBuffer();
    }

    /**
     * See {@link #asFlatGraph(long, ExecutorConfiguration, boolean)}.
     *
     * Uses the default {@link ExecutorConfiguration} with output mode as
     * {@link OutputMode#VARIABLE_SPACE}, execution mode as {@link ExecutionMode#SEQUENTIAL},
     * with profiling disabled and gather timings enabled.
     */
    public FlatGraph asFlatGraph(boolean includeUpdaterState) {
        return FlatGraph.getRootAsFlatGraph(this.asFlatBuffers(includeUpdaterState));
    }

    /**
     * This method returns FlatGraph structure
     *
     * @param configuration
     * @param includeUpdaterState If true: include the updater state (state for updaters such as Adam, Nesterov, AdaGrad etc)
     * @return
     */
    public FlatGraph asFlatGraph(long graphId, ExecutorConfiguration configuration, boolean includeUpdaterState) {
        return FlatGraph.getRootAsFlatGraph(asFlatBuffers(graphId, configuration, includeUpdaterState));
    }

    /**
     * This method exports the current SameDiff instance into FlatBuffers format, returning the array ops and
     * all arrays as a ByteBuffer containing the FlatBuffers format data
     *
     * Uses the default {@link ExecutorConfiguration} with output mode as
     * {@link OutputMode#VARIABLE_SPACE}, execution mode as {@link ExecutionMode#SEQUENTIAL},
     * with profiling disabled and gather timings enabled.
     *
     * @param includeUpdaterState If true: include the updater state (state for updaters such as Adam, Nesterov, AdaGrad etc)
     * @return a ByteBuffer holding the exported FlatBuffers representation of the graph
     */
    public ByteBuffer asFlatBuffers(boolean includeUpdaterState) {
        val configuration = ExecutorConfiguration.builder()
                .outputMode(OutputMode.VARIABLE_SPACE)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .build();

        return asFlatBuffers(configuration, includeUpdaterState);
    }

    /**
     * Save the SameDiff instance to a file. Files can be loaded using {@link #load(File, boolean)}
     *
     * @param file             File to save to
     * @param saveUpdaterState If true: save the updater state (arrays etc for Adam, Nesterov, RmsProp etc). If false: don't save
     *                         the updater state. If you want to continue training after loading your model, this should be true,
     *                         however may increase the file size significantly.
     *                         If the network is to be used for inference only, set this to false to save space
     */
    public void save(@NonNull File file, boolean saveUpdaterState) {
        try {
            asFlatFile(file, saveUpdaterState);
        } catch (IOException e) {
            throw new RuntimeException("Error saving SameDiff instance to file", e);
        }
    }

    /**
     * As per {@link #save(File, boolean)} but the serialized SameDiff instance is written to the output stream instead.
     * Note that this temporarily saves to disk (using {@link ND4JFileUtils#createTempFile(String, String)} then copies all
     * file bytes to the stream
     *
     * @param outputStream Stream to write the serialized SameDiff instance to
     * @param saveUpdater  If true: save the updater state (arrays etc for Adam, Nesterov, RmsProp etc). If false: don't save
     *                     the updater state. If you want to continue training after loading your model, this should be true,
     *                     however may increase the file size significantly.
     *                     If the network is to be used for inference only, set this to false to save space.
     */
    public void save(@NonNull OutputStream outputStream, boolean saveUpdater) {
        File tempFile = ND4JFileUtils.createTempFile("SameDiffFile", "temp");
        try {
            save(tempFile, saveUpdater);
            if (!(outputStream instanceof BufferedOutputStream)) {
                outputStream = new BufferedOutputStream(outputStream);
            }
            try (OutputStream os = outputStream; InputStream is = new BufferedInputStream(new FileInputStream(tempFile))) {
                IOUtils.copy(is, os);
            } catch (IOException e) {
                throw new RuntimeException("Error writing to output stream (or reading from temp file)", e);
            }
        } finally {
            tempFile.delete();
        }
    }

    /**
     * Load the SameDiff instance previously saved with {@link #save(File, boolean)}
     *
     * @param file             The file to load the network from
     * @param loadUpdaterState If true - load the updater state (history etc for updaters such as Adam, Nesterov momentum, RMSProp etc).
     *                         For inference only, this should be false, as the updater state will take more memory, but
     *                         is not required for training.
     *                         If the network is to be trained further, this should be true.
     *                         The updater state can only be loaded if it was saved with the network.
     * @return The loaded SameDiff network
     */
    public static SameDiff load(@NonNull File file, boolean loadUpdaterState) {
        try {
            return fromFlatFile(file, loadUpdaterState);
        } catch (IOException e) {
            throw new RuntimeException("Error loading SameDiff instance from file", e);
        }
    }

    /**
     * As per {@link #load(File, boolean)} but the SameDiff instance
     *
     * @param is               Input stream to load the saved network from
     * @param loadUpdaterState If true - load the updater state (history etc for updaters such as Adam, Nesterov momentum, RMSProp etc).
     *                         For inference only, this should be false, as the updater state will take more memory, but
     *                         is not required for training.
     *                         If the network is to be trained further, this should be true.
     *                         The updater state can only be loaded if it was saved with the network.
     * @return The loaded SameDiff network
     */
    public static SameDiff load(@NonNull InputStream is, boolean loadUpdaterState) {
        File tempFile = ND4JFileUtils.createTempFile("SameDiffFile", "temp");
        try {
            try (OutputStream os = new BufferedOutputStream(new FileOutputStream(tempFile))) {
                IOUtils.copy(is, os);
            }
            return fromFlatFile(tempFile, loadUpdaterState);
        } catch (IOException e) {
            throw new RuntimeException("Error loading SameDiff instance from file", e);
        } finally {
            tempFile.delete();
        }
    }

    /**
     * This method converts SameDiff instance to FlatBuffers and saves it to file which can be restored later<br>
     * This includes the updater state, if applicable.
     *
     * Uses the default {@link ExecutorConfiguration} with output mode as
     * {@link OutputMode#VARIABLE_SPACE}, execution mode as {@link ExecutionMode#SEQUENTIAL},
     * with profiling disabled and gather timings enabled.
     *
     * @param file File to save the FlatBuffers serialized graph (including arrays) to
     */
    public void asFlatFile(@NonNull File file) throws IOException {
        asFlatFile(file, true);
    }

    /**
     * See {@link #asFlatFile(File, ExecutorConfiguration, boolean)}.
     *
     * Uses the default {@link ExecutorConfiguration} with output mode as
     * {@link OutputMode#VARIABLE_SPACE}, execution mode as {@link ExecutionMode#SEQUENTIAL},
     * with profiling disabled and gather timings enabled.
     */
    public void asFlatFile(@NonNull File file, boolean withUpdaterState) throws IOException {
        val fb = asFlatBuffers(withUpdaterState);
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }

    /**
     * This method converts SameDiff instance to FlatBuffers and saves it to file which can be restored later
     *
     * @param file                File to save the FlatBuffers serialized graph (including arrays) to
     * @param includeUpdaterState If true: include the updater state (state for updaters such as Adam, Nesterov, AdaGrad etc)
     */
    public void asFlatFile(@NonNull File file, @NonNull ExecutorConfiguration configuration, boolean includeUpdaterState) throws IOException {
        val fb = asFlatBuffers(configuration, includeUpdaterState);
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }


    /**
     * Create a {@link SameDiff} instance from a file, including the updater state
     * The method to save the file is {@link #save(File, boolean)}
     *
     * @param file the file to load from
     * @return the loaded same diff instance
     * @throws IOException
     */
    public static SameDiff fromFlatFile(@NonNull File file) throws IOException {
        return fromFlatFile(file, true);
    }

    /**
     * Create a {@link SameDiff} instance from a file, optionally also loading the updater state
     * The method to save the file is {@link #save(File, boolean)}
     *
     * @param file             the file to load from
     * @param loadUpdaterState If true, load the updater state (Adam etc state). For training, use true. For inference, use false
     * @return the loaded same diff instance
     * @throws IOException
     */
    public static SameDiff fromFlatFile(@NonNull File file, boolean loadUpdaterState) throws IOException {
        byte[] bytes;
        try (InputStream is = new BufferedInputStream(new FileInputStream(file))) {
            bytes = IOUtils.toByteArray(is);
        }

        ByteBuffer bbIn = ByteBuffer.wrap(bytes);
        return fromFlatBuffers(bbIn, loadUpdaterState);
    }

    /**
     * Create a {@link SameDiff}
     * instance from a byte buffers
     * instance.
     *
     * See {@link #fromFlatBuffers(ByteBuffer, boolean)}.  Loads updater state (loadUpdaterState is true).
     *
     * @param bbIn the input byte buffer
     * @return the created samediff instance
     * @throws IOException
     */
    public static SameDiff fromFlatBuffers(ByteBuffer bbIn) throws IOException {
        return fromFlatBuffers(bbIn, true);
    }

    /**
     * Create a {@link SameDiff}
     * instance from a byte buffers
     * instance.
     *
     * @param bbIn the input byte buffer
     * @param loadUpdaterState If true, load the updater state (Adam etc state). For training, use true. For inference, use false
     * @return the created samediff instance
     * @throws IOException
     */
    public static SameDiff fromFlatBuffers(ByteBuffer bbIn, boolean loadUpdaterState) throws IOException {

        FlatGraph fg = FlatGraph.getRootAsFlatGraph(bbIn);

        int numOps = fg.nodesLength();
        int numVars = fg.variablesLength();
        List<FlatNode> ops = new ArrayList<>(numOps);
        for (int i = 0; i < numOps; i++) {
            ops.add(fg.nodes(i));
        }
        List<FlatVariable> vars = new ArrayList<>(numVars);
        for (int i = 0; i < numVars; i++) {
            vars.add(fg.variables(i));
        }

//        FlatConfiguration conf = fg.configuration();

        /* Reconstruct the graph
        We'll do the reconstruction manually here, rather than using sd.var(...), so that we have more control
        over the final result.
         */

        SameDiff sd = SameDiff.create();

        //Reconstruct placeholders
        int numPlaceholders = fg.placeholdersLength();
        Set<String> ph = new LinkedHashSet<>();
        for (int i = 0; i < numPlaceholders; i++) {
            ph.add(fg.placeholders(i));
        }

        //Reconstruct variables:
        Map<Integer, SDVariable> varNodeIds = new HashMap<>();
        Map<Pair<Integer, Integer>, SDVariable> variablesByNodeAndOutNum = new HashMap<>();
        Map<String, List<SDVariable>> variablesByName = new HashMap<>();
        for (FlatVariable v : vars) {
            int shapeLength = v.shapeLength();
            long[] shape = new long[shapeLength];
            for (int i = 0; i < shapeLength; i++) {
                shape[i] = v.shape(i);
            }

            String n = v.name();

            byte dtypeByte = v.dtype();
            org.nd4j.linalg.api.buffer.DataType dtype = FlatBuffersMapper.getDataTypeFromByte(dtypeByte);

            //TODO Infer this properly! Could be constant, etc.
            VariableType vt = VariableType.values()[v.variabletype()];
            SDVariable var = new SDVariable(n, vt, sd, shape, dtype);
            sd.variables.put(n, Variable.builder().name(n).variable(var).build());
            Variable v2 = sd.variables.get(n);

            //Reconstruct control dependencies
            if(v.controlDepsLength() > 0){
                int num = v.controlDepsLength();
                List<String> l = new ArrayList<>(num);
                for( int i=0; i<num; i++ ){
                    l.add(v.controlDeps(i));
                }
                v2.setControlDeps(l);
            }
            if(v.controlDepForOpLength() > 0){
                int num = v.controlDepForOpLength();
                List<String> l = new ArrayList<>(num);
                for( int i=0; i<num; i++ ){
                    l.add(v.controlDepForOp(i));
                }
                v2.setControlDepsForOp(l);
            }

            if(v.controlDepsForVarLength() > 0){
                int num = v.controlDepsForVarLength();
                List<String> l = new ArrayList<>(num);
                for( int i=0; i<num; i++ ){
                    l.add(v.controlDepsForVar(i));
                }
                v2.setControlDepsForVar(l);
            }



            FlatArray fa = v.ndarray();
            if (fa != null && vt != VariableType.ARRAY) {
                INDArray arr;
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    arr = Nd4j.createFromFlatArray(fa);
                }
                sd.setArrayForVariable(n, arr);
            }

            IntPair id = v.id();    //First value: node (op) id. Second: output number
            variablesByNodeAndOutNum.put(new Pair<>(id.first(), id.second()), var);

            if (!variablesByName.containsKey(n)) {
                variablesByName.put(n, new ArrayList<SDVariable>());
            }

            List<SDVariable> list = variablesByName.get(n);
            list.add(var);
        }

        //Reconstruct ops:
        for (FlatNode fn : ops) {
            DifferentialFunction df = FlatBuffersMapper.fromFlatNode(fn);
            String name = fn.name();
            df.setSameDiff(sd);
            df.setOwnName(name);
            if (sd.ops.containsKey(name)) {
                sd.ops.get(name).setOp(df);
            } else {
                sd.ops.put(name, SameDiffOp.builder().name(name).op(df).build());
            }

            int outLength = fn.outputLength();
            int[] outs = new int[outLength];
            for (int i = 0; i < outLength; i++) {
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
            List<Pair<Integer, Integer>> intPairList = new ArrayList<>();
            for (int i = 0; i < inputPaired.length; i++) {
                inputPaired[i] = fn.inputPaired(i);
                intPairList.add(new Pair<>(inputPaired[i].first(), inputPaired[i].second()));
            }

            String[] inputNames = new String[inputPaired.length];
            for (int i = 0; i < inputPaired.length; i++) {
                int nodeId = inputPaired[i].first();
                int nodeOutNum = inputPaired[i].second();
                SDVariable varIn = variablesByNodeAndOutNum.get(new Pair<>(nodeId, nodeOutNum));
                if (varIn == null) {
                    //The variable corresponding to this op was not
                }
                inputNames[i] = varIn.name();
            }
            SameDiffOp op = sd.ops.get(df.getOwnName());
            op.setInputsToOp(Arrays.asList(inputNames));

            //Reconstruct control dependencies
            if (fn.controlDepsLength() > 0) {
                int l = fn.controlDepsLength();
                List<String> list = new ArrayList<>(l);
                for( int i=0; i<l; i++ ){
                    list.add(fn.controlDeps(i));
                }
                op.setControlDeps(list);
            }

            if (fn.varControlDepsLength() > 0) {
                int l = fn.varControlDepsLength();
                List<String> list = new ArrayList<>(l);
                for( int i=0; i<l; i++ ){
                    list.add(fn.varControlDeps(i));
                }
                op.setVarControlDeps(list);
            }

            if (fn.controlDepForLength() > 0) {
                int l = fn.controlDepForLength();
                List<String> list = new ArrayList<>(l);
                for( int i=0; i<l; i++ ){
                    list.add(fn.controlDepFor(i));
                }
                op.setControlDepFor(list);
            }


            //Record that input variables are input to this op
            for (String inName : inputNames) {
                Variable v = sd.getVariables().get(inName);
                if (v.getInputsForOp() == null) {
                    v.setInputsForOp(new ArrayList<String>());
                }
                if (!v.getInputsForOp().contains(df.getOwnName())) {
                    v.getInputsForOp().add(df.getOwnName());
                }
            }

            List<SDVariable> varsForOp = variablesByName.get(name);

            //Can't assume that variables for the op have all been defined. For example, if we export before execution in SameDiff
            //In theory, we can reconstruct the output variables (minus names) if we know the number of op outputs
            //And we can calculate the op outputs - in most cases - after the op has been created and parameters set
            int numOutputs = df.getNumOutputs();
            if (numOutputs <= 0) {
                numOutputs = fn.outputLength();
            }

            String[] varNames = null;
            if (varsForOp != null && varsForOp.size() == numOutputs) {
                varNames = new String[varsForOp.size()];
                for (int i = 0; i < varNames.length; i++) {
                    varNames[i] = varsForOp.get(i).name();
                    sd.getVariables().get(varNames[i]).setOutputOfOp(df.getOwnName());
                }
                sd.ops.get(df.getOwnName()).setOutputsOfOp(Arrays.asList(varNames));
            } else {
                //We're missing some variables...
                int outputNamesLength = fn.outputNamesLength();
                varNames = new String[outputNamesLength];
                for (int i = 0; i < outputNamesLength; i++) {
                    String n = fn.outputNames(i);
                    varNames[i] = n;
                    if (!sd.variables.containsKey(n)) {
                        //Need to create the variable - perhaps it wasn't exported. Note output of node -> can only be VARIABLE type
                        SDVariable var = new SDVariable(n, VariableType.VARIABLE, sd, null, null);
                        sd.variables.put(n, Variable.builder().name(n).variable(var).build());
                        variablesByNodeAndOutNum.put(new Pair<>(opId, i), var);
                    }
                    sd.getVariables().get(varNames[i]).setOutputOfOp(df.getOwnName());
                }
                sd.ops.get(df.getOwnName()).setOutputsOfOp(Arrays.asList(varNames));
            }

            //Check the op mapping int he variablesByNodeAndOutputNum
            //For multi-output ops, variables will have their own index, not related to the op index
            for (int i = 0; i < varNames.length; i++) {
                Pair<Integer, Integer> p = new Pair<>(opId, i);
                if (!variablesByNodeAndOutNum.containsKey(p)) {
                    variablesByNodeAndOutNum.put(p, sd.getVariable(varNames[i]));
                }
            }
        }

        //Reconstruct loss variables
        if (fg.lossVariablesLength() > 0) {
            for (int i = 0; i < fg.lossVariablesLength(); i++) {
                sd.addLossVariable(fg.lossVariables(i));
            }
        }

        //Reconstruct training config
        String tc = fg.trainingConfig();
        if (tc != null) {
            sd.trainingConfig = TrainingConfig.fromJson(tc);
        }

        if (loadUpdaterState) {
            //Reconstruct updater state
            if (fg.updaterStateLength() > 0) {
                sd.updaterMap = new HashMap<>();
                int n = fg.updaterStateLength();
                for (int i = 0; i < n; i++) {
                    UpdaterState us = fg.updaterState(i);
                    String name = us.paramName();
                    int nKeys = us.updaterStateKeysLength();
                    Map<String, INDArray> m = new HashMap<>();
                    for (int j = 0; j < nKeys; j++) {
                        String key = us.updaterStateKeys(j);
                        FlatArray fa = us.updaterStateValues(j);
                        INDArray stateArr = Nd4j.createFromFlatArray(fa);
                        m.put(key, stateArr);
                    }

                    //Initialize the updater
                    GradientUpdater gu = sd.trainingConfig.getUpdater().instantiate(m, false);
                    sd.updaterMap.put(name, gu);
                }

                sd.initializedTraining = true;
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
        val fb = asFlatBuffers(false);

        val graph = FlatGraph.getRootAsFlatGraph(fb);

        sb.append("\nExternal variables:\n\n");
        for (int e = 0; e < graph.variablesLength(); e++) {
            FlatVariable var = graph.variables(e);
            INDArray ndarray = null;
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                FlatArray fa = var.ndarray();
                if (fa != null) {
                    ndarray = Nd4j.createFromFlatArray(fa);
                }
            }

            sb.append(var.id().first())
                    .append(":<").append(var.name()).append("> ");
            if (ndarray == null) {
                sb.append("<no array>").append("; Values: ").append("<no array>").append(";\n");
            } else {
                sb.append(Arrays.toString(ndarray.shapeInfoDataBuffer().asInt())).append("; Values: ");
                if (ndarray.data() == null) {
                    //Empty array
                    sb.append("<empty array>");
                } else if (ndarray.dataType() == DataType.UTF8) {
                    sb.append("<string array>");
                } else {
                    if (ndarray.length() < 50) {
                        sb.append(Arrays.toString(ndarray.data().asFloat()).replaceAll(" ", ""));
                    } else {
                        //Array is too long - only tak. last few values...
                        sb.append("[");
                        for (int i = 0; i < 50; i++) {
                            if (i > 0)
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
            FlatNode node = graph.nodes(e);

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
                IntPair pair = node.inputPaired(i);

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
        DifferentialFunction[] functions = ops();


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
                .append(String.format(format, "SameDiff Function Defs:", sameDiffFunctionInstances.size())).append("\n")
                .append("Loss function variables: ").append(getLossVariables())
                .append("\n\n");

        sb.append("--- Variables ---\n");
        //Work out which function - if any - this arg is an output of...
        Map<String, String> outputOfFn = new HashMap<>();
        int maxLengthOutputOf = 22;     //Length of "- Output Of Function -"
        int maxLengthOfName = 8;       //Length of "- Name -"
        for (String s : varMap.keySet()) {
            String outputOf = null;
            for (SameDiffOp op : ops.values()) {
                List<String> outputsOfOp = op.getOutputsOfOp();
                if (outputsOfOp != null && outputsOfOp.contains(s)) {
                    outputOf = op.getName();
                    break;
                }
            }

            if (outputOf == null) {
                outputOf = "<none>";
            } else {
                DifferentialFunction d = getOpById(outputOf);
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
            } else if (varMap.get(s).isPlaceHolder()) {
                SDVariable v = varMap.get(s);
                long[] phShape = v.placeholderShape();
                if (phShape != null) {
                    arrayShape = Arrays.toString(phShape);
                }
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
                int fns = (sd.ops() == null ? 0 : sd.ops().length);
                int defFns = sd.definedFunctionNames().size();

                sb.append(String.format(format, e.getKey(), String.valueOf(vars), String.valueOf(fns), String.valueOf(defFns))).append("\n");
            }
        }

        return sb.toString();
    }

    /**
     * For internal use only.
     * Creates a new discinct block name from baseName.
     * Block names are used by If and While
     */
    public String newBlockName(String baseName) {

        if (baseName == null)
            return null;

        if (!blockNames.contains(baseName)) {
            blockNames.add(baseName);
            return baseName;
        } else {
            int i = 1;
            while (blockNames.contains(baseName + "_" + i)) {
                i++;
            }
            blockNames.add(baseName + "_" + i);
            return baseName + "_" + i;
        }
    }

    /**
     * Import a frozen Tensorflow graph to a new SameDiff graph.
     *
     * @param graphFile The text or binary file containing the graph
     * @return The imported graph
     */
    public static SameDiff importFrozenTF(File graphFile) {
        return TFGraphMapper.importGraph(graphFile);
    }

    /**
     * See {@link #importFrozenTF(File)}
     */
    public static SameDiff importFrozenTF(GraphDef graphDef) {
        return TFGraphMapper.importGraph(graphDef);
    }


    /**
     * See {@link #importFrozenTF(File)}
     * <p>
     * Again, the input can be text or binary.
     */
    public static SameDiff importFrozenTF(InputStream graph) {
        return TFGraphMapper.importGraph(graph);
    }


    /**
     * Generate a new, distinct op name of the form &lt;base&gt;_#.
     * <p>
     * Applies name scope if active.
     *
     * @param base  The base name to use
     * @param force Whether to force the result name to be the same as base.
     */
    public String getOpName(String base, boolean force) {

        base = nameWithScope(base);

        if (force && ops.containsKey(base))
            throw new IllegalArgumentException("Op with name \"" + base + "\" already exists");
        else if (force)
            return base;

        int start = 1;

        // if we already have a name like "op_2", start from trying "op_3"
        if (base.contains("_") && base.matches(".*_\\d+")) {
            // extract number used to generate base
            Matcher num = Pattern.compile("(.*)_(\\d+)").matcher(base);
            // extract argIndex used to generate base
            if (num.find()) {
                start = Integer.parseInt(num.group(2));
                base = num.group(1);
            }
        }

        String name = base;
        for (int i = start; true; i++) {

            // ensure that there are no variables that look like they are outputs of this op
            boolean varWithName = false;
            for (String varName : variables.keySet())
                if (varName.startsWith(name + ":") || varName.equals(name))
                    varWithName = true;

            if (!ops.containsKey(name) && !varWithName)
                break;

            name = base + "_" + i;
        }
        return name;
    }

    /**
     * See {@link #getOpName(String, boolean)}
     * force is false
     */
    public String getOpName(String base) {
        return getOpName(base, false);
    }

    /**
     * Generate a new, distinct variable name of the form &lt;base&gt;_#[:#].
     * <p>
     * Applies name scopes if active.
     *
     * @param base       The base of the name.
     * @param argIndex   The argument index, used in the ":#".  A value of 0 (or negative) does not include the ":#" part.
     * @param existingOp Whether to generate an distinct operation name from base (if false), or just use base (if true).
     */
    public String generateNewVarName(String base, int argIndex, boolean existingOp) {

        base = nameWithScope(base);

        if (argIndex > 0 && base.contains(":")) {
            Matcher num = Pattern.compile("(.*):(\\d+)").matcher(base);
            // extract argIndex used to generate base
            if (num.find()) {
                argIndex = Integer.parseInt(num.group(2)) + 1;
                base = num.group(1);
            }
        }

        if (!existingOp)
            base = getOpName(base);

        if (argIndex > 0)
            base += ":" + argIndex;

        if (variables.containsKey(base))
            throw new IllegalArgumentException("Variable with name \"" + base + "\" already exists");

        return base;
    }

    /**
     * See {@link #generateNewVarName(String, int, boolean)}
     * existingOp is true.
     */
    @Override
    public String generateNewVarName(String base, int argIndex) {
        return generateNewVarName(base, argIndex, true);
    }

    /**
     * Returns an unused variable name of the format &lt;base&gt;_#.
     *
     * Intended to be used for custom variables (like weights), arguments and op outputs should use {@link #generateNewVarName(String, int)}.
     */
    public String generateDistinctCustomVariableName(String base){
        if(!variables.containsKey(base))
            return base;

        int inc = 1;

        while(variables.containsKey(base + "_" + inc)){
            inc++;
        }

        return base + "_" + inc;
    }
}
