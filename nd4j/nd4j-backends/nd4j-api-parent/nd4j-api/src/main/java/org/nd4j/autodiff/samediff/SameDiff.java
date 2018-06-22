package org.nd4j.autodiff.samediff;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import com.rits.cloning.Cloner;
import com.rits.cloning.IFastCloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.FunctionProperties;
import org.nd4j.autodiff.samediff.flow.FlowPath;
import org.nd4j.autodiff.util.cloner.DataBufferFastCloner;
import org.nd4j.autodiff.util.cloner.INDArrayFastCloner;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;
import org.nd4j.linalg.api.ops.impl.shape.Eye;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.BaseTensorOp;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayV3;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4UnresolvedOutputVariables;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.list.compat.TensorList;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * SameDiff is the
 * entrypoint for
 * nd4j's autodiff.
 * <p>
 * You define a graph symbolically.
 * <p>
 * That graph accumulates operations.
 * <p>
 * In order to execute the graph, you run
 * {@link #exec()} to get all the operations
 * {@link #exec(List)} for an already created set of ops
 * {@link #execAndEndResult()} for the end result only
 * {@link #execAndEndResult(List)} for a cached set of ops
 */
@AllArgsConstructor
@Builder
@Slf4j
public class SameDiff {
    private Map<String, String[]> incomingArgsReverse;              //Key: DifferentialFunction.getOwnName(). Value: name of SDVariables as inputs to that function
    private Map<String, String[]> outgoingArgsReverse;              //Key: DifferentialFunction.getOwnName(). Value: name of SDVariables as outputs from that function
    private Map<String, int[]> permuteOrder;
    private boolean shouldBootStrap = true;
    private Set<String> importedVarName;
    //map a function's instance id to a base name, used for propagating variable names
    //for output during import
    private Map<String, String> baseNameForFunctionInstanceId;

    private DifferentialFunctionFactory functionFactory;
    private Map<String, SDVariable> variableMap;                    //Key: SDVariable name. Value: SDVariable
    private Map<String, long[]> variableNameToShape;                //Key: SDVariable name. Value: shape for that variable
    //gradient information
    private Map<String, SDVariable> gradients;                      //Key:
    private Map<String, SDVariable> forwardVarForGrad;

    private Map<String, INDArray> variableNameToArr;                //Key: name of SDVariable. Value: Array for that variable

    //individual index for variable names
    private Map<String, List<DifferentialFunction>> functionsArgsFor;   //Key: SDVariable name. Value: all DifferentialFunctions it is an input to
    private Map<String, List<DifferentialFunction>> functionOutputFor;  //Key: SDVariable name. Value: DifferentialFunctions this variable is an output for (TODO: Why is this a list? Isn't it *always* length 1?)

    private Map<String, TensorList> lists = new HashMap<>();    // Key - node name; Value - TensorList

    // this entity holds runtime information for Switch/Merge/NextIteration etc stuff
    private transient ThreadLocal<FlowPath> localFlowPath = new ThreadLocal<FlowPath>();

    // here we save String -> Integer conversion to variables
    private transient Map<String, Integer> reverseMap = null;


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


    private Map<String, List<String[]>> placeHolderMap;
    private Map<String, long[]> placeHolderOriginalShapes;
    private Set<String> placeHolderVarNames;
    private IdentityHashMap<INDArray, SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String, SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String, SameDiff> sameDiffFunctionInstances;
    private Set<String> placeHolderFunctions;
    private static Cloner cloner = newCloner();
    private static Map<String, Method> opMethods;

    private Map<String, DifferentialFunction> functionInstancesById;

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


    static {
        opMethods = new HashMap<>();
        Method[] methods = SameDiff.class.getDeclaredMethods();
        for (Method method : methods) {
            if (method.getReturnType().equals(SDVariable.class)) {
                opMethods.put(method.getName(), method);
            }
        }
    }

    public static Cloner newCloner() {
        Cloner cloner = new Cloner();

        //Implement custom cloning for INDArrays (default can have problems with off-heap and pointers)
        //Sadly: the cloner library does NOT support interfaces here, hence we need to use the actual classes
        //cloner.registerFastCloner(INDArray.class, new INDArrayFastCloner());  //Does not work due to interface
        IFastCloner fc = new INDArrayFastCloner();
        cloner.registerFastCloner(Nd4j.getBackend().getNDArrayClass(), fc);
        cloner.registerFastCloner(Nd4j.getBackend().getComplexNDArrayClass(), fc);

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
        variableMap.remove(oldVarNameRef.getVarName());
        val oldVarName = varName;
        oldVarNameRef.setVarName(withName);
        variableMap.put(withName, oldVarNameRef);


        for (val reverseValues : outgoingArgsReverse.entrySet()) {
            for (int i = 0; i < reverseValues.getValue().length; i++) {
                if (reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }


        for (val reverseValues : incomingArgsReverse.entrySet()) {
            for (int i = 0; i < reverseValues.getValue().length; i++) {
                if (reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }

        if (variableNameToArr.containsKey(oldVarName)) {
            val arr = variableNameToArr.remove(oldVarName);
            variableNameToArr.put(withName, arr);
        }


        if (variableNameToShape.containsKey(oldVarName)) {
            val shape = variableNameToShape.remove(oldVarName);
            variableNameToShape.put(withName, shape);
        }


        if (gradients.containsKey(oldVarName)) {
            val grad = gradients.remove(oldVarName);
            gradients.put(withName, grad);
        }

        if (forwardVarForGrad.containsKey(oldVarName)) {
            val forwardGrad = forwardVarForGrad.remove(oldVarName);
            forwardVarForGrad.put(withName, forwardGrad);
        }

        if (placeHolderMap.containsKey(oldVarName)) {
            val placeholders = placeHolderMap.remove(oldVarName);
            placeHolderMap.put(withName, placeholders);
        }


        if (functionsArgsFor.containsKey(oldVarName)) {
            val funcs = functionsArgsFor.remove(oldVarName);
            for (val func : funcs) {
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

            functionsArgsFor.put(withName, funcs);
        }


        if (functionOutputFor.containsKey(oldVarName)) {
            val funcs = functionOutputFor.remove(oldVarName);
            for (val func : funcs) {
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

            functionOutputFor.put(withName, funcs);
        }

        variableMap.remove(oldVarName);


    }


    /**
     * Clears debugging state
     * and disables debug mode.
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
     * Returns this samediff instance's
     * {@link DifferentialFunctionFactory}
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
            val clone = cloner.deepCloneDontCloneInstances(var, var.getSameDiff());
            val newVar = sameDiff.var(clone);
            if (var.getArr() != null) {
                sameDiff.associateArrayWithVariable(var.getArr(), newVar);
            }


            thisVertexIdToNew.put(idx, idx);
            clone.setSameDiff(sameDiff);
            idx++;

        }


        val newFunctions = new LinkedHashMap<String, DifferentialFunction>();
        for (DifferentialFunction function : functionInstancesById.values()) {
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

            sameDiff.functionInstancesById.put(function.getOwnName(), function);
        }

        for (val reverseArrayEntry : reverseArrayLookup.entrySet()) {
            sameDiff.reverseArrayLookup.put(reverseArrayEntry.getKey(), sameDiff.getVariable(reverseArrayEntry.getValue().getVarName()));
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
        return functionInstancesById.containsKey(id);
    }


    /**
     * Get the function by the {@link DifferentialFunction#getOwnName()}
     *
     * @param id the id of the function
     * @return the function for the given id if it exists
     */
    public DifferentialFunction getFunctionById(String id) {
        if (!functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("No function with id " + id + " found!");
        }
        return functionInstancesById.get(id);
    }


    /**
     * Put the function for id
     *
     * @param id       the id
     * @param function the function
     */
    public void putFunctionForId(String id, DifferentialFunction function) {
        if (functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("Function by id already exists!");
        } else if (function instanceof SDVariable) {
            throw new ND4JIllegalStateException("Function must not be a variable!");
        }

        functionInstancesById.put(id, function);
    }


    /**
     * Returns the inputs for the given function
     *
     * @param function the function to get the
     *                 inputs for
     * @return the input ids for a given function
     */
    public String[] getInputsForFunction(DifferentialFunction function) {
        if (!incomingArgsReverse.containsKey(function.getOwnName()))
            throw new ND4JIllegalStateException("Illegal function instance id found " + function.getOwnName());
        return incomingArgsReverse.get(function.getOwnName());
    }

    /**
     * Returns the outputs for the given function
     *
     * @param function the function to get the
     *                 inputs for
     * @return the outputs ids for a given function
     */
    public String[] getOutputsForFunction(DifferentialFunction function) {
        return outgoingArgsReverse.get(function.getOwnName());
    }


    /**
     * Get the output variables given a set of ids
     * from {@link #getOutputsForFunction(DifferentialFunction)}
     *
     * @param function the function reference to get the id for
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
     * Get the input variables given a set of ids
     * from {@link #getInputVariablesForFunction(DifferentialFunction)}
     *
     * @param function the function reference to get the id for
     * @return the output variables for the given function
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


    /**
     * Update the ndarray for the given vertex id.
     *
     * @param varName
     * @param arr
     * @throws {@link ND4JIllegalStateException} when the array does not exist.
     */
    public void updateArrayForVarName(String varName, INDArray arr) {
        if (!variableNameToArr.containsKey(varName)) {
            throw new ND4JIllegalStateException("Array for " + varName + " does not exist. Please use putArrayForVertexId instead.");
        }

        variableNameToArr.put(varName, arr);
        reverseArrayLookup.put(arr, getVariable(varName));
    }

    /**
     * Adds an ndarray for a given vertex id.
     * Use {@link #updateArrayForVarName(String, INDArray)}
     * if the array already exists.
     *
     * @param varName the vertex id to add
     * @param arr     the array to add
     * @throws {@link ND4JIllegalStateException} when the array already exists.
     */
    public void putArrayForVarName(String varName, INDArray arr) {
        if (varName == null)
            throw new ND4JIllegalStateException("No null names allowed!");

        if (variableNameToArr.containsKey(varName)) {
//            throw new ND4JIllegalStateException("Array for " + varName + " already exists!");
///            return;
        }

        variableNameToArr.put(varName, arr);
    }


    /**
     * Get the shape for the given vertex id.
     * Note that if an array is defined, it will use that shape instead.
     * <p>
     * A shape *and* an array should not be defined at the same time.
     * This wastes memory. The internal map used for tracking shapes for particular
     * vertex ids should also delete redundant shapes stored to avoid redundant sources of information.
     *
     * @param varName the vertex id to get the shape for
     * @return the shape for the given vertex if if any.
     */
    public long[] getShapeForVarName(String varName) {
        if (variableNameToArr.containsKey(varName)) {
            return variableNameToArr.get(varName).shape();
        }

        return variableNameToShape.get(varName);
    }


    /**
     * Update a vertex id with the given shape.
     * Note that you should use {@link #putShapeForVarName(String, long[])}
     * if you want to add a new shape.
     * Update is meant to be an in place replacement
     * of the shape for the vertex id *only*.
     *
     * @param varName the vertex id to associate
     * @param shape   the shape to associate with
     */
    public void updateShapeForVarName(String varName, long[] shape) {
        updateShapeForVarName(varName, shape, false);
    }

    public void updateShapeForVarName(String varName, long[] shape, boolean clearArrayOnShapeMismatch) {
        if (shape == null) {
            throw new ND4JIllegalStateException("Null shapes not allowed!");
        }

        if (variableNameToArr.containsKey(varName) && !Arrays.equals(variableNameToArr.get(varName).shape(), shape)) {
            if(clearArrayOnShapeMismatch){
                if(log.isTraceEnabled()){
                    log.trace("Clearing array for variable {}: array shape {}, new shape {}", varName,
                            Arrays.toString(variableNameToArr.get(varName).shape()), Arrays.toString(shape));
                }
                variableNameToArr.remove(varName);
            } else {
                throw new ND4JIllegalStateException("Already found an existing array!");
            }
        }


        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                addAsPlaceHolder(varName);
                placeHolderOriginalShapes.put(varName, shape);
                return;
            }
        }


        variableNameToShape.put(varName, shape);
    }


    /**
     * Associate a vertex id with the given shape.
     *
     * @param varName the vertex id to associate
     * @param shape   the shape to associate with
     */
    public void putShapeForVarName(String varName, long[] shape) {
        if (shape == null) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }

        if (variableNameToShape.containsKey(varName)) {
            throw new ND4JIllegalStateException("Shape for " + varName + " already exists!");
        }

        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                addAsPlaceHolder(varName);
                placeHolderOriginalShapes.put(varName, shape);
                return;
            }
        }

        variableNameToShape.put(varName, shape);
    }

    public void putOrUpdateShapeForVarName(String varName, @NonNull long[] shape, boolean clearArrayOnShapeMismatch){
        if(variableNameToArr.containsKey(varName)){
            updateShapeForVarName(varName, shape, clearArrayOnShapeMismatch);
        } else {
            putShapeForVarName(varName, shape);
        }
    }


    /**
     * Returns true if the given vertex id
     * and shape already exist.
     *
     * @param varName the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean shapeAlreadyExistsForVarName(String varName) {
        return variableNameToShape.containsKey(varName) || arrayAlreadyExistsForVarName(varName);
    }


    /**
     * Returns true if the given vertex id
     * and {@link INDArray} already exist.
     *
     * @param varName the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean arrayAlreadyExistsForVarName(String varName) {
        return variableNameToArr.containsKey(varName);
    }

    /**
     * Get an {@link INDArray}
     * for a given vertex id
     *
     * @param varName
     * @return
     */
    public INDArray getArrForVarName(String varName) {
        return variableNameToArr.get(varName);
    }

    /**
     * Associate the array with the given variable.
     *
     * @param arr      the array to get the variable for
     * @param variable the variable to associate
     */
    public void associateArrayWithVariable(INDArray arr, @NonNull String variable) {
        associateArrayWithVariable(arr, this.getVariable(variable));
    }

    /**
     * Associate the array with the given variable.
     *
     * @param arr      the array to get the variable for
     * @param variable the variable to associate
     */
    public void associateArrayWithVariable(INDArray arr, SDVariable variable) {
        if (variable == null) {
            throw new ND4JIllegalArgumentException("Variable must not be null!");
        }

        if (arr == null) {
            throw new ND4JIllegalArgumentException("Array must not be null");
        }

        reverseArrayLookup.put(arr, variable);
        variableNameToArr.put(variable.getVarName(), arr);
        if (!shapeAlreadyExistsForVarName(variable.getVarName()))
            putShapeForVarName(variable.getVarName(), arr.shape());
        else {
            updateShapeForVarName(variable.getVarName(), arr.shape());
        }
        // invalidate exec cache
        exec_cache = null;

        //Also update nested SameDiff instances (such as gradient function)
        if(sameDiffFunctionInstances != null && sameDiffFunctionInstances.size() > 0){
            for(Map.Entry<String,SameDiff> e : sameDiffFunctionInstances.entrySet()){
                SameDiff sd = e.getValue();
                if(sd.variableNameToArr != null && sd.variableNameToArr.containsKey(variable.getVarName())){
                    sd.associateArrayWithVariable(arr, variable);
                }
            }
        }
    }


    /**
     * Associate a {@link SameDiff}
     * namespace as a sub function.
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
     * Return the internal variable map
     *
     * @return
     */
    public Map<String, SDVariable> variableMap() {
        return variableMap;
    }


    /**
     * Invoke an op by opName
     *
     * @param op the op
     * @param x  the first input
     * @param y  the second input
     * @return the result variable
     */
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
     * Get an {@link SDVariable}
     * for an array reference.
     * Internally samediff associates array references
     * with variables. This will typically be a shortcut
     * for the array associated with {@link SDVariable#getArr()}
     *
     * @param arr the array reference
     * @return the variable if one exists
     */
    public SDVariable getVariableForArray(INDArray arr) {
        return reverseArrayLookup.get(arr);
    }


    /**
     * The set of defined function names
     *
     * @return
     */
    public Collection<String> definedFunctionNames() {
        return this.sameDiffFunctionInstances.keySet();
    }


    /**
     * Returns the number of bytes
     * for the graph
     *
     * @return
     */
    public long memoryForGraph() {
        return numElements() * DataTypeUtil.lengthForDtype(Nd4j.dataType());
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
        variableMap = new LinkedHashMap<>();
        sameDiffFunctionDefinitionMap = new LinkedHashMap<>();
        sameDiffFunctionInstances = new LinkedHashMap<>();
        gradients = new LinkedHashMap<>();
        forwardVarForGrad = new LinkedHashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        reverseArrayLookup = new IdentityHashMap<>();
        variableNameToArr = new LinkedHashMap<>();
        variableNameToShape = new LinkedHashMap<>();
        placeHolderMap = new LinkedHashMap<>();
        placeHolderVarNames = new LinkedHashSet<>();
        placeHolderOriginalShapes = new LinkedHashMap<>();
        incomingArgsReverse = new LinkedHashMap<>();
        outgoingArgsReverse = new LinkedHashMap<>();
        functionInstancesById = new LinkedHashMap<>();
        placeHolderFunctions = new LinkedHashSet<>();
        functionsArgsFor = new LinkedHashMap<>();
        functionOutputFor = new LinkedHashMap<>();
        baseNameForFunctionInstanceId = new LinkedHashMap<>();
        importedVarName = new LinkedHashSet<>();
        permuteOrder = new LinkedHashMap<>();
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
     * Returns true if the given function
     * has ndarray properties to resolve.
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
     * Adds a field name -> variable name
     * mapping for a given function.
     * This is used for model import
     * where there is an unresolved variable
     * at the time of calling any
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
     * Returns true if the variable name is imported
     *
     * @param variableName the imported variable name
     * @return true if the name is imported, false otherwise
     */
    public boolean isImportVariable(String variableName) {
        return importedVarName.contains(variableName);
    }

    /**
     * Marks a variable name as imported.
     * This is used in conjunction with model
     * import to ensure immutability
     * when referencing graph variables
     * mapped from an external source.
     *
     * @param varName the var name to add.
     */
    public void addVarNameForImport(String varName) {
        importedVarName.add(varName);
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
     * Attempts to insert the {@link DifferentialFunction}
     * reference in to this {@link SameDiff}
     * instance.
     * If the given array field with the given
     * index already exists, it will do a reference
     * check to ensure that the 2 array fields are the same.
     * <p>
     * If not, an exception is thrown.
     * If the instances are the same (by semantics, not reference)
     * then it will just return the original instance.
     * This is to ensure that instances that are created are unique
     * and reference checked.
     *
     * @param function the array field to attempt to create
     * @return
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
     * Adds outgoing args to the graph
     *
     * @param variables
     * @param function
     */
    public void addOutgoingFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            varNames[i] = variables[i].getVarName();
        }

        addOutgoingFor(varNames, function);
    }


    /**
     * Adds outgoing arguments to the graph.
     * Also checks for input arguments
     * and updates the graph adding an appropriate edge
     * when the full graph is declared.
     *
     * @param varNames
     * @param function
     */
    public void addOutgoingFor(String[] varNames, DifferentialFunction function) {

        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        if (outgoingArgsReverse.containsKey(function.getOwnName())) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for " + function);
        }

        if (varNames == null)
            throw new ND4JIllegalStateException("Var names can not be null!");


        for (int i = 0; i < varNames.length; i++) {
            if (varNames[i] == null)
                throw new ND4JIllegalStateException("Variable name elements can not be null!");
        }

        outgoingArgsReverse.put(function.getOwnName(), varNames);

        for (val resultName : varNames) {
            List<DifferentialFunction> funcs = functionOutputFor.get(resultName);
            if (funcs == null) {
                funcs = new ArrayList<>();
                functionOutputFor.put(resultName, funcs);
            }

            funcs.add(function);
        }

    }

    /**
     * Adds incoming args to the graph
     *
     * @param variables
     * @param function
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

        incomingArgsReverse.put(function.getOwnName(), variables);
        for (val variableName : variables) {
            List<DifferentialFunction> funcs = functionsArgsFor.get(variableName);
            if (funcs == null) {
                funcs = new ArrayList<>();
                functionsArgsFor.put(variableName, funcs);
            }

            funcs.add(function);
        }

    }


    /**
     * Adds incoming args to the graph
     *
     * @param variables
     * @param function
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
        List<DifferentialFunction> list = functionOutputFor.get(variableName);
        if (list == null) {
            return null;
        }
        return list.get(0);
    }

    /**
     * Return a list of differential functions (if any) that this variable is the input argument for
     *
     * @param variableName Name of the variable
     * @return The differential functions that this variable is an input argument for, or null if it is not the input to any function
     */
    public List<DifferentialFunction> getVariableArgOfFunctions(String variableName) {
        return functionsArgsFor.get(variableName);
    }


    /**
     * Returns true if this function already
     * has defined arguments
     *
     * @param function the function to check
     * @return true if the function has args false otherwise
     */
    public boolean hasArgs(DifferentialFunction function) {
        String[] vertexIdArgs = incomingArgsReverse.get(function.getOwnName());
        return vertexIdArgs != null && vertexIdArgs.length > 0;
    }


    public DifferentialFunction[] functions() {
        val ret = functionInstancesById.values();
        return ret.toArray(new DifferentialFunction[ret.size()]);
    }


    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (variableMap != null ? variableMap.hashCode() : 0);
        return result;
    }


    /**
     * @param originalSameDiff
     * @return
     */
    public static SameDiff create(SameDiff originalSameDiff) {
        SameDiff ret = SameDiff.builder()
                .variableMap(originalSameDiff.variableMap)
                .sameDiffFunctionInstances(originalSameDiff.sameDiffFunctionInstances)
                .build();
        //ensuring proper sameDiff reference
        DifferentialFunctionFactory differentialFunctionFactory =
                new
                        DifferentialFunctionFactory(ret);
        ret.functionFactory = differentialFunctionFactory;
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SameDiff sameDiff = (SameDiff) o;

        if (variableMap != null ? !variableMap.equals(sameDiff.variableMap) : sameDiff.variableMap != null)
            return false;
        if (sameDiffFunctionDefinitionMap != null ? !sameDiffFunctionDefinitionMap.equals(sameDiff.sameDiffFunctionDefinitionMap) : sameDiff.sameDiffFunctionDefinitionMap != null)
            return false;
        return sameDiffFunctionInstances != null ? sameDiffFunctionInstances.equals(sameDiff.sameDiffFunctionInstances) : sameDiff.sameDiffFunctionInstances == null;
    }

    /**
     * @return
     */
    public static SameDiff create() {
        return new SameDiff();
    }


    /**
     * Evaluate the given inputs
     * based on the current graph
     *
     * @param inputs the inputs to evaluate
     * @return
     */
    public INDArray[] eval(Map<String, INDArray> inputs) {

        SameDiff execPipeline = dup();

        List<DifferentialFunction> opExecAction = execPipeline.exec().getRight();
        if (opExecAction.isEmpty())
            throw new IllegalStateException("No ops found to execute.");
        INDArray[] ret = new INDArray[opExecAction.size()];
        for (int i = 0; i < ret.length; i++) {
            val varName = opExecAction.get(i).outputVariables()[0].getVarName();
            ret[i] = execPipeline.getArrForVarName(varName);
        }
        return ret;
    }




    /**
     * @return
     */
    public SameDiff dup() {
        Cloner cloner = newCloner();
        val clone = cloner.deepClone(this);
        //clone.exec_cache = this.exec_cache;
        //clone.parent = this;
        return clone;

    }


    /**
     * @return
     */
    public long numElements() {
        long ret = 0;
        for (SDVariable variable : variables()) {
            ret += ArrayUtil.prod(variable.getShape());
        }

        return ret;
    }


    private void initWorkspace() {
        workspace = Nd4j.getWorkspaceManager().createNewWorkspace(
                WorkspaceConfiguration.builder()
                        .initialSize(memoryForGraph())
                        .policyAllocation(AllocationPolicy.OVERALLOCATE)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .build());
        Nd4j.getWorkspaceManager().setWorkspaceForCurrentThread(workspace);


    }


    /**
     * The list of available
     * variables in the graph
     *
     * @return
     */
    public List<SDVariable> variables() {
        return new ArrayList<>(variableMap.values());
    }

    /**
     * Variable initialization
     * with 1.0
     *
     * @param name  the opName of the variable
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, int[] shape) {
        return var(name, ArrayUtil.toLongArray(shape), new ConstantInitScheme('f', 1.0));
    }

    public SDVariable one(String name, long[] shape) {
        return var(name, shape, new ConstantInitScheme('f', 1.0));
    }

    /**
     * Return a variable of all 1s, with the same shape as the input
     *
     * @param input
     * @return
     */
    public SDVariable onesLike(SDVariable input) {
        return onesLike(null, input);
    }

    /**
     * Return a variable of all 1s, with the same shape as the input
     *
     * @param input
     * @return
     */
    public SDVariable onesLike(String name, SDVariable input) {
        SDVariable ret = f().onesLike(name, input);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Variable initialization
     * with 0.0
     *
     * @param name  the opName of the variable
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, long[] shape) {
        return var(name, shape, new ZeroInitScheme());
    }

    public SDVariable zero(String name, int[] shape) {
        return var(name, ArrayUtil.toLongArray(shape), new ZeroInitScheme());
    }

    /**
     * Return a variable of all 0s with the same shape as the input
     *
     * @param input
     * @return
     */
    public SDVariable zerosLike(SDVariable input) {
        return zerosLike(null, input);
    }

    /**
     * Return a variable of all 0s, with the same shape as the input
     *
     * @param input
     * @return
     */
    public SDVariable zerosLike(String name, SDVariable input) {
        SDVariable ret = f().zerosLike(name, input);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable constant(SDVariable value, long... shape) {
        return constant(null, value, shape);
    }

    public SDVariable constant(String name, SDVariable value, long... shape) {
        SDVariable ret = f().constant(value, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable linspace(double start, double stop, long number) {
        return linspace(null, start, stop, number);
    }

    public SDVariable linspace(String name, double start, double stop, long number) {
        SDVariable ret = f().linspace(start, stop, number);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable range(double from, double to, double step){
        return range(null, from, to, step);
    }

    public SDVariable range(String name, double from, double to, double step){
        SDVariable ret = f().range(from, to, step);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable[] meshgrid(SDVariable... inputs){
        return meshgrid(null, inputs);
    }

    public SDVariable[] meshgrid(List<String> names, SDVariable... inputs){
        return meshgrid(names, true, inputs);
    }

    public SDVariable[] meshgrid(List<String> names, boolean cartesian, SDVariable... inputs){
        Preconditions.checkState(names == null || names.size() == inputs.length,
                "Got %s names but %s inputs", (names == null ? 0 : names.size()), inputs.length);
        SDVariable[] ret = f().meshgrid(cartesian, inputs);
        for( int i=0; i<ret.length; i++ ){
            ret[i] = updateVariableNameAndReference(ret[i], names == null ? null : names.get(i));
        }
        return ret;
    }

    /**
     * Variable initialization
     * with a specified {@link WeightInitScheme}
     *
     * @param name             the opName of the variable
     * @param shape            the shape of the array to be created
     * @param weightInitScheme the weight init scheme
     * @return the created variable
     */
    public SDVariable var(String name, long[] shape, WeightInitScheme weightInitScheme) {
        if (variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name +
                    " already exists.");


        if (name == null || name.length() < 1)
            name = getNewVarName();

        if (workspace == null)
            initWorkspace();


        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(shape).weightInitScheme(weightInitScheme)
                .varName(name)
                .build();


        addVariable(ret);
        variableMap.put(name, ret);
        return ret;

    }


    /**
     * Creates a {@link SDVariable}
     * with the given shape
     * and a depth of 0.
     *
     * @param name  the opName of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, long... shape) {
        Preconditions.checkArgument(shape != null && shape.length > 0, "Invalid shape: %s", shape);
        return var(name, shape, new ZeroInitScheme());
    }

    public SDVariable var(String name, int... shape) {
        Preconditions.checkArgument(shape != null && shape.length > 0, "Invalid shape: %s", shape);
        return var(name, ArrayUtil.toLongArray(shape), new ZeroInitScheme());
    }


    /**
     * Initialize a {@link SDVariable}
     * reference tying this variable to this
     * samediff instance.
     * <p>
     * {@link NDArraySupplierInitScheme} is used
     * to ensure that if the array is allocated anywhere
     * and {@link SameDiff} instance to exist as a copy of the variable.
     *
     * @param arr
     * @return
     */
    public SDVariable var(final SDVariable arr) {
        if (variableMap.containsKey(arr.getVarName()) && variableMap.get(arr.getVarName()).getArr() != null)
            return variableMap.get(arr.getVarName());

        if (arr.getVarName() == null || arr.getVarName().length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if (arr == null)
            throw new IllegalArgumentException("Array for " + arr.getVarName() + " must not be null");

        if (workspace == null)
            initWorkspace();

        final SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(arr.getShape())
                .varName(arr.getVarName())
                .weightInitScheme(new NDArraySupplierInitScheme(new NDArraySupplierInitScheme.NDArraySupplier() {
                    @Override
                    public INDArray getArr() {
                        /**
                         * Pre allocate the array if it doesn't already exist.
                         * The reason we do this is to avoid race conditions with
                         * {@link #allocate()}
                         */
                        if (arr.getArr() == null) {
                            INDArray retArr = arr.getWeightInitScheme().create(arr.getShape());
                            associateArrayWithVariable(retArr, arr);
                        }
                        return arr.getArr();
                    }
                }))
                .build();


        variableMap.put(arr.getVarName(), ret);
        return ret;

    }

    // auto naming

    private int _var_id = 0;

    private String getNewVarName() {
        String varName = "sd_var_" + String.valueOf(_var_id);
        while (variableMap.containsKey(varName)) {
            _var_id++;
            varName = "sd_var_" + String.valueOf(_var_id);
        }
        return varName;
    }

    public SDVariable var(int... shape) {
        return var(getNewVarName(), shape);
    }

    public SDVariable var(long... shape) {
        return var(getNewVarName(), shape);
    }

    public SDVariable var(WeightInitScheme weightInitScheme, long... shape) {
        return var(getNewVarName(), shape, weightInitScheme);
    }

    public SDVariable var(INDArray arr) {
        return var(getNewVarName(), arr);
    }


    /**
     * Generate a square identity matrix with the specified number of rows
     *
     * @param rows Number of rows
     */
    public SDVariable eye(int rows) {
        return eye(rows, rows);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns
     *
     * @param rows Number of rows
     */
    public SDVariable eye(String name, int rows) {
        return eye(name, rows, rows);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    public SDVariable eye(int rows, int cols) {
        return eye(null, rows, cols);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    public SDVariable eye(String name, int rows, int cols) {
        return eye(name, rows, cols, null);
    }

    /**
     * see {@link #eye(String, int, int, int...)}
     */
    public SDVariable eye(int rows, int cols, int... batchDimension) {
        return eye(null, rows, cols, batchDimension);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns, with optional leading dims<br>
     * Example:<br>
     * batchShape: [3,3]<br>
     * numRows: 2<br>
     * numCols: 4<br>
     * returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:<br>
     * 1 0 0 0<br>
     * 0 1 0 0<br>
     *
     * @param rows           Number of rows
     * @param cols           Number of columns
     * @param batchDimension Batch dimensions. May be null
     */
    public SDVariable eye(String name, int rows, int cols, int... batchDimension) {
        SDVariable eye = new Eye(this, rows, cols, batchDimension).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    public SDVariable eye(String name, SDVariable rows, SDVariable cols, SDVariable batchDimension){
        SDVariable eye = new Eye(this, rows, cols, batchDimension).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    public SDVariable eye(SDVariable rows, SDVariable cols, SDVariable batchDimension){
        return eye(null, rows, cols, batchDimension);
    }


    public SDVariable eye(String name, SDVariable rows, SDVariable cols){
        SDVariable eye = new Eye(this, rows, cols).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    public SDVariable eye(SDVariable rows, SDVariable cols){
        SDVariable eye = new Eye(this, rows, cols).outputVariables()[0];
        return updateVariableNameAndReference(eye, null);
    }

    public SDVariable eye(String name, SDVariable rows){
        SDVariable eye = new Eye(this, rows).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    public SDVariable eye(SDVariable rows){
        SDVariable eye = new Eye(this, rows).outputVariables()[0];
        return updateVariableNameAndReference(eye, null);
    }

    /**
     * Remove an argument for a function. Note that if this function
     * does not contain the argument, it will just be a no op.
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
                val reverseArgs = incomingArgsReverse.get(function.getOwnName());
                incomingArgsReverse.remove(function.getOwnName());
                val newArgs = new ArrayList<String>(args.length - 1);
                for (int arg = 0; arg < args.length; arg++) {
                    if (!reverseArgs[arg].equals(varName)) {
                        newArgs.add(reverseArgs[arg]);
                    }
                }

                val newArgsArr = newArgs.toArray(new String[newArgs.size()]);
                incomingArgsReverse.put(function.getOwnName(), newArgsArr);
                //no further need to scan
                break;
            }
        }
    }


    /**
     * @param name
     * @param arr
     * @return
     */
    public SDVariable var(String name, INDArray arr) {
        if (variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name +
                    " already exists.");


        if (name == null || name.length() < 1)
            name = getNewVarName();

        if (arr == null)
            throw new IllegalArgumentException("Array for " + name + " must not be null");

        if (workspace == null)
            initWorkspace();

        val arrRef = arr.migrate();
        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(arr.shape())
                .varName(name)
                .weightInitScheme(new NDArraySupplierInitScheme(new NDArraySupplierInitScheme.NDArraySupplier() {
                    @Override
                    public INDArray getArr() {
                        return arrRef;
                    }
                }))
                .build();


        associateArrayWithVariable(arr, ret);
        if (ArrayUtil.prod(arr.shape()) == 1)
            ret.setScalarValue(arr.getDouble(0));

        addVariable(ret);
        if (getShapeForVarName(name) == null)
            putShapeForVarName(name, arr.shape());
        //ensure there is a reference to the array in the integer index
        //this is used later for op creation
        reverseArrayLookup.put(arr, ret);
        variableMap.put(name, ret);
        return ret;

    }

    /**
     * Get the variable based on the opName
     *
     * @param name the opName of the variable
     * @return the variabel instance if there is one
     */
    public SDVariable getVariable(String name) {
        return variableMap.get(name);
    }


    /**
     * Get the gradient for the given vertex id
     *
     * @param varName the vertex id
     * @return the gradient for this variable or null
     */
    public SDVariable getGradForVariable(String varName) {
        return gradients.get(varName);
    }


    /**
     * Assign a vertex id
     * to a gradient
     *
     * @param variableName the vertex id
     *                     to assign
     * @param variable     the variable
     */
    public void setGradientForVariableName(String variableName, SDVariable variable) {
        if (variable == null) {
            throw new ND4JIllegalStateException("Unable to set null gradient for variable name " + variableName);
        }

        gradients.put(variableName, variable);
    }


    /**
     * Get the forward variable for gradient
     * based on the gradient's vertex id
     *
     * @param vertexId the vertex id
     * @return the gradient for the variable or null
     */
    public SDVariable getForwardVariableForVertexId(int vertexId) {
        return forwardVarForGrad.get(vertexId);
    }


    /**
     * @param varName
     * @param forwardVariable
     */
    public void setForwardVariableForVarName(String varName, SDVariable forwardVariable) {
        forwardVarForGrad.put(varName, forwardVariable);
    }

    /**
     * Gradient with respect
     * to the given variable opName.
     * Note that in order to run this function,
     * {@link #execBackwards()} must be executed first.
     * All gradient functions are obtained within that time.
     *
     * @param varName the variable opName to get the gradient for.
     * @return
     */
    public SDVariable grad(String varName) {
        if (!sameDiffFunctionInstances.containsKey("grad")) {
            throw new IllegalStateException("Unable to obtain gradient. Please run execBackwards() first.");
        }

        SameDiff grad = getFunction("grad");
        SDVariable var = grad.getVariable(varName);
        return getFunction("grad").getGradForVariable(var.getVarName());
    }

    public SDVariable randomUniform(double min, double max, SDVariable shape){
        return randomUniform(null, min, max, shape);
    }

    public SDVariable randomUniform(String name, double min, double max, SDVariable shape){
        SDVariable ret = f().randomUniform(min, max, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomUniform(double min, double max, long... shape){
        return randomUniform(null, min, max, shape);
    }

    public SDVariable randomUniform(String name, double min, double max, long... shape){
        SDVariable ret = f().randomUniform(min, max, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomNormal(double mean, double stddev, SDVariable shape){
        return randomNormal(null, mean, stddev, shape);
    }

    public SDVariable randomNormal(String name, double mean, double stddev, SDVariable shape){
        SDVariable ret = f().randomNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomNormal(double mean, double stddev, long... shape){
        return randomNormal(null, mean, stddev, shape);
    }

    public SDVariable randomNormal(String name, double mean, double stddev, long... shape){
        SDVariable ret = f().randomNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomLogNormal(double mean, double stddev, long... shape){
        return randomLogNormal(null, mean, stddev, shape);
    }

    public SDVariable randomLogNormal(String name, double mean, double stddev, long... shape){
        SDVariable ret = f().randomLogNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomNormalTruncated(double mean, double stddev, long... shape){
        return randomNormalTruncated(null, mean, stddev, shape);
    }

    public SDVariable randomNormalTruncated(String name, double mean, double stddev, long... shape){
        SDVariable ret = f().randomNormalTruncated(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomBernoulli(double p, SDVariable shape){
        return randomBernoulli(null, p, shape);
    }

    public SDVariable randomBernoulli(String name, double p, SDVariable shape){
        SDVariable ret = f().randomBernoulli(p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomBernoulli(double p, long... shape){
        return randomBernoulli(null, p, shape);
    }

    public SDVariable randomBernoulli(String name, double p, long... shape){
        SDVariable ret = f().randomBernoulli(p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomBinomial(int nTrials, double p, long... shape){
        return randomBinomial(null, nTrials, p, shape);
    }

    public SDVariable randomBinomial(String name, int nTrials, double p, long... shape){
        SDVariable ret = f().randomBinomial(nTrials, p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Exponential distribution: P(x) = lambda * exp(-lambda * x)
     *
     * @param lambda Must be > 0
     * @param shape  Shape of the output
     */
    public SDVariable randomExponential(double lambda, SDVariable shape) {
        return randomExponential(null, lambda, shape);
    }

    /**
     * Exponential distribution: P(x) = lambda * exp(-lambda * x)
     *
     * @param name   Name of the output variable
     * @param lambda Must be > 0
     * @param shape  Shape of the output
     */
    public SDVariable randomExponential(String name, double lambda, SDVariable shape) {
        SDVariable ret = f().randomExponential(lambda, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Upsampling 2d - same scale for both dimensions. NCHW input format.
     *
     * @param input Input, in NCHW format
     * @param scale Scale to upsample in both H and W dimensions
     * @return Upsampled input
     */
    public SDVariable upsampling2d(SDVariable input, int scale) {
        return upsampling2d(null, input, true, scale, scale);
    }

    /**
     * Upsampling 2d - same scale for both dimensions. NCHW input format.
     *
     * @param input Input, in NCHW format
     * @param scale Scale to upsample in both H and W dimensions
     * @return Upsampled input
     */
    public SDVariable upsampling2d(String name, SDVariable input, int scale) {
        return upsampling2d(name, input, true, scale, scale);
    }

    /**
     * Upsampling 2d
     *
     * @param input  Input, in NCHW format
     * @param nchw   If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
     * @param scaleH Scale to upsample in height dimension
     * @param scaleW Scale to upsample in width dimension
     * @return Upsampled input
     */
    public SDVariable upsampling2d(SDVariable input, boolean nchw, int scaleH, int scaleW) {
        return upsampling2d(null, input, nchw, scaleH, scaleW);
    }

    /**
     * Upsampling 2d
     *
     * @param input  Input, in NCHW format
     * @param nchw   If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
     * @param scaleH Scale to upsample in height dimension
     * @param scaleW Scale to upsample in width dimension
     * @return Upsampled input
     */
    public SDVariable upsampling2d(String name, SDVariable input, boolean nchw, int scaleH, int scaleW) {
        SDVariable ret = f().upsampling2d(input, nchw, scaleH, scaleW);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Average pooling 2d operation.
     *
     * @param input           the input to average pooling 2d
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable avgPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return avgPooling2d(null, input, pooling2DConfig);
    }

    /**
     * Average pooling 2d operation.
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to average pooling 2d
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable avgPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        SDVariable ret = f().avgPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Max pooling 2d operation.
     *
     * @param input           the input to max pooling 2d
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable maxPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return maxPooling2d(null, input, pooling2DConfig);
    }

    /**
     * Max pooling 2d operation.
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to max pooling 2d
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable maxPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        SDVariable ret = f().maxPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Average pooling 3d operation.
     *
     * @param input           the input to average pooling 3d
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return avgPooling3d(null, input, pooling3DConfig);
    }

    /**
     * Average pooling 3d operation.
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to average pooling 3d
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable avgPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        SDVariable ret = f().avgPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Max pooling 3d operation.
     *
     * @param input           the input to max pooling 3d
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return maxPooling3d(null, input, pooling3DConfig);
    }

    /**
     * Max pooling 3d operation.
     *
     * @param name            name of the operation in SameDiff
     * @param input           the inputs to max pooling 3d
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable maxPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        SDVariable ret = f().maxPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Conv1d operation.
     *
     * @param input        the input array to conv1d op
     * @param weights      weights for conv1d op
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        return conv1d(null, input, weights, conv1DConfig);
    }

    /**
     * Conv1d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param input        the inputs to conv1d
     * @param weights      weights for conv1d op
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(String name, SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        SDVariable ret = f().conv1d(input, weights, conv1DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Local response normalization operation.
     *
     * @param inputs    the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(SDVariable inputs, LocalResponseNormalizationConfig lrnConfig) {
        return localResponseNormalization(null, inputs, lrnConfig);
    }

    /**
     * Local response normalization operation.
     *
     * @param name      name of the operation in SameDiff
     * @param input    the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(String name, SDVariable input,
                                                 LocalResponseNormalizationConfig lrnConfig) {
        SDVariable ret = f().localResponseNormalization(input, lrnConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param weights conv 2D weights
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, Conv2DConfig config) {
        return conv2d(layerInput, weights, null, config);
    }


    /**
     * 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param weights conv 2D weights
     * @param bias conv 2D bias
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return conv2d(arr, config);
    }

    /**
     * Conv2d operation.
     *
     * @param inputs       the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable conv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        return conv2d(null, inputs, conv2DConfig);
    }

    /**
     * Conv2d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param inputs       the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable conv2d(String name, SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        SDVariable ret = f().conv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Depth-wise 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param depthWeights depth-wise conv 2D weights
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, Conv2DConfig config) {
        return depthWiseConv2d(layerInput, depthWeights, null, config);
    }


    /**
     * Depth-wise 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param depthWeights depth-wise conv 2D weights
     * @param bias conv 2D bias
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        if (bias != null)
            arr[2] = bias;
        return depthWiseConv2d(arr, config);
    }


    /**
     * Depth-wise Conv2d operation.
     *
     * @param inputs            the inputs to depth-wise conv2d
     * @param depthConv2DConfig the configuration
     * @return
     */
    public SDVariable depthWiseConv2d(SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        return depthWiseConv2d(null, inputs, depthConv2DConfig);
    }


    /**
     * Depth-wise Conv2d operation.
     *
     * @param name              name of the operation in SameDiff
     * @param inputs            the inputs to sconv2d
     * @param depthConv2DConfig the configuration
     * @return
     */
    public SDVariable depthWiseConv2d(String name, SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        SDVariable ret = f().depthWiseConv2d(inputs, depthConv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Separable 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param depthWeights conv 2D weights
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      Conv2DConfig config) {
        return separableConv2d(layerInput, depthWeights, pointWeights, null, config);
    }


    /**
     * Separable 2D convolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param depthWeights depth-wise conv 2D weights
     * @param pointWeights point-wise conv 2D weights
     * @param bias conv 2D bias
     * @param config Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 3 : 4];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        arr[2] = pointWeights;
        if (bias != null)
            arr[3] = bias;
        return sconv2d(arr, config);
    }

    /**
     * Separable Conv2d operation.
     *
     * @param inputs       the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable sconv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        return sconv2d(null, inputs, conv2DConfig);
    }


    /**
     * Separable Conv2d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param inputs       the inputs to sconv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable sconv2d(String name, SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        SDVariable ret = f().sconv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * 2D deconvolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param weights conv 2D weights
     * @param deconv2DConfig DeConv2DConfig configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, DeConv2DConfig deconv2DConfig) {
        return deconv2d(layerInput, weights, null, deconv2DConfig);
    }


    /**
     * 2D deconvolution operation
     *
     * @param layerInput input tensor to conv 2D op
     * @param weights conv 2D weights
     * @param bias conv 2D bias
     * @param deconv2DConfig DeConv2DConfig configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, DeConv2DConfig deconv2DConfig) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return deconv2d(arr, deconv2DConfig);
    }

    /**
     * Deconv2d operation.
     *
     * @param inputs         the inputs to sconv2d
     * @param deconv2DConfig the configuration
     * @return
     */
    public SDVariable deconv2d(SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        return deconv2d(null, inputs, deconv2DConfig);
    }


    /**
     * Deconv2d operation.
     *
     * @param name           name of the operation in SameDiff
     * @param inputs         the inputs to sconv2d
     * @param deconv2DConfig the configuration
     * @return
     */
    public SDVariable deconv2d(String name, SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        SDVariable ret = f().deconv2d(inputs, deconv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Conv3d operation.
     *
     * @param input        the input activations to conv3d
     * @param weights      Weights for conv3d
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, null, conv3DConfig);
    }

    /**
     * Conv3d operation.
     *
     * @param input        the input activations to conv3d
     * @param weights      Weights for conv3d
     * @param bias         bias for the Conv3d op. May be null if not present/used
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, bias, conv3DConfig);
    }

    /**
     * Conv3d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param input        the input activations to conv3d
     * @param weights      Weights for conv3d
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, null, conv3DConfig);
    }

    /**
     * Conv3d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param input        the input activations to conv3d
     * @param weights      Weights for conv3d
     * @param bias         bias for the Conv3d op. May be null if not present/used
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        SDVariable[] args;
        if (bias == null) {
            args = new SDVariable[]{input, weights};
        } else {
            args = new SDVariable[]{input, weights, bias};
        }
        SDVariable ret = f().conv3d(args, conv3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Batch norm operation.
     */
    public SDVariable batchNorm(SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta,
                                boolean applyGamma, boolean applyBeta, double epsilon) {
        return batchNorm(null, input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon);
    }

    /**
     * Batch norm operation.
     */
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta,
                                boolean applyGamma, boolean applyBeta, double epsilon) {
        SDVariable res = f().batchNorm(input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable im2Col(SDVariable in, Conv2DConfig config) {
        return im2Col(null, in, config);
    }

    public SDVariable im2Col(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().im2Col(in, config);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable col2Im(SDVariable in, Conv2DConfig config) {
        return col2Im(null, in, config);
    }

    public SDVariable col2Im(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().col2Im(in, config);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param name
     * @param value
     * @return
     */
    public SDVariable scalar(String name, double value) {
        return var(name, Nd4j.scalar(value));
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gte(SDVariable iX, double iy) {
        return gte(null, iX, iy);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, double iy) {
        return lte(null, iX, iy);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, double iy) {
        return gt(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, double iy) {
        return lt(null, iX, iy);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, double iy) {
        return neq(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, double iy) {
        return eq(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable gte(SDVariable iX, SDVariable iy) {
        return gte(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, SDVariable iy) {
        return lte(null, iX, iy);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, SDVariable iy) {
        return gt(null, iX, iy);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, SDVariable iy) {
        return lt(null, iX, iy);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        return neq(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, SDVariable iy) {
        return eq(null, iX, iy);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable or(SDVariable iX, SDVariable iy) {
        return or(null, iX, iy);
    }

    public SDVariable and(SDVariable iX, SDVariable iY) {
        return and(null, iX, iY);
    }

    public SDVariable and(String name, SDVariable ix, SDVariable iy) {
        SDVariable result = f().and(ix, iy);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable xor(SDVariable ix, SDVariable iy) {
        return xor(null, ix, iy);
    }

    public SDVariable xor(String name, SDVariable ix, SDVariable iy) {
        SDVariable result = f().xor(ix, iy);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable abs(SDVariable ix) {
        return abs(null, ix);
    }

    public SDVariable abs(String name, SDVariable ix) {
        SDVariable result = f().abs(ix);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable neg(SDVariable iX) {
        return neg(null, iX);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable cos(SDVariable iX) {
        return cos(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sin(SDVariable iX) {
        return sin(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable tan(SDVariable iX) {
        return tan(null, iX);
    }

    public SDVariable identity(SDVariable input) {
        return identity(null, input);
    }

    public SDVariable identity(String name, SDVariable input) {
        SDVariable s = f().identity(input);
        return updateVariableNameAndReference(s, name);
    }

    public SDVariable invertPermutation(SDVariable input) {
        return invertPermutation(null, input);
    }

    public SDVariable invertPermutation(String name, SDVariable input) {
        SDVariable ret = f().invertPermutation(input, false);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable acos(SDVariable iX) {
        return acos(null, iX);
    }

    /**
     * @param iX
     * @return
     */

    public SDVariable asin(SDVariable iX) {
        return asin(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable atan(SDVariable iX) {
        return atan(null, iX);
    }

    public SDVariable atan2(SDVariable y, SDVariable x) {
        return atan2(null, y, x);
    }

    public SDVariable atan2(String name, SDVariable y, SDVariable x) {
        SDVariable ret = f().atan2(y, x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable cosh(SDVariable iX) {
        return cosh(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sinh(SDVariable iX) {
        return sinh(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable tanh(SDVariable iX) {
        return tanh(null, iX);
    }

    public SDVariable step(SDVariable in, double cutoff) {
        return step(null, in, cutoff);
    }

    public SDVariable step(String name, SDVariable in, double cutoff) {
        SDVariable ret = f().step(in, cutoff);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable acosh(SDVariable iX) {
        return acosh(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable asinh(SDVariable iX) {
        return asinh(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable atanh(SDVariable iX) {
        return atanh(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable exp(SDVariable iX) {
        return exp(null, iX);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable rsqrt(SDVariable iX) {
        return rsqrt(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable expm1(SDVariable iX) {
        return expm1(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable log1p(SDVariable iX) {
        return log1p(null, iX);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable isInfinite(SDVariable iX) {
        return isInfinite(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isNaN(SDVariable iX) {
        return isNaN(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable round(SDVariable iX) {
        return round(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isFinite(SDVariable iX) {
        return isFinite(null, iX);
    }

    public SDVariable isMax(SDVariable ix) {
        return isMax(null, ix);
    }

    public SDVariable isMax(String name, SDVariable ix) {
        SDVariable ret = f().isMax(ix);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable replaceWhere(SDVariable update, SDVariable from, Condition condition) {
        return replaceWhere(null, update, from, condition);
    }

    public SDVariable replaceWhere(String name, SDVariable update, SDVariable from, Condition condition) {
        SDVariable ret = f().replaceWhere(update, from, condition);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable replaceWhere(SDVariable update, Number value, Condition condition) {
        return replaceWhere(null, update, value, condition);
    }

    public SDVariable replaceWhere(String name, SDVariable to, Number value, Condition condition) {
        SDVariable ret = f().replaceWhere(to, value, condition);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable log(SDVariable iX) {
        return log(null, iX);
    }

    public SDVariable log(SDVariable in, double base) {
        return log(null, in, base);
    }

    public SDVariable log(String name, SDVariable in, double base) {
        SDVariable ret = f().log(in, base);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable logSumExp(SDVariable input, int... dimensions) {
        return logSumExp(null, input, dimensions);
    }

    public SDVariable logSumExp(String name, SDVariable input, int... dimensions) {
        SDVariable ret = f().logSumExp(input, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable cube(SDVariable iX) {
        return cube(null, iX);
    }


    /**
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(SDVariable iX, double value) {
        return pow(null, iX, value);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sqrt(SDVariable iX) {
        return sqrt(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable square(SDVariable iX) {
        return square(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable floor(SDVariable iX) {
        return floor(null, iX);
    }

    public SDVariable ceil(SDVariable x) {
        return ceil(null, x);
    }

    public SDVariable ceil(String name, SDVariable x) {
        SDVariable ret = f().ceil(x);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
        return clipByValue(null, x, clipValueMin, clipValueMax);
    }

    public SDVariable clipByValue(String name, SDVariable x, double clipValueMin, double clipValueMax) {
        SDVariable ret = f().clipByValue(x, clipValueMin, clipValueMax);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue) {
        return clipByNorm(null, x, clipValue);
    }

    public SDVariable clipByNorm(String name, SDVariable x, double clipValue) {
        SDVariable ret = f().clipByNorm(x, clipValue);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue, int... dimensions) {
        return clipByNorm(null, x, clipValue, dimensions);
    }

    public SDVariable clipByNorm(String name, SDVariable x, double clipValue, int... dimensions) {
        SDVariable ret = f().clipByNorm(x, clipValue, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable relu(SDVariable iX, double cutoff) {
        return relu(null, iX, cutoff);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable relu6(SDVariable iX, double cutoff) {
        return relu6(null, iX, cutoff);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softmax(SDVariable iX) {
        return softmax(null, iX);
    }

    public SDVariable logSoftmax(SDVariable iX) {
        return logSoftmax(null, iX);
    }

    public SDVariable logSoftmax(String name, SDVariable iX) {
        SDVariable ret = f().logSoftmax(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable selu(SDVariable iX) {
        return selu(null, iX);
    }

    public SDVariable selu(String name, SDVariable iX) {
        SDVariable ret = f().selu(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeAdd(SDVariable... iX) {
        return mergeAdd(null, iX);
    }

    public SDVariable mergeAdd(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeAdd(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeMax(SDVariable... iX) {
        return mergeMax(null, iX);
    }

    public SDVariable mergeMax(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeMax(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeAvg(SDVariable... inputs) {
        return mergeAvg(null, inputs);
    }

    public SDVariable mergeAvg(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeAvg(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable batchToSpace(SDVariable iX, int[] blocks, int[][] crops) {
        return batchToSpace(null, iX, blocks, crops);
    }

    public SDVariable batchToSpace(String name, SDVariable iX, int[] blocks, int[][] crops) {
        SDVariable ret = f().batchToSpace(iX, blocks, crops);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable depthToSpace(SDVariable iX, int blockSize, String dataFormat) {
        return depthToSpace(null, iX, blockSize, dataFormat);
    }

    public SDVariable depthToSpace(String name, SDVariable iX, int blockSize, String dataFormat) {
        SDVariable ret = f().depthToSpace(iX, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable spaceToBatch(SDVariable iX, int[] blocks, int[][] padding) {
        return spaceToBatch(null, iX, blocks, padding);
    }

    public SDVariable spaceToBatch(String name, SDVariable iX, int[] blocks, int[][] padding) {
        SDVariable ret = f().spaceToBatch(iX, blocks, padding);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable spaceToDepth(SDVariable iX, int blockSize, String dataFormat) {
        return spaceToDepth(null, iX, blockSize, dataFormat);
    }

    public SDVariable spaceToDepth(String name, SDVariable iX, int blockSize, String dataFormat) {
        SDVariable ret = f().spaceToDepth(iX, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable[] dynamicPartition(SDVariable iX, SDVariable partitions, int numPartitions) {
        return dynamicPartition(null, iX, partitions, numPartitions);
    }

    public SDVariable[] dynamicPartition(String[] name, SDVariable iX, SDVariable partitions, int numPartitions) {
        SDVariable[] ret = f().dynamicPartition(iX, partitions, numPartitions);
        return updateVariableNamesAndReferences(ret, name);
    }

    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] iX) {
        return dynamicStitch(null, indices, iX);
    }

    public SDVariable dynamicStitch(String name, SDVariable[] indices, SDVariable[] iX) {
        SDVariable ret = f().dynamicStitch(indices, iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        return dilation2D(null, df, weights, strides, rates, isSameMode);
    }

    public SDVariable dilation2D(String name, SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        SDVariable ret = f().dilation2D(df, weights, strides, rates, isSameMode);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable shape(SDVariable df) {
        return shape(null, df);
    }

    public SDVariable shape(String name, SDVariable df) {
        SDVariable ret = f().shape(df);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable size(SDVariable in){
        return size(null, in);
    }

    public SDVariable size(String name, SDVariable in){
        SDVariable ret = f().size(in);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable rank(SDVariable in) {
        return rank(null, in);
    }

    public SDVariable rank(String name, SDVariable in) {
        SDVariable ret = f().rank(in);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable cross(SDVariable a, SDVariable b) {
        return cross(null, a, b);
    }

    public SDVariable cross(String name, SDVariable a, SDVariable b) {
        SDVariable ret = f().cross(a, b);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gather(SDVariable df, int[] indices, int axis) {
        return gather(null, df, indices, axis);
    }

    public SDVariable gather(String name, SDVariable df, int[] indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gather(SDVariable df, SDVariable indices, int axis) {
        return gather(null, df, indices, axis);
    }

    public SDVariable gather(String name, SDVariable df, SDVariable indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        return gatherNd(null, df, indices);
    }

    public SDVariable gatherNd(String name, SDVariable df, SDVariable indices) {
        SDVariable ret = f().gatherNd(df, indices);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable repeat(SDVariable df, int axis) {
        return repeat(null, df, axis);
    }


    public SDVariable repeat(String name, SDVariable df, int axis) {
        SDVariable ret = f().repeat(df, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stack(int axis, SDVariable... values) {
        return stack(null, axis, values);
    }

    public SDVariable stack(String name, int axis, SDVariable... values) {
        SDVariable ret = f().stack(values, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable parallel_stack(SDVariable[] values) {
        return parallel_stack(null, values);
    }

    public SDVariable parallel_stack(String name, SDVariable[] values) {
        SDVariable ret = f().parallel_stack(values);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable[] unstack(SDVariable value, int axis) {
        return unstack(null, value, axis);
    }

    public SDVariable[] unstack(String[] names, SDVariable value, int axis) {
        SDVariable[] ret = f().unstack(value, axis);
        return updateVariableNamesAndReferences(ret, names);
    }

    public SDVariable[] unstack(SDVariable value, int axis, int num) {
        return unstack(null, value, axis, num);
    }

    public SDVariable[] unstack(String[] names, SDVariable value, int axis, int num) {
        SDVariable[] ret = f().unstack(value, axis, num);
        return updateVariableNamesAndReferences(ret, names);
    }

    public SDVariable erf(SDVariable iX) {
        return erf(null, iX);
    }

    public SDVariable erf(String name, SDVariable iX) {
        SDVariable ret = f().erf(iX);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable erfc(SDVariable iX) {
        return erfc(null, iX);
    }

    public SDVariable erfc(String name, SDVariable iX) {
        SDVariable ret = f().erfc(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable diag(SDVariable iX) {
        return diag(null, iX);
    }

    public SDVariable diag(String name, SDVariable iX) {
        SDVariable ret = f().diag(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable diagPart(SDVariable iX) {
        return diagPart(null, iX);
    }

    public SDVariable diagPart(String name, SDVariable iX) {
        SDVariable ret = f().diagPart(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable setDiag(SDVariable in, SDVariable diag) {
        return setDiag(null, in, diag);
    }

    public SDVariable setDiag(String name, SDVariable in, SDVariable diag) {
        SDVariable ret = f().setDiag(in, diag);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable oneHot(SDVariable indices, int depth) {
        return oneHot(null, indices, depth, -1, 1.00, 0.00);
    }

    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off) {
        return oneHot(null, indices, depth, axis, on, off);
    }

    public SDVariable oneHot(String name, SDVariable indices, int depth) {
        return oneHot(name, indices, depth, -1, 1.00, 0.00);
    }

    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on, double off) {
        SDVariable ret = f().onehot(indices, depth, axis, on, off);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reciprocal(SDVariable a) {
        return reciprocal(null, a);
    }

    public SDVariable reciprocal(String name, SDVariable a) {
        SDVariable ret = f().reciprocal(a);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return gradientBackwardsMarker(generateNewVarName(new GradientBackwardsMarker().opName(), 0), iX);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable hardTanh(SDVariable iX) {
        return hardTanh(null, iX);
    }

    public SDVariable hardSigmoid(SDVariable in) {
        return hardSigmoid(null, in);
    }

    public SDVariable hardSigmoid(String name, SDVariable in) {
        SDVariable ret = f().hardSigmoid(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(SDVariable iX) {
        return hardTanhDerivative(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sigmoid(SDVariable iX) {
        return sigmoid(null, iX);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return sigmoidDerivative(null, iX, wrt);
    }

    public SDVariable logSigmoid(SDVariable iX) {
        return logSigmoid(null, iX);
    }

    public SDVariable logSigmoid(String name, SDVariable iX) {
        SDVariable ret = f().logSigmoid(iX);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sign(SDVariable iX) {
        return sign(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softsign(SDVariable iX) {
        return softsign(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(SDVariable iX) {
        return softsignDerivative(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softplus(SDVariable iX) {
        return softplus(null, iX);
    }

    public SDVariable swish(SDVariable iX) {
        return swish(null, iX);
    }

    public SDVariable swish(String name, SDVariable iX) {
        SDVariable ret = f().swish(iX);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable elu(SDVariable iX) {
        return elu(null, iX);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(SDVariable iX) {
        return eluDerivative(null, iX);
    }

    /**
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return leakyRelu(null, iX, cutoff);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable mean(SDVariable iX) {
        return mean(null, iX);
    }


    /**
     * @param iX
     * @param dimension
     * @return
     */
    public SDVariable mean(SDVariable iX, int... dimension) {
        return mean(null, iX, dimension);
    }

    /**
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable standardDeviation(SDVariable iX,
                                        boolean biasCorrected,
                                        int... dimensions) {
        return standardDeviation(null, iX, biasCorrected, dimensions);
    }

    /**
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable variance(SDVariable iX,
                               boolean biasCorrected,
                               int... dimensions) {
        return variance(null, iX, biasCorrected, dimensions);
    }

    /**
     * Entropy reduction: -sum(x * log(x))
     *
     * @param in         Input
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable
     */
    public SDVariable entropy(SDVariable in, int... dimensions) {
        return entropy(null, in, dimensions);
    }

    /**
     * Entropy reduction: -sum(x * log(x))
     *
     * @param in         Input
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable
     */
    public SDVariable entropy(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().entropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Log entropy reduction: log(-sum(x * log(x)))
     *
     * @param in         Input
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable
     */
    public SDVariable logEntropy(SDVariable in, int... dimensions) {
        return logEntropy(null, in, dimensions);
    }

    /**
     * Log entropy reduction: log(-sum(x * log(x)))
     *
     * @param in         Input
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable
     */
    public SDVariable logEntropy(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().logEntropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(SDVariable iX, int... dimensions) {
        return sum(null, iX, dimensions);
    }

    public SDVariable sum(SDVariable iX, boolean keepDims, int... dimensions) {
        return sum(null, iX, keepDims, dimensions);
    }



    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(SDVariable iX, int... dimensions) {
        return prod(null, iX, dimensions);
    }


    public SDVariable scalarMax(SDVariable in, Number value) {
        return scalarMax(null, in, value);
    }

    public SDVariable scalarMax(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarMax(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarMin(SDVariable in, Number value) {
        return scalarMin(null, in, value);
    }

    public SDVariable scalarMin(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarMin(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarFloorMod(SDVariable in, Number value) {
        return scalarFloorMod(null, in, value);
    }

    public SDVariable scalarFloorMod(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarFloorMod(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarSet(SDVariable in, Number set) {
        return scalarSet(null, in, set);
    }

    public SDVariable scalarSet(String name, SDVariable in, Number set) {
        SDVariable ret = f().scalarSet(in, set);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(SDVariable iX, int... dimensions) {
        return max(null, iX, dimensions);
    }

    public SDVariable max(SDVariable first, SDVariable second) {
        return max(null, first, second);
    }

    public SDVariable max(String name, SDVariable first, SDVariable second) {
        SDVariable result = f().max(first, second);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable amax(SDVariable in, int... dimensions) {
        return amax(null, in, dimensions);
    }

    public SDVariable amax(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amax(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable amin(SDVariable in, int... dimensions) {
        return amin(null, in, dimensions);
    }

    public SDVariable amin(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amin(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable amean(SDVariable in, int... dimensions) {
        return amean(null, in, dimensions);
    }

    public SDVariable amean(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amean(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable asum(SDVariable in, int... dimensions) {
        return asum(null, in, dimensions);
    }

    public SDVariable asum(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().asum(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable countZero(SDVariable input, int... dimensions) {
        return countZero(null, input, dimensions);
    }

    public SDVariable countZero(String name, SDVariable input, int... dimensions) {
        SDVariable res = f().countZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable zeroFraction(SDVariable input) {
        return zeroFraction(null, input);
    }

    public SDVariable zeroFraction(String name, SDVariable input) {
        SDVariable res = f().zeroFraction(input);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable countNonZero(SDVariable input, int... dimensions) {
        return countNonZero(null, input, dimensions);
    }

    public SDVariable countNonZero(String name, SDVariable input, int... dimensions) {
        SDVariable res = f().countNonZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(SDVariable iX, int... dimensions) {
        return min(null, iX, dimensions);
    }

    public SDVariable min(SDVariable first, SDVariable second) {
        return min(null, first, second);
    }

    public SDVariable min(String name, SDVariable first, SDVariable second) {
        SDVariable result = f().min(first, second);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable argmax(SDVariable in, int... dimensions) {
        return argmax(null, in, false, dimensions);
    }

    public SDVariable argmax(SDVariable in, boolean keepDims, int... dimensions) {
        return argmax(null, in, dimensions);
    }

    public SDVariable argmax(String name, SDVariable in, int... dimensions) {
        return argmax(name, in, false, dimensions);
    }

    public SDVariable argmax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().argmax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable argmin(SDVariable in, int... dimensions) {
        return argmin(null, in, dimensions);
    }

    public SDVariable argmin(SDVariable in, boolean keepDims, int... dimensions) {
        return argmin(null, in, keepDims, dimensions);
    }

    public SDVariable argmin(String name, SDVariable in, int... dimensions) {
        return argmin(name, in, false, dimensions);
    }

    public SDVariable argmin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().argmin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable iamax(SDVariable in, int... dimensions) {
        return iamax(null, in, dimensions);
    }

    public SDVariable iamax(SDVariable in, boolean keepDims, int... dimensions) {
        return iamax(null, in, keepDims, dimensions);
    }

    public SDVariable iamax(String name, SDVariable in, int... dimensions) {
        return iamax(name, in, false, dimensions);
    }

    public SDVariable iamax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().iamax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable iamin(SDVariable in, int... dimensions) {
        return iamin(null, in, dimensions);
    }

    public SDVariable iamin(SDVariable in, boolean keepDims, int... dimensions) {
        return iamin(null, in, keepDims, dimensions);
    }

    public SDVariable iamin(String name, SDVariable in, int... dimensions) {
        return iamin(name, in, false, dimensions);
    }

    public SDVariable iamin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().iamin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable firstIndex(SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(null, in, condition, dimensions);
    }

    public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return firstIndex(null, in, condition, keepDims, dimensions);
    }

    public SDVariable firstIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(name, in, condition, false, dimensions);
    }

    public SDVariable firstIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        SDVariable ret = f().firstIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(null, in, condition, dimensions);
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return lastIndex(null, in, condition, keepDims, dimensions);
    }

    public SDVariable lastIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(name, in, condition, false, dimensions);
    }

    public SDVariable lastIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        SDVariable ret = f().lastIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     * @param in        Input
     * @param condition Condition
     * @return          Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition) {
        return matchConditionCount(null, in, condition);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     * @param in        Input
     * @param condition Condition
     * @return          Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition) {
        return matchConditionCount(name, in, condition, false);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     * @param in        Input
     * @param condition Condition
     * @return          Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition, boolean keepDim, int... dimensions) {
        SDVariable ret = f().matchConditionCount(in, condition, keepDim, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied
     * @param in        Input
     * @param condition Condition
     * @return          Boolean mask
     */
    public SDVariable matchCondition(SDVariable in, Condition condition){
        return matchCondition(null, in, condition);
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied
     * @param in        Input
     * @param condition Condition
     * @return          Boolean mask
     */
    public SDVariable matchCondition(String name, SDVariable in, Condition condition){
        SDVariable ret = f().matchCondition(in, condition);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable cumsum(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return cumsum(null, in, axis, exclusive, reverse);
    }

    public SDVariable cumsum(String name, SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        SDVariable ret = f().cumsum(in, axis, exclusive, reverse);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable cumprod(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return cumprod(null, in, axis, exclusive, reverse);
    }

    public SDVariable cumprod(String name, SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        SDVariable ret = f().cumprod(in, axis, exclusive, reverse);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable biasAdd(SDVariable input, SDVariable bias) {
        return biasAdd(null, input, bias);
    }

    public SDVariable biasAdd(String name, SDVariable input, SDVariable bias) {
        SDVariable ret = f().biasAdd(input, bias);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @param shape
     * @return
     */
    public SDVariable reshape(SDVariable iX, int... shape) {
        return reshape(null, iX, shape);
    }

    public SDVariable reshape(SDVariable iX, SDVariable shape) {
        return reshape(null, iX, shape);
    }


    /**
     * @param x
     * @param dimensions
     * @return
     */
    public SDVariable reverse(SDVariable x, int... dimensions) {
        return reverse(null, x, dimensions);
    }

    /**
     * @param x
     * @param dimensions
     * @return
     */
    public SDVariable reverse(String name, SDVariable x, int... dimensions) {
        SDVariable ret = f().reverse(x, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        SDVariable ret = f().reverseSequence(x, seq_lengths, seqDim, batchDim);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths) {
        SDVariable ret = f().reverseSequence(x, seq_lengths);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        return reverseSequence(null, x, seq_lengths, seqDim, batchDim);
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        return reverseSequence(null, x, seq_lengths);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths, SDVariable maxLen) {
        SDVariable ret = f().sequenceMask(lengths, maxLen);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen) {
        return sequenceMask(null, lengths, maxLen);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths, int maxLen) {
        SDVariable ret = f().sequenceMask(lengths, maxLen);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths, int maxLen) {
        return sequenceMask(null, lengths, maxLen);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths) {
        SDVariable ret = f().sequenceMask(lengths);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths) {
        SDVariable ret = f().sequenceMask(lengths);
        return updateVariableNameAndReference(ret, null);
    }

    public SDVariable assign(SDVariable x, SDVariable y) {
        return assign(null, x, y);
    }

    public SDVariable assign(String name, SDVariable x, SDVariable y) {
        SDVariable ret = f().assign(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable assign(SDVariable in, Number value) {
        return assign(null, in, value);
    }

    public SDVariable assign(String name, SDVariable in, Number value) {
        SDVariable ret = f().assign(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable transpose(SDVariable iX) {
        return transpose(null, iX);
    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable permute(SDVariable iX, int... dimensions) {
        return permute(null, iX, dimensions);
    }

    /**
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(SDVariable x, int axis) {
        return rollAxis(null, x, axis);
    }

    /**
     * @param dimension
     * @param inputs
     * @return
     */
    public SDVariable concat(int dimension, SDVariable... inputs) {
        return concat(null, dimension, inputs);
    }

    public SDVariable[] moments(SDVariable input, int... axes) {
        return moments(null, input, axes);
    }

    public SDVariable[] moments(String[] name, SDVariable input, int... axes) {
        SDVariable[] res = f().moments(input, axes);
        return updateVariableNamesAndReferences(res, name);
    }

    public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        return normalizeMoments(null, counts, means, variances, shift);
    }

    public SDVariable[] normalizeMoments(String[] name, SDVariable counts, SDVariable means, SDVariable variances,
                                         double shift) {
        SDVariable[] res = f().normalizeMoments(counts, means, variances, shift);
        return updateVariableNamesAndReferences(res, name);
    }

    /**
     * @param iX
     * @param repeat
     * @return
     */
    public SDVariable tile(SDVariable iX, int[] repeat) {
        return tile(null, iX, repeat);
    }

    public SDVariable fill(SDVariable shape, double value) {
        return fill(null, shape, value);
    }


    /**
     *
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(SDVariable input, double inputRetainProbability) {
        return dropout(null, input, inputRetainProbability);
    }

    /**
     *
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(String name, SDVariable input, double inputRetainProbability) {
        SDVariable res = f().dropout(input, inputRetainProbability);
        return updateVariableNameAndReference(res, name);
    }


    public SDVariable xwPlusB(SDVariable input, SDVariable weights, SDVariable bias) {
        return xwPlusB(null, input, weights, bias);
    }

    public SDVariable xwPlusB(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().xwPlusB(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }


    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return reluLayer(null, input, weights, bias);
    }

    public SDVariable reluLayer(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().reluLayer(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * @param x
     * @param y
     * @param transpose
     * @return
     */
    public SDVariable mmul(SDVariable x, SDVariable y, MMulTranspose transpose) {
        return mmul(null, x, y, transpose);

    }

    /**
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(SDVariable x, SDVariable y) {
        return mmul(null, x, y);
    }

    /**
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        return tensorMmul(null, x, y, dimensions);
    }


    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions) {
        return dot(null, x, y, dimensions);
    }

    public SDVariable dot(String name, SDVariable x, SDVariable y, int... dimensions) {
        SDVariable ret = f().dot(x, y, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return cosineSimilarity(generateNewVarName(CosineSimilarity.OP_NAME, 0), iX, i_y, dimensions);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return euclideanDistance(generateNewVarName(EuclideanDistance.OP_NAME, 0), iX, i_y, dimensions);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return manhattanDistance(generateNewVarName(ManhattanDistance.OP_NAME, 0), iX, i_y, dimensions);
    }

    public SDVariable cosineDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return cosineDistance(null, ix, iy, dimensions);
    }

    public SDVariable cosineDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.cosineDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable hammingDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return hammingDistance(null, ix, iy, dimensions);
    }

    public SDVariable hammingDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.hammingDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable jaccardDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return jaccardDistance(null, ix, iy, dimensions);
    }

    public SDVariable jaccardDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.jaccardDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossBinaryXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossBinaryXENT(generateNewVarName(new LossBinaryXENT().opName(), 0), iX, i_y, dimensions);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossCosineSimilarity(generateNewVarName(new LossCosineProximity().opName(), 0), iX, i_y, dimensions);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossHinge(generateNewVarName(new LossHinge().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossKLD(generateNewVarName(new LossKLD().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossL1(generateNewVarName(new LossL1().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossL2(generateNewVarName(new LossL2().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMAE(generateNewVarName(new LossMAE().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMSE(generateNewVarName(new LossMSE().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMCXENT(generateNewVarName(new LossMCXENT().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMSLE(generateNewVarName(new LossMSLE().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossNegativeLogLikelihood(generateNewVarName(new LossNegativeLogLikelihood().opName(), 0), iX, i_y, dimensions);

    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossPoisson(generateNewVarName(new LossPoisson().opName(), 0), iX, i_y, dimensions);

    }


    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossSquaredHinge(generateNewVarName(new LossSquaredHinge().opName(), 0), iX, i_y, dimensions);
    }


    /**
     * @param name
     * @param iX
     * @return
     */
    public SDVariable gradientBackwardsMarker(String name, SDVariable iX) {
        SDVariable result = functionFactory.gradientBackwardsMarker(iX);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable neq(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.neq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eq(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.eq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.gte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.lte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gt(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.gt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lt(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.lt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable neq(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.neq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eq(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.eq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable gt(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable lt(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable or(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.or(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable neg(String name, SDVariable iX) {
        SDVariable result = functionFactory.neg(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isNonDecreasing(SDVariable iX) {
        return isNonDecreasing(null, iX);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isNonDecreasing(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNonDecreasing(iX);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable isStrictlyIncreasing(SDVariable iX) {
        return isStrictlyIncreasing(null, iX);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isStrictlyIncreasing(String name, SDVariable iX) {
        SDVariable result = functionFactory.isStrictlyIncreasing(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param
     * @return
     */
    public SDVariable isNumericTensor(SDVariable iX) {
        return isNumericTensor(null, iX);

    }

    /**
     * @param
     * @return
     */
    public SDVariable isNumericTensor(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNumericTensor(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable cos(String name, SDVariable iX) {
        SDVariable result = functionFactory.cos(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sin(String name, SDVariable iX) {
        SDVariable result = functionFactory.sin(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable tan(String name, SDVariable iX) {
        SDVariable result = functionFactory.tan(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable acos(String name, SDVariable iX) {
        SDVariable result = functionFactory.acos(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */

    public SDVariable asin(String name, SDVariable iX) {
        SDVariable result = functionFactory.asin(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable atan(String name, SDVariable iX) {
        SDVariable result = functionFactory.atan(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable cosh(String name, SDVariable iX) {
        SDVariable result = functionFactory.cosh(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sinh(String name, SDVariable iX) {
        SDVariable result = functionFactory.sinh(iX);
        return updateVariableNameAndReference(result, name);


    }

    /**
     * @param iX
     * @return
     */
    public SDVariable tanh(String name, SDVariable iX) {
        SDVariable
                result = functionFactory.tanh(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable acosh(String name, SDVariable iX) {
        SDVariable result = functionFactory.acosh(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable asinh(String name, SDVariable iX) {
        SDVariable result = functionFactory.asinh(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable atanh(String name, SDVariable iX) {
        SDVariable result = functionFactory.atanh(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable exp(String name, SDVariable iX) {
        SDVariable result = functionFactory.exp(iX);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable expm1(String name, SDVariable iX) {
        SDVariable result = functionFactory.expm1(iX);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @return
     */
    public SDVariable rsqrt(String name, SDVariable iX) {
        SDVariable result = functionFactory.rsqrt(iX);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable log(String name, SDVariable iX) {
        SDVariable result = functionFactory.log(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable log1p(String name, SDVariable iX) {
        SDVariable result = functionFactory.log1p(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isFinite(String name, SDVariable iX) {
        SDVariable result = functionFactory.isFinite(iX);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isInfinite(String name, SDVariable iX) {
        SDVariable result = functionFactory.isInfinite(iX);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable isNaN(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNaN(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable round(String name, SDVariable iX) {
        SDVariable result = functionFactory.round(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(String name, SDVariable iX, double value) {
        SDVariable result = functionFactory.pow(iX, value);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable cube(String name, SDVariable iX) {
        SDVariable result = functionFactory.cube(iX);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable sqrt(String name, SDVariable iX) {
        SDVariable result = functionFactory.sqrt(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable square(String name, SDVariable iX) {
        SDVariable result = functionFactory.square(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable floor(String name, SDVariable iX) {
        SDVariable result = functionFactory.floor(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable relu(String name, SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.relu(iX, cutoff);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable relu6(String name, SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.relu6(iX, cutoff);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softmax(String name, SDVariable iX) {
        SDVariable result = functionFactory.softmax(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softmaxDerivative(String name, SDVariable iX, SDVariable wrt) {
        SDVariable result = functionFactory.softmaxDerivative(iX, wrt);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable hardTanh(String name, SDVariable iX) {
        SDVariable result = functionFactory.hardTanh(iX);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.hardTanhDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sigmoid(String name, SDVariable iX) {
        SDVariable result = functionFactory.sigmoid(iX);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(String name, SDVariable iX, SDVariable wrt) {
        SDVariable result = functionFactory
                .sigmoidDerivative(iX, wrt);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable sign(String name, SDVariable iX) {
        SDVariable result = functionFactory
                .sign(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softsign(String name, SDVariable iX) {
        SDVariable result = functionFactory.softsign(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.softsignDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable softplus(String name, SDVariable iX) {
        SDVariable result = functionFactory.softplus(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable elu(String name, SDVariable iX) {
        SDVariable result = functionFactory.elu(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.eluDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param alpha
     * @return
     */
    public SDVariable leakyRelu(String name, SDVariable iX, double alpha) {
        SDVariable result = functionFactory.leakyRelu(iX, alpha);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param alpha
     * @return
     */
    public SDVariable leakyReluDerivative(String name, SDVariable iX, double alpha) {
        SDVariable result = functionFactory.leakyReluDerivative(iX, alpha);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable mean(String name, SDVariable iX, int... dimension) {
        return mean(name, iX, false, dimension);
    }

    public SDVariable mean(String name, SDVariable iX, boolean keepDims, int... dimension) {
        SDVariable result = functionFactory.mean(iX, keepDims, dimension);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable standardDeviation(String name, SDVariable iX,
                                        boolean biasCorrected,
                                        int... dimensions) {
        return standardDeviation(name, iX, biasCorrected, false, dimensions);
    }

    public SDVariable standardDeviation(String name, SDVariable iX,
                                        boolean biasCorrected,
                                        boolean keepDims,
                                        int... dimensions) {
        SDVariable result = functionFactory.std(
                iX,
                biasCorrected,
                keepDims,
                dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable variance(String name, SDVariable iX, boolean biasCorrected, int... dimensions) {
        return variance(name, iX, biasCorrected, false, dimensions);
    }

    public SDVariable variance(String name, SDVariable iX, boolean biasCorrected, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.variance(iX, biasCorrected, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(String name, SDVariable iX, int... dimensions) {
        return sum(name, iX, false, dimensions);
    }

    public SDVariable sum(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.sum(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(String name, SDVariable iX, int... dimensions) {
        return prod(name, iX, false, dimensions);
    }

    public SDVariable prod(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.prod(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(String name, SDVariable iX, int... dimensions) {
        return max(name, iX, false, dimensions);
    }

    public SDVariable max(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.max(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(String name, SDVariable iX, int... dimensions) {
        return min(name, iX, false, dimensions);
    }

    public SDVariable min(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.min(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }

    public SDVariable norm1(String name, SDVariable ix, int... dimensions) {
        return norm1(name, ix, false, dimensions);
    }

    public SDVariable norm1(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().norm1(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable norm2(String name, SDVariable ix, int... dimensions) {
        return norm2(name, ix, false, dimensions);
    }

    public SDVariable norm2(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().norm2(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable squaredNorm(SDVariable ix, int... dimensions) {
        return squaredNorm(null, ix, false, dimensions);
    }

    public SDVariable squaredNorm(String name, SDVariable ix, int... dimensions) {
        return squaredNorm(name, ix, false, dimensions);
    }

    public SDVariable squaredNorm(SDVariable ix, boolean keepDims, int... dimensions) {
        return squaredNorm(null, ix, keepDims, dimensions);
    }

    public SDVariable squaredNorm(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().squaredNorm(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable normmax(String name, SDVariable ix, int... dimensions) {
        return normmax(name, ix, false, dimensions);
    }

    public SDVariable normmax(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().normmax(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @param shape
     * @return
     */
    public SDVariable reshape(String name, SDVariable iX,
                              int... shape) {
        SDVariable result = functionFactory
                .reshape(iX, shape);
        return updateVariableNameAndReference(result, name);

    }

    public SDVariable reshape(String name, SDVariable iX,
                              SDVariable shape) {
        SDVariable result = functionFactory
                .reshape(iX, shape);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @return
     */
    public SDVariable transpose(String name, SDVariable iX) {
        SDVariable result = functionFactory.transpose(iX);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable permute(String name, SDVariable iX, int... dimensions) {
        SDVariable result = functionFactory.permute(iX, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(String name, SDVariable x, int axis) {
        SDVariable result = functionFactory.rollAxis(x, axis);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param shape
     * @param value
     * @return
     */
    public SDVariable fill(String name, SDVariable shape, double value) {
        SDVariable result = functionFactory.fill(shape, value);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param dimension
     * @param inputs
     * @return
     */
    public SDVariable concat(String name, int dimension, SDVariable... inputs) {
        SDVariable result = functionFactory.concat(dimension, inputs);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param iX
     * @param repeat
     * @return
     */
    public SDVariable tile(String name, SDVariable iX, int[] repeat) {
        SDVariable result = functionFactory.tile(iX, repeat);
        return updateVariableNameAndReference(result, name);

    }


    /**
     * @param x
     * @param y
     * @param transpose
     * @return
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y, MMulTranspose transpose) {
        SDVariable result = functionFactory.mmul(x, y, transpose);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y) {
        return mmul(name, x, y, MMulTranspose.allFalse());
    }

    /**
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(String name,
                                 SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        SDVariable result = functionFactory.tensorMmul(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable cosineSimilarity(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable cosim = functionFactory.cosineSimilarity(
                iX,
                i_y,
                dimensions);
        return updateVariableNameAndReference(cosim, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable euclideanDistance(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.euclideanDistance(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable manhattanDistance(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.manhattanDistance(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable sigmoidCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return sigmoidCrossEntropyWithLogits(null, logits, weights, labels, reductionMode, labelSmoothing);
    }

    public SDVariable sigmoidCrossEntropyWithLogits(String name, SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        SDVariable res = f().sigmoidCrossEntropyWithLogits(logits, weights, labels, reductionMode, labelSmoothing);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable softmaxCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return softmaxCrossEntropyWithLogits(null, logits, weights, labels, reductionMode, labelSmoothing);
    }

    public SDVariable softmaxCrossEntropyWithLogits(String name, SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        SDVariable res = f().softmaxCrossEntropyWithLogits(logits, weights, labels, reductionMode, labelSmoothing);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        return weightedCrossEntropyWithLogits(null, targets, inputs, weights);
    }

    public SDVariable weightedCrossEntropyWithLogits(String name, SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        SDVariable res = f().weightedCrossEntropyWithLogits(targets, inputs, weights);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossBinaryXENT(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossBinaryXENT(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossCosineSimilarity(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossCosineSimilarity(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossHinge(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossHinge(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossKLD(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossKLD(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL1(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossL1(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL2(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossL2(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMAE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMAE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMSE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMCXENT(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMCXENT(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSLE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMSLE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossNegativeLogLikelihood(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossNegativeLogLikelihood(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossPoisson(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossPoisson(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    /**
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossSquaredHinge(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossSquaredHinge(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable expandDims(SDVariable ix, int axis) {
        return expandDims(null, ix, axis);
    }

    public SDVariable expandDims(String name, SDVariable ix, int axis) {
        SDVariable result = f().expandDims(ix, axis);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable squeeze(SDVariable ix, int axis) {
        return squeeze(null, ix, axis);
    }

    public SDVariable squeeze(String name, SDVariable ix, int axis) {
        SDVariable result = f().squeeze(ix, axis);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable predictions) {
        return confusionMatrix((String) null, labels, predictions);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred) {
        SDVariable result = f().confusionMatrix(labels, pred);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses) {
        return confusionMatrix(null, labels, pred, numClasses);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses) {
        SDVariable result = f().confusionMatrix(labels, pred, numClasses);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
        return confusionMatrix(null, labels, pred, weights);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, SDVariable weights) {
        SDVariable result = f().confusionMatrix(labels, pred, weights);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        return confusionMatrix(null, labels, pred, numClasses, weights);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        SDVariable result = f().confusionMatrix(labels, pred, numClasses, weights);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param variable
     */
    public void addVariable(SDVariable variable) {
        if (variableMap == null)
            variableMap = new HashMap<>();

        Preconditions.checkState(variable.getSameDiff() == this, "Samediff instance must be the same.");


        /**
         * Of note here:
         * We don't validate base don vertex id
         * because more than one input can have the same
         * vertex id as a result.
         *
         * We validate based on variable opName instead
         * which takes in to account function names as well
         * as input ids
         */
        if (variableMap.containsKey(variable.getVarName()) && !variableMap.get(variable.getVarName()).equals(variable)) {
            throw new IllegalArgumentException("Variable already found with variable opName " + variable.getVarName());
        }

        Preconditions.checkState(variable.getSameDiff() == this, "Same diff instance for variable must be the same!");
        variableMap.put(variable.getVarName(), variable);

    }


    /**
     * Generate a new variable name
     * based on the uniqueness
     * of thebase name and arg index
     *
     * @param baseName the base name to use (use function.opName() where function is a {@link DifferentialFunction}
     * @param argIndex the arg index
     * @return the new generated name
     */
    public String generateNewVarName(String baseName, int argIndex) {
        if (getVariable(baseName) == null && argIndex == 0) {
            return baseName;
        }

        //need to find a new name
        int count = 1;
        String name = baseName + "_" + count + (argIndex > 0 ? ":" + argIndex : "");
        while (getVariable(name) != null) {
            name = baseName + "_" + (++count) + (argIndex > 0 ? ":" + argIndex : "");
        }

        if (getVariable(name) != null) {
            throw new ND4JIllegalStateException("Converged on already generated variable!");
        }

        return name;
    }


    /**
     * LSTM unit
     *
     * @param baseName      the base name for outputs
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable lstm(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(this, configuration).outputVariables(baseName)[0];
    }


    /**
     * An sru cell
     *
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables()[0];
    }


    /**
     * Simple recurrent  unit
     *
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables()[0];
    }

    /**
     * The gru cell
     *
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables()[0];
    }


    /**
     * An sru cell
     *
     * @param baseName      the base name to  use for the output variables
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(String baseName, SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables(baseName)[0];
    }


    /**
     * Simiple recurrent  unit
     *
     * @param baseName      the base name to use for output variables
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(String baseName, SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables(baseName)[0];
    }

    /**
     * The gru cell
     *
     * @param baseName      the base name for the gru cell
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(String baseName, GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables(baseName)[0];
    }


    public SDVariable slice(SDVariable input, int[] begin, int[] size) {
        return slice(null, input, begin, size);
    }

    public SDVariable slice(String name, SDVariable input, int[] begin, int[] size) {
        SDVariable ret = f().slice(input, begin, size);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stridedSlice(SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    public SDVariable stridedSlice(String name, SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public SDVariable stridedSlice(SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    public SDVariable stridedSlice(String name, SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    public SDVariable stridedSlice(String name, SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stridedSlice(SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    public SDVariable stridedSlice(String name, SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterAdd(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterAdd(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterMul(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterMul(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterSub(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterSub(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterDiv(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterDiv(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterUpdate(null, ref, indices, updates);
    }

    public SDVariable scatterUpdate(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterUpdate(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterAdd(null, ref, indices, updates);
    }

    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterMul(null, ref, indices, updates);
    }

    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterSub(null, ref, indices, updates);
    }

    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterDiv(null, ref, indices, updates);
    }


    /**
     * Generate the variables based on the given input op and return the output variable names.
     *
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function, String baseName) {
        //xyz ops only have 1 output
        //if there is already a base name defined, use that
        if (baseName == null || baseName.isEmpty() && getBaseNameForFunction(function) != null)
            baseName = getBaseNameForFunction(function);

        if (baseName == null)
            baseName = function.opName();

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
                //dynamic shapes
                for (int i = 0; i < ret.length; i++) {
                    SDVariable checkGet = getVariable(baseName);
                    if (checkGet == null) {
                        checkGet = var(generateNewVarName(baseName, i), null, new ZeroInitScheme(ordering));
                    } else if (i > 0 && !importedVarName.contains(baseName)) {
                        //need to find a new name
                        String newName = generateNewVarName(baseName, i);
                        checkGet = getVariable(newName);
                    }
                    if (checkGet == null) {
                        String newName = generateNewVarName(baseName, i);
                        checkGet = var(newName, null, new ZeroInitScheme(ordering));
                    }
                    checkGet.setOutputIndex(i);
                    checkGet.setCreator(function);
                    ret[i] = checkGet;
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
                    checkGet = var(baseName, null, new ZeroInitScheme(ordering));
                } else if (!importedVarName.contains(baseName)) {
                    //need to find a new name
                    String newName = generateNewVarName(baseName, 0);
                    checkGet = var(newName, null, new ZeroInitScheme(ordering));
                }


                if (checkGet == null) {
                    checkGet = var(baseName, null, new ZeroInitScheme(ordering));
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


        char ordering = 'c';
        if (function.args() != null && function.args().length > 0 && function.args()[0].getArr() != null) {
            ordering = function.args()[0].getArr().ordering();
        }

        SDVariable[] ret = new SDVariable[outputShape.size()];

        // ownName/baseName will be used to get variables names
        val ownName = function.getOwnName();
        val rootName = baseName;
        for (int i = 0; i < ret.length; i++) {
            val shape = outputShape.get(i);
            // it should be: rootName:index. i.e.: split:1, split:2, split:3, split:4 etc
            baseName = rootName + (i > 0 ? ":" + i : "");
            SDVariable checkGet = getVariable(baseName);
            if (checkGet == null) {
                // obviously - there's no such var, just add it
                checkGet = var(baseName, shape, new ZeroInitScheme(ordering));
            } else if (shape != null && !shapeAlreadyExistsForVarName(checkGet.getVarName())) {
                // var exists, let's update its shape
                putShapeForVarName(checkGet.getVarName(), shape);
            } else if (shape != null && shapeAlreadyExistsForVarName(checkGet.getVarName())) {
                // no-op.
                // TODO: maybe we should check shapes equality here?
                // it's either var that already exist, or something bad happening
            } else if (!importedVarName.contains(baseName)) {
                // FIXME: dead end.  it's impossible to get here with null as shape
                //need to find a new name
                int count = 1;
                String name = baseName + "_" + count + (i > 0 ? ":" + i : "");
                while (getVariable(name) != null) {
                    count++;
                    name = baseName + "_" + count + (i > 0 ? ":" + i : "");
                }

                if (getVariable(name) != null) {
                    throw new ND4JIllegalStateException("Converged on already generated variable!");
                }


                checkGet = var(name, shape, new ZeroInitScheme(ordering));
            }

            if (checkGet == null) {
                checkGet = var(baseName + (i > 0 ? ":" + i : ""), shape, new ZeroInitScheme(ordering));
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
        return generateOutputVariableForOp(function, function.opName());
    }


    /**
     * Get a function instance
     * given the opName
     *
     * @param functionName the opName of the function
     * @return the same diff function instance
     * defined for the given opName
     */
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }


    /**
     * u
     *
     * @return
     */
    public INDArray execAndEndResult(List<DifferentialFunction> ops) {
        List<DifferentialFunction> exec = exec(ops);
        Op op = (Op) exec.get(exec.size() - 1);
        return op.z();
    }

    /**
     * @return
     */
    public INDArray execAndEndResult() {
        List<DifferentialFunction> exec = exec().getRight();
        val finalOp = exec.get(exec.size() - 1);
        val output = finalOp.outputVariables();
        if (output.length > 1) {
            throw new ND4JIllegalStateException(finalOp.opName() + " has multiple outputs. Use execAndEndResults instead.");
        }
        return output[0].getArr();
    }

    public INDArray[] execAndEndResults() {
        List<DifferentialFunction> exec = exec().getRight();
        val finalOp = exec.get(exec.size() - 1);
        val output = finalOp.outputVariables();
        INDArray outArrays[] = new INDArray[output.length];
        for (int i = 0; i < outArrays.length; i++) {
            outArrays[i] = output[i].getArr();
        }
        return outArrays;
    }

    public INDArray execAndEndResult(int outputIndex) {
        List<DifferentialFunction> exec = exec().getRight();
        val output = exec.get(exec.size() - 1).outputVariables()[outputIndex];
        return output.getArr();
    }


    public INDArray yetAnotherExecMethod(@NonNull Map<String, INDArray> inputs) {
        if (!wasRegistered.get()) {
            synchronized (this) {
                if (!wasRegistered.get()) {
                    val bb = asFlatBuffers();
                    val ptr = new BytePointer(bb);

                    Nd4j.getExecutioner().registerGraph(this.hashCode(), ptr);

                    wasRegistered.set(true);
                }
            }
        }

        val newMap = new LinkedHashMap<String, INDArray>();
        val keySet = inputs.keySet();

        for (val key : keySet) {
            val vx = variableMap.get(key);
            newMap.put(vx.getVarName(), inputs.get(key));
        }

        val result = Nd4j.getExecutioner().executeGraph(this.hashCode(), newMap, this.reverseMap);
        if (result.size() == 0)
            throw new ND4JIllegalStateException("Execution failed");

        val list = new ArrayList<INDArray>(result.values());

        return list.get(list.size() - 1);
    }


    /**
     * Executes the list of operations.
     * This exec method is for
     * only invoking operations
     * rather than creating them
     *
     * @param ops the list of already created ops
     * @return the passes in list
     */
    public List<DifferentialFunction> exec(List<DifferentialFunction> ops) {
        for (int i = 0; i < ops.size(); i++) {
            Op op = (Op) ops.get(i);
            Nd4j.getExecutioner().exec(op);
        }
        return ops;
    }

    public TensorList getListByName(@NonNull String name) {
        return lists.get(name);
    }

    public void putListByName(@NonNull String name, TensorList list) {
        lists.put(name, list);
    }

    /**
     * An interface for representing a conditional statement
     */
    public interface SameDiffConditional {


        /**
         * @param context
         * @param body
         * @return
         */
        SDVariable eval(SameDiff context, SameDiffFunctionDefinition body, SDVariable[] inputVars);

    }

    public static class DefaultSameDiffConditional implements SameDiffConditional {

        @Override
        public SDVariable eval(SameDiff context, SameDiff.SameDiffFunctionDefinition body, SDVariable[] inputVars) {
            context.defineFunction("eval", body, inputVars);
            context.invokeFunctionOn("eval", context);
            return new ArrayList<>(context.functionInstancesById.values()).get(context.functionInstancesById.size() - 1).outputVariables()[0];
        }
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
                                SameDiff.SameDiffFunctionDefinition loopBody
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


    public TensorArrayV3 tensorArray() {
        return new TensorArrayV3(this);
    }

    /**
     * A function definition for
     * samediff
     */
    public interface SameDiffFunctionDefinition {

        /**
         * @param inputs
         * @param variableInputs
         * @return
         */
        SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs);
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
            sub.workspace = (workspace);
            this.child = sub;
            sub.parent = this;
            //setup subgraph
            //re execute to populate subgraph
            SDVariable[] ret = new SDVariable[variables.length];
            for (int i = 0; i < ret.length; i++) {
                ret[i] = sub.var(variables[i]);
            }

            sub.inputs = ret;
            sub.outputs = functionDefinition.define(sub, null, ret);

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
            sub.workspace = (workspace);
            //setup subgraph
            //re execute to populate subgraph
            functionDefinition.define(sub, inputs, null);

            sameDiffFunctionInstances.put(function, sub);
        }

    }


    /**
     * Exec a given function
     *
     * @param functionName the opName of the function
     *                     to invoke
     * @return
     */
    public INDArray execAndEndResult(String functionName) {
        return sameDiffFunctionInstances.get(functionName).execAndEndResult();
    }


    /**
     * Exec a given function
     *
     * @param functionName the opName of the function
     *                     to invoke
     * @return
     */
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec(String functionName) {
        if (debugMode) {
            return sameDiffFunctionInstances.get(functionName).enableDebugMode().exec();
        } else
            return sameDiffFunctionInstances.get(functionName).exec();
    }

    /**
     * Exec the given function
     * given the ops
     *
     * @param functionName the opName of the function to
     *                     exec
     * @param cachedOps    the cached operations
     * @return
     */
    public List<DifferentialFunction> exec(String functionName, List<DifferentialFunction> cachedOps) {
        return sameDiffFunctionInstances.get(functionName).exec(cachedOps);
    }


    /**
     * Builds a backwards graph
     * and executes the operations
     * on that graph.
     *
     * @return
     */
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards() {
        if (getFunction("grad") == null) {
            createGradFunction();
        }


        if (log.isTraceEnabled()) {
            log.trace("About to execute backward function");
        }
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> forward = exec("grad");
        SameDiff grad = getFunction("grad");
        if (grad.isDebugMode()) {
            //ensure all gradients are present for all variables
            for (SDVariable sdVariable : grad.variables()) {
                sdVariable.gradient();
            }
        }

        return forward;
    }

    public void createGradFunction() {
        if (log.isTraceEnabled()) {
            log.trace("Defining function \"grad\"");
        }

        final SameDiff outer = this;
        defineFunction("grad", new SameDiffFunctionDefinition() {

            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                //propagate graph to this samediff instance
                //which will also contain the backward
                if (SameDiff.this.debugMode) {
                    sameDiff.enableDebugMode();
                }

                outer.invokeGraphOn(sameDiff);
                if (debugMode) {
                    //Expect incoming args and outgoing args to be the same
                    Preconditions.checkState(sameDiff.incomingArgsReverse.keySet().equals(incomingArgsReverse.keySet()), "incomingArgsReverse keysets not equal");
                    Preconditions.checkState(sameDiff.outgoingArgsReverse.keySet().equals(outgoingArgsReverse.keySet()), "outgoingArgsReverse keysets not equal");
                }

                List<DifferentialFunction> allFunctions = new ArrayList<>(sameDiff.functionInstancesById.values());
                if (allFunctions.isEmpty()) {
                    throw new ND4JIllegalStateException("No ops found!");
                }


                for (val func : allFunctions) {
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

                val initialOuts = allFunctions.get(allFunctions.size() - 1).outputVariables();
                val firstBackward = initialOuts[0];

                if (log.isTraceEnabled()) {
                    String[] initialOutputsStr = allFunctions.get(allFunctions.size() - 1).outputVariablesNames();
                    String s = initialOutputsStr == null ? "null" : Arrays.toString(initialOutputsStr);
                    log.trace("Defining backward function: initial outputs {}", s);
                }

                //start with scalar backprop
                SDVariable initialGrad = sameDiff.var("one-var", Nd4j.trueScalar(1.0));
                sameDiff.forwardVarForGrad.put(firstBackward.getVarName(), initialGrad);
                sameDiff.gradients.put(firstBackward.getVarName(), initialGrad);

                SDVariable gradientBackwardsMarker = sameDiff.gradientBackwardsMarker(firstBackward);

                //reinitialize list with all declared variables
                allFunctions = new ArrayList<>(sameDiff.functionInstancesById.values());
                Collections.reverse(allFunctions);


                for (int i = 0; i < allFunctions.size(); i++) {
                    DifferentialFunction action = allFunctions.get(i);
                    if (log.isTraceEnabled()) {
                        log.trace("Defining backward function step {} of {}: {} ({}) - {}", (i + 1), allFunctions.size(),
                                action.opName(), action.getOwnName(), action.getClass().getName());
                    }

                    if (action instanceof GradientBackwardsMarker) {
                        continue;
                    }

                    DifferentialFunction currFunction = action;
                    Preconditions.checkState(currFunction.getSameDiff() == sameDiff, "Wrong samediff instance found!");
                    //Preconditions.checkNotNull("Gradient for " + currFunction.opName() + " was null ! " + sameDiff.getVariableForVertexId(currFunction.getVertexId()).getGradient());
                    val args = currFunction.outputVariables();
                    for (val arg : args) {
                        if (arg.getSameDiff() != sameDiff) {
                            arg.setSameDiff(sameDiff);
                        }
                    }


                    List<SDVariable> grads = new ArrayList<>();
                    for (val varToGrad : args) {
                        val grad = varToGrad.gradient();
                        if (grad == null)
                            throw new ND4JIllegalStateException("No gradient found for " + varToGrad.getVarName());
                        grads.add(grad);
                    }

                    List<SDVariable> currFnGrads = currFunction.diff(grads);

                    if (log.isTraceEnabled()) {
                        log.trace("Finished Defining backward function step {} of {}: {} ({}) - {}", (i + 1), allFunctions.size(),
                                action.opName(), action.getOwnName(), action.getClass().getName());
                    }

                    if (debugMode) {
                        //Expect incoming args and outgoing args to be the same
                        Preconditions.checkState(sameDiff.incomingArgsReverse.keySet().equals(sameDiff.outgoingArgsReverse.keySet()),
                                "incomingArgsReverse and outgoingArgsReverse keysets not equal after backprop of function %s of %s: %s (%s)",
                                (i + 1), allFunctions.size(), action.getOwnName(), action.getClass().getName());
                    }
                }


                if (sameDiff.isDebugMode()) {
                    //ensure all gradients are present for all variables
                    for (SDVariable sdVariable : variables()) {
                        sdVariable.gradient();
                    }
                }

                if (log.isTraceEnabled()) {
                    log.trace("Defining backward function complete");
                }

                return new SDVariable[]{sameDiff.var("grad", new int[]{1, 1})};
            }
        });
    }


    /**
     * Exec a backwards operation
     * and return the end result
     *
     * @return
     */
    public INDArray execBackwardAndEndResult() {
        List<DifferentialFunction> backwards = execBackwards().getRight();
        DifferentialFunction df = backwards.get(backwards.size() - 1);
        if (df instanceof Op) {
            return ((Op) df).z();
        } else if (df instanceof DynamicCustomOp) {
            return ((DynamicCustomOp) df).getOutputArgument(0);
        } else {
            return null;
        }
    }


    /**
     * Creates and executes a list of operations
     *
     * @return
     */
    public INDArray execWithPlaceHolderAndEndResult(Map<String, INDArray> inputs) {
        resolveVariablesWith(inputs);
        return execAndEndResult();
    }


    /**
     * Set the original shape for a given place holder.
     * This is used to track original shapes of place holder variables.
     * The reason we track original shapes is to validate
     * possible candidate arrays coming in (especially with -1
     * as the expected shapes).
     * <p>
     * Note that if {@link #isPlaceHolder(String)}
     * returns false for the passed in vertex id,
     * a {@link ND4JIllegalStateException} is thrown.
     * <p>
     * A vertex id must be added first. You can
     * do this with {@link #addAsPlaceHolder(String)}
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
     * Get the original shape for the vertex id if one was set
     * (other wise returns null).
     * This is mainly for use in validating passed in arrays
     * as arguments to {@link #resolveVariablesWith(Map)}
     * usually when executing using {@link #execWithPlaceHolder(Map)}
     *
     * @param varName the vertex id to get the original shape for.
     * @return the set vertex
     */
    public long[] getOriginalShapeForPlaceHolder(String varName) {
        return placeHolderOriginalShapes.get(varName);
    }

    /**
     * Returns true if this vertex id
     * is a place holder variable or not
     *
     * @param varName the vertex id to test
     * @return
     */
    public boolean isPlaceHolder(String varName) {
        return placeHolderVarNames.contains(varName);
    }


    /**
     * Add  this vertex id as a place holder
     *
     * @param varName the vertex id to add
     */
    public void addAsPlaceHolder(String varName) {
        placeHolderVarNames.add(varName);
        if (getVariable(varName) != null && getVariable(varName).getShape() != null) {
            placeHolderOriginalShapes.put(varName, getVariable(varName).getShape());
        }
    }


    /**
     * Resolve all ndarrays by updating the variables
     * for each array specified in the given map.
     * An {@link IllegalStateException} will be thrown
     * if not all arrays are
     * specified for resolution.
     *
     * @param arrays the arrays to resolve.
     */
    public void resolveVariablesWith(Map<String, INDArray> arrays) {
        for (val arrayEntry : arrays.entrySet()) {
            val varForName = getVariable(arrayEntry.getKey());
            if (varForName == null) {
                throw new ND4JIllegalStateException("No variable name found for " + arrayEntry.getKey());
            }

            if (placeHolderOriginalShapes.containsKey(arrayEntry.getKey())) {
                val originalShape = placeHolderOriginalShapes.get(arrayEntry.getKey());
                if (originalShape.length == arrayEntry.getValue().rank()) {
                    for (int i = 0; i < originalShape.length; i++) {
                        if (originalShape[i] != arrayEntry.getValue().shape()[i] && originalShape[i] >= 1) {
                            throw new ND4JIllegalStateException("Incompatible shape passed for variable. " + Arrays.toString(arrayEntry.getValue().shape()));
                        }
                    }
                }
            }
        }


        for (val entry : arrays.entrySet()) {
            if (!placeHolderVarNames.contains(entry.getKey())) {
                throw new ND4JIllegalStateException("Illegal variable " + entry.getKey() + " passed in. Variable found not to be a place holder variable");
            }

            val specifiedShape = getOriginalShapeForPlaceHolder(entry.getKey());
            //whole shape was specified: validate whether the input array shape is equal
            if (!Shape.isPlaceholderShape(specifiedShape)) {
                if (!Shape.shapeEquals(specifiedShape, entry.getValue().shape())) {
                    throw new ND4JIllegalStateException("Place holder shape specified was " + Arrays.toString(specifiedShape) + " but array shape was " + Arrays.toString(entry.getValue().shape()));
                }
            }


            updateShapeForVarName(entry.getKey(), entry.getValue().shape());
            associateArrayWithVariable(entry.getValue(), getVariable(entry.getKey()));
            updateArrayForVarName(entry.getKey(), entry.getValue());

        }


        for (val funcName : propertiesToResolve.keySet()) {
            val func = functionInstancesById.get(funcName);
            if (!functionInstancesById.containsKey(funcName)) {
                throw new ND4JIllegalStateException("Unable to resolve function name " + funcName);
            }

            if (func instanceof CustomOp) {
                CustomOp customOp = (CustomOp) func;
                customOp.populateInputsAndOutputsFromSameDiff();
            }

        }


        //declare resolved
        resolvedVariables = true;
    }

    /**
     * Returns true if all place holder variables
     * are resolved.
     * A place holder variable is resolved when
     * {@link #getVariable(String)}
     * getArr() does not return null and
     * the shape is properly resolved.
     *
     * @return true if all place holder variables are resolved.
     */
    public boolean allPlaceHolderVariablesResolved() {
        for (val vertexId : placeHolderVarNames) {
            val var = getVariable(vertexId);
            if (var.getArr() == null) {
                return false;
            }
        }

        return true;
    }

    /**
     * Add one or or more place holder variables
     * for the given vertex id.
     * <p>
     * Note that if a vertex id in placeHolderVariables
     * isn't present in this samediff instance anyways,
     * an {@link ND4JIllegalStateException} is thrown
     *
     * @param varName              the vertex id to add place holders for
     * @param placeHolderVariables the place holder variables
     */
    public void putPlaceHolderForVariable(String varName, String... placeHolderVariables) {
        for (val placeHolderVariable : placeHolderVariables) {
            if (!variableMap.containsKey(placeHolderVariable)) {
                throw new ND4JIllegalStateException("No variable found for " + placeHolderVariable);
            }
        }


        List<String[]> placeHolders = placeHolderMap.get(varName);
        if (placeHolders == null) {
            placeHolders = new ArrayList<>();
            placeHolderMap.put(varName, placeHolders);
        }

        placeHolders.add(placeHolderVariables);
    }


    /**
     * Returns true if the given vertex id
     * has any placeholder variables
     *
     * @param vertexId the vertex id to check for
     * @return true if this vertex has any place holder
     * variables or not
     */
    public boolean hasPlaceHolderVariables(String vertexId) {
        return placeHolderMap.containsKey(vertexId);
    }

    /**
     * Get the place holders for a given
     * vertex id. May return null.
     * <p>
     * Consider using {@link #hasPlaceHolderVariables(String)}
     *
     * @param varName the vertex id to get the place holders for
     * @return the place holder variables for the given vertex
     * id or null
     */
    public List<String[]> getPlaceHoldersFor(String varName) {
        return placeHolderMap.get(varName);
    }


    /**
     * Creates and executes a list of operations
     * based on the given variables passed in.
     * {@link #resolveVariablesWith(Map)}
     * is called
     *
     * @return
     */
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execWithPlaceHolder(Map<String, INDArray> inputs) {
        resolveVariablesWith(inputs);
        return exec();
    }

    /**
     * Get the {@link SDVariable}
     * associated with each function
     * based on the {@link DifferentialFunction#outputVariables()} ()}
     *
     * @param functions the functions to get the variables for
     * @return the list of variables associated with the given {@link DifferentialFunction}
     */
    public List<SDVariable> getVariablesAssociatedWithFunctions(List<DifferentialFunction> functions) {
        List<SDVariable> ret = new ArrayList<>(functions.size());
        for (DifferentialFunction function : functions) {
            ret.addAll(Arrays.asList(function.outputVariables()));
        }

        return ret;
    }


    /**
     * Updates the variable name
     * property on the passed in variable,
     * the reference in samediff,
     * and returns the variable.
     * <p>
     * Note that if null for the new variable is passed in,
     * it will just return the original input variable.
     *
     * @param varToUpdate the variable to update
     * @param newVarName  the new variable name
     * @return the passed in variable
     */
    public SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName) {
        if (varToUpdate == null) {
            throw new NullPointerException("Null input: No variable found for updating!");
        }

        if (newVarName == null && variableMap.containsKey(varToUpdate.getVarName())) {
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


    /**
     * Updates the variable name property on the passed in variables,
     * its reference in samediff, and returns the variable.
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
     * Creates and executes a list of operations
     *
     * @return
     */


    // required for loops
    private SDVariable[] outputs;
    private SDVariable[] inputs;


    private Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec_cache;

    public void clearExecutionCache(){
        exec_cache = null;
    }

    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec() {

        /*
        if (exec_cache != null){
            return exec_cache;
        }
        */

        if (log.isTraceEnabled()) {
            log.trace("Starting execution: {} functions", functionInstancesById.size());
        }


        if (!resolvedVariables)
            resolveVariablesWith(new LinkedHashMap<String, INDArray>());

        List<DifferentialFunction> ops = new ArrayList<>();

        // we don't care if this thread had any other FlowPath objects attached. we'll just create new one
        localFlowPath.set(new FlowPath());

        val flowPath = localFlowPath.get();

        Map<SDVariable, DifferentialFunction> opMap = new HashMap<>();
        val funcs = new ArrayList<DifferentialFunction>(functionInstancesById.values());
        List<String> funcNames = new ArrayList<>(functionInstancesById.keySet());       //LinkedHashMap, so order for both these vars should be identical
        boolean onBackward = false;

        // dequeue for Frames (nested, probably)
        val frames = new ArrayDeque<String>();

        // simple flag, set true if within frame
        boolean inFrame = false;

        // yet another flag, to remove LastFrame once we really left last frame
        boolean frameLeft = false;

        //If true: this execution includes gradient functions...
        boolean isExecBackwards = functionInstancesById.containsKey(GradientBackwardsMarker.OP_NAME);

        int i = 0;
        int exec_counter = 0;
        for (; i < funcs.size(); i++) {
            ++exec_counter;

            if (log.isTraceEnabled()) {
                val f = funcs.get(i);
                String[] argNames = f.argNames();
                String[] outNames = f.outputVariablesNames();
                log.trace("Starting execution of step {} of {}: Function {} (ownName={}) - {}", exec_counter, funcs.size(),
                        f.opName(), f.getOwnName(), f.getClass().getName());
                log.trace("Function inputs: {} - Function outputs: {}", (argNames == null ? "(none)" : Arrays.toString(argNames)),
                        (outNames == null ? "(none)" : Arrays.toString(outNames)));
                SDVariable[] args = f.args();
                for (int arg = 0; arg < args.length; arg++) {
                    if (args[arg] == null) {
                        log.trace("--> arg {} - {}: argument is null!", arg, argNames[arg]);
                    } else {
                        INDArray arr = args[arg].getArr();
                        String arrShape = (arr == null ? "<array not present>" : Arrays.toString(arr.shape()));
                        log.trace("--> arg {} - {}: array shape: {}", arg, argNames[arg], arrShape);
                    }

                }
            }

            val opName = funcs.get(i).opName();
            if (!onBackward && GradientBackwardsMarker.OP_NAME.equals(opName)) {
                onBackward = true;
            }

            if (GradientBackwardsMarker.OP_NAME.equals(opName))
                continue;

            DifferentialFunction differentialFunction = funcs.get(i);

            if((differentialFunction instanceof ExternalErrorsFunction)) {
                if(isExecBackwards)
                    ((ExternalErrorsFunction) differentialFunction).updateBeforeExecution();

                continue;
            }

            val ownName = differentialFunction.getOwnName();

            // just registering function for this pass
            flowPath.ensureNodeStateExists(differentialFunction.getOwnName());

            if (differentialFunction instanceof SDVariable) {
                if (log.isTraceEnabled()) {
                    log.trace("Skipping differentialFunction that is instanceof SDVariable: {}", opName);
                }
                continue;
            }

            val args = getInputsForFunction(differentialFunction);

            log.debug("Step: {}; Executing op [{}] for node [{}]", exec_counter, opName, ownName);

            // check if inputs are active nodes. skip step otherwise
            // please note: Exit node can't be skipped, because it's either rewind point or exit loop point
            boolean shouldSkip = false;
            if (differentialFunction instanceof Merge) {
                val arg0 = args[0];
                val arg1 = args[1];

                if (!flowPath.isActive(arg0) && !flowPath.isActive(arg1))
                    shouldSkip = true;
            } else {
                if (!(differentialFunction instanceof Exit)) {

                    // if we've left Exit nodes, we can finally delete last frame name
                    if (frameLeft) {
                        frameLeft = false;

                        val frame_name = frames.removeLast();
                        flowPath.activateFrame(frame_name, false);
                        flowPath.forgetFrame(frame_name);
                    }

                    // we must check, if there's inactive nodes used as inputs for this node
                    for (val input : args) {
                        if (!flowPath.isActive(input)) {
                            // propagate inactivity
                            flowPath.markActive(differentialFunction.getOwnName(), false);
                            shouldSkip = true;
                            break;
                        }
                    }
                }
            }

            if (shouldSkip) {
                if (log.isTraceEnabled()) {
                    log.trace("Skipping function {}: shouldSkip = true", opName);
                }
                continue;
            }

            differentialFunction.resolvePropertiesFromSameDiffBeforeExecution();
            flowPath.markActive(differentialFunction.getOwnName(), true);

            /**
             * This set of operations (Enter/Exit/NextIteration/Exit/Switch) are special snowflakes: they modify graph execution order, and basically used here to replicate TF logic.
             * Since SameDiff itself has own logic for loops and conditionals using Scopes
             */
            if (differentialFunction instanceof LoopCond) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of LoopCond op");

                // this node just passes single input forward, for future evaluation
                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                if ((int) array.getDouble(0) == 1) {
                    val frameName = frames.getLast();
                    // incrementing number of cycles for THIS frame, only if LoopCond is true
                    flowPath.incrementNumberOfCycles(frameName);
                }
            } else if (differentialFunction instanceof Enter) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Enter op");

                //  if (flowPath.wasExecuted(differentialFunction.getOwnName()))
                //      continue;

                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                val name = inputs[0].getVarName();

                if (array != null)
                    variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                else {
                    val cleansed = name.replaceAll(":.*","");
                    val list = lists.get(cleansed);
                    if (list != null)
                        lists.put(ownName, list);
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                // frame_name MUST be non-null here
                val frame_name = ((Enter) differentialFunction).getFrameName();
                if (!flowPath.isRegisteredFrame(frame_name)) {
                    flowPath.registerFrame(frame_name);
                    frames.addLast(frame_name);
                    inFrame = true;
                }


            } else if (differentialFunction instanceof Exit) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Exit op");

                // this is just exit point of graph: it maps own input to own output or rewinds graph to specific position planned at first NextIteration node

                val frame_name = frames.getLast();

                // saving frame_name for backward pass
                ((Exit) differentialFunction).setFrameName(frame_name);

                if (!flowPath.isFrameActive(frame_name)) {
                    flowPath.markActive(differentialFunction.getOwnName(), false);

                    // if frame is inactive, lets remove it from queue as well
                    frameLeft = true;
                    continue;
                }

                // Exit node is called in any way, doesn't matters if body was executed or not
                // so, we're checking if rewind was planned (so, NextIteration was executed before Exit)
                // and if it's TRUE - we're setting applying rewind by setting loop idx and calling continue
                if (flowPath.isRewindPlanned(frame_name)) {
                    // just reset loop
                    flowPath.planRewind(frame_name, false);
                    val currentPosition = i;
                    i = flowPath.getRewindPosition(frame_name);
                    val startPosition = i + 1;
                    flowPath.setRewindPosition(frame_name, -1);

                    continue;
                }

                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                val name = inputs[0].getVarName();

                if (array != null)
                    variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                else {
                    val cleansed = name.replaceAll(":.*","");
                    val list = lists.get(cleansed);
                    if (list != null)
                        lists.put(ownName, list);
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                // now it's safe to remove LastFrame
                frameLeft = true;

            } else if (differentialFunction instanceof NextIteration) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of NextIteration op");

                // this operations merges own input, and schedules rewind to specific Merge node
                val inputs = getInputVariablesForFunction(differentialFunction);
                val frame_name = frames.getLast();

                val array = inputs[0].getArr();
                val name = inputs[0].getVarName();

                if (array != null)
                    variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                else {
                    val cleansed = name.replaceAll(":.*","");
                    val list = lists.get(cleansed);
                    if (list != null)
                        lists.put(ownName, list);
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                // if NextIteration wasn't skipped with inactive branch, we'll plan rewind for this frame. obviously, only once
                if (!flowPath.isRewindPlanned(frame_name)) {
                    flowPath.planRewind(frame_name, true);

                    continue;
                }

            } else if (differentialFunction instanceof Merge) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Merge op");

                // merge operation takes two inputs, and saves one of them as own output.
                // if SDVariable exists for second input - we use it. First input used otherwise
                val inputs = getInputVariablesForFunction(differentialFunction);

                val frame_name = frames.size() > 0 ? frames.getLast() : null;

                if (frame_name != null)
                    flowPath.activateFrame(frame_name, true);

                // frame_name can be null if this merge node is used for something that's not loop. i.e. switch/merge pair
                if (frame_name != null)
                    flowPath.setRewindPositionOnce(frame_name, i - 1);

                // NextIteration can have NO frame_name defined. so let's propagate it
                if (inputs.length == 2) {
                    val secondArg = functionInstancesById.get(inputs[1].getVarName());

                    if (secondArg != null && secondArg instanceof NextIteration) {
                        ((NextIteration) secondArg).setFrameName(frame_name);
                    }
                }

                // we must check second input first here
                if (flowPath.wasExecuted(inputs[1].getVarName())) {
                    // propagate second input
                    val array = inputs[1].getArr();
                    val name = inputs[1].getVarName();

                    if (array != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                    else {
                        val cleansed = name.replaceAll(":.*","");
                        val list = lists.get(cleansed);
                        if (list != null)
                            lists.put(ownName, list);
                    }

                    // nullify executed mark
                    flowPath.markExecuted(inputs[1].getVarName(), false);
                } else {
                    // propagate first input
                    val array = inputs[0].getArr();
                    val name = inputs[0].getVarName();

                    if (array != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                    else {
                        val cleansed = name.replaceAll(":.*","");
                        val list = lists.get(cleansed);
                        if (list != null)
                            lists.put(ownName, list);
                    }
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);
            } else if (differentialFunction instanceof Switch) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Switch op");

                // switch takes 2 inputs: actual input and boolean scalar. If scalar is false, input is saved as output:0, if scalar is true, input is saved as output:1
                ((CustomOp) differentialFunction).populateInputsAndOutputsFromSameDiff();

                val inputs = getInputVariablesForFunction(differentialFunction);

                val input = inputs[0].getArr();
                val bool = inputs[1].getArr();
                val name = inputs[0].getVarName();

                // basically we're setting one of the graph branches inactive. branch 0 for false, branch 1 for true
                if ((int) bool.getDouble(0) == 0) {
                    // false step, we'll propagate output:0 here
                    flowPath.setActiveBranch(differentialFunction.getOwnName(), 0);
                    flowPath.markActive(differentialFunction.getOwnName(), true);
                    flowPath.markActive(differentialFunction.getOwnName() + ":1", false);

                    if (input != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), input.dup(input.ordering()));
                    else {
                        val cleansed = name.replaceAll(":.*","");
                        val list = lists.get(cleansed);
                        if (list != null)
                            lists.put(ownName, list);
                    }
                } else {
                    // true step, we'll propagate output:1 here
                    flowPath.setActiveBranch(differentialFunction.getOwnName(), 1);

                    if (input != null)
                        variableNameToArr.put(differentialFunction.getOwnName() + ":1", input.dup(input.ordering()));
                    else {
                        val cleansed = name.replaceAll(":.*","");
                        val list = lists.get(cleansed);
                        if (list != null)
                            lists.put(ownName, list);
                    }

                    flowPath.markActive(differentialFunction.getOwnName(), false);
                    flowPath.markActive(differentialFunction.getOwnName() + ":1", true);
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);
            } else if (differentialFunction instanceof BaseTensorOp) {
                //if(log.isTraceEnabled())
                log.info("Starting execution of Tensor op [{}]", opName);

                // we just pull actual code out of
                val list = ((BaseTensorOp) differentialFunction).execute(this);

                if (!lists.containsKey(list.getName()))
                    lists.put(list.getName(), list);

                ops.add(differentialFunction);
            } else if (differentialFunction instanceof If) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of If op");

                If ifOp = (If) differentialFunction;
                if (!onBackward) {
                    ifOp.getPredicateExecution().exec();
                    //depending on the block add the proper graph body to this for persistence
                    //and possible later processing.
                    if (ifOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                        ifOp.getLoopBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(true);
                    } else {
                        ifOp.getFalseBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(false);

                    }
                } else {
                    if (ifOp.getTrueBodyExecuted() != null) {
                        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = null;
                        List<SDVariable> variablesForFunctions = null;
                        if (ifOp.getTrueBodyExecuted()) {
                            execBackwards = ifOp.getLoopBodyExecution().execBackwards();

                            variablesForFunctions = ifOp.getLoopBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        } else {
                            execBackwards = ifOp.getFalseBodyExecution().execBackwards();
                            variablesForFunctions = ifOp.getFalseBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        }

                        /**
                         * Maps the variables from the child namespace body to
                         * the parent. This allows access to the underlying ndarray
                         * and returning a valid variable reference for autodiff.
                         */
                        for (SDVariable variable : variablesForFunctions) {
                            SDVariable proxyVar = var(variable);
                        }


                    } else
                        throw new ND4JIllegalStateException("No body was run.");

                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(differentialFunction);

            } else if (differentialFunction instanceof While) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of While op");

                While whileOp = (While) differentialFunction;

                if (!onBackward) {
                    SameDiff execBody = whileOp.getLoopBodyExecution();
                    //depending on the block add the proper graph body to this for persistence
                    //and possible later processing.
                    //note that we need to update the graph predicate by running the execution


                    whileOp.getPredicateExecution().exec();
                    if (execBody.outputs == null) {
                        // No explicit inputs/outputs provided.
                        //Op was probably created by tensorflow import.
                        // Non-inplace ops not supported.
                        while (whileOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                            //run the body
                            execBody.exec();
                            whileOp.getPredicateExecution().exec();
                            whileOp.incrementLoopCounter();
                        }
                    } else {
                        if (whileOp.getTargetBoolean().getSameDiff().inputs == null) {
                            whileOp.getTargetBoolean().getSameDiff().inputs = new SDVariable[whileOp.getInputVars().length];
                            for (int e = 0; e < whileOp.getInputVars().length; e++) {
                                whileOp.getTargetBoolean().getSameDiff().inputs[i] = whileOp.getTargetBoolean().getSameDiff().variables().get(i);
                            }
                        }
                        while (whileOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                            //run the body
                            execBody.exec();
                            val outputs = execBody.outputs;

                            int cnt = 0;
                            for (val out : execBody.outputs) {
                                execBody.associateArrayWithVariable(out.getArr(), execBody.inputs[cnt]);
                                whileOp.getTargetBoolean().getSameDiff().associateArrayWithVariable(out.getArr(),
                                        whileOp.getTargetBoolean().getSameDiff().inputs[cnt++]);
                            }
                            //update the predicate
                            whileOp.getPredicateExecution().exec();
                            whileOp.incrementLoopCounter();

                        }
                    }

                    List<SDVariable> outputs = new ArrayList<>();
                    val outputFuncArgs = new ArrayList<>(execBody.functionInstancesById.values()).get(execBody.functionInstancesById.values().size() - 1).outputVariables();
                    outputs.addAll(Arrays.asList(outputFuncArgs));

                    whileOp.setOutputVars(outputs.toArray(new SDVariable[outputs.size()]));
                    ops.add(differentialFunction);
                } else {
                    /**
                     * Note: Need to accumulate gradients.
                     * Multiply each value by the number of times looped.
                     * This approximates accumulating the gradient
                     * across a number of loop cycles.
                     * We only compute the gradient for the internal loop once
                     * and from that we multiply the gradient by 5.
                     *
                     */
                    Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = whileOp.getLoopBodyExecution().execBackwards();
                    for (SDVariable variable : mapListPair.getFirst().keySet()) {
                        variable.getArr().muli(whileOp.getNumLooped());
                    }


                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

            } else if (differentialFunction instanceof CustomOp) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of CustomOp op");


                DynamicCustomOp customOp = (DynamicCustomOp) differentialFunction;

                if (customOp.opName().equalsIgnoreCase("identity")) {
                    val cleansed = args[0].replaceAll(":.*","");
                    val list = lists.get(cleansed);
                    if (list != null) {
                        lists.put(ownName, list);

                        flowPath.markExecuted(differentialFunction.getOwnName(), true);

                        ops.add(customOp);

                        continue;
                    }
                }

                try {
                    customOp.populateInputsAndOutputsFromSameDiff();
                } catch (Throwable t) {
                    throw new RuntimeException("Error populating inputs and outputs for function \"" + differentialFunction.getOwnName()
                            + "\" of type " + differentialFunction.getClass().getName(), t);
                }
                customOp.assertValidForExecution();

                Nd4j.getExecutioner().exec(customOp);

                /*
                if (customOp instanceof LessThanOrEqual) {
                    log.info("Step: {}; InnerCondition: {} <= {} = {}", exec_counter, customOp.getInputArgument(0), customOp.getInputArgument(1), customOp.getOutputArgument(0));
                } else if (customOp instanceof LessThan) {
                    log.info("Step: {}; OuterCondition: {} <= {} = {}", exec_counter, customOp.getInputArgument(0), customOp.getInputArgument(1), customOp.getOutputArgument(0));
                }
                */

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(customOp);
            } else if (differentialFunction instanceof Op) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Op op");

                val inputs = getInputVariablesForFunction(differentialFunction);

                Op op = (Op) differentialFunction;

                // ops in differential function might have stale NDArrays used. we should renew them
                if(inputs != null && inputs.length > 0) {
                    op.setX(inputs[0].getArr());
                    if (inputs.length == 2)
                        op.setY(inputs[1].getArr());
                }

                //Check output shape; allocate a new Z if required
                //For example, if minibatch size has changed since last op execution
                List<long[]> outputShape = ((BaseOp)op).calculateOutputShape();
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                INDArray z = op.z();
                Preconditions.checkNotNull(z, "Could not get output array for op: %s", op.getClass());
                if(!Arrays.equals(outputShape.get(0), z.shape())){
                    if(log.isTraceEnabled()){
                        log.trace("Existing op result (z) array shape for op {} was {}, allocating new array of shape {}",
                                op.getClass().getSimpleName(), Arrays.toString(outputShape.get(0)), Arrays.toString(z.shape()));
                    }
                    //Get output variable:
                    String fnName = funcNames.get(i);
                    String outputName = outgoingArgsReverse.get(fnName)[0];
                    SDVariable outputVar = getVariable(outputName);

                    putOrUpdateShapeForVarName(outputName, outputShape.get(0), true);
                    INDArray newZ = outputVar.storeAndAllocateNewArray();
                    op.setZ(newZ);
                }


                if (differentialFunction.getDimensions() == null)
                    Nd4j.getExecutioner().exec(op);
                else if (op.isExecSpecial()) {
                    op.exec();
                } else {
                    int[] axes = differentialFunction.getDimensions();
                    if (differentialFunction instanceof Accumulation) {
                        Accumulation accumulation = (Accumulation) differentialFunction;

                        Nd4j.getExecutioner().exec(accumulation, axes);

                        if (differentialFunction.outputVariable().getArr() == null) {
                            val var = differentialFunction.outputVariables()[0];
                            updateVariable(var.getVarName(), accumulation.z());
                            updateShapeForVarName(var.getVarName(), accumulation.z().shape());
                        }
                    } else if (differentialFunction instanceof BroadcastOp) {
                        BroadcastOp broadcastOp = (BroadcastOp) differentialFunction;
                        Nd4j.getExecutioner().exec(broadcastOp, axes);
                    } else if (differentialFunction instanceof GradientOp) {
                        Nd4j.getExecutioner().exec(op);
                    } else if (differentialFunction instanceof IndexAccumulation) {
                        IndexAccumulation indexAccumulation = (IndexAccumulation) differentialFunction;
                        Nd4j.getExecutioner().exec(indexAccumulation, axes);

                    } else if (differentialFunction instanceof TransformOp) {
                        TransformOp t = (TransformOp) differentialFunction;
                        Nd4j.getExecutioner().exec(t, axes);
                    }
                }


                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(differentialFunction);
            } else {
                throw new IllegalStateException("Unknown function type: " + differentialFunction.getClass().getName());
            }

            //debug
            // printFunction(differentialFunction);

            if (log.isTraceEnabled()) {
                log.trace("Execution completed for DifferentialFunction {} - {}", opName, differentialFunction.getOwnName());
                SDVariable[] outputVars = differentialFunction.outputVariables();
                for (int x = 0; x < outputVars.length; x++) {
                    INDArray arr = outputVars[x].getArr();
                    String arrShape = (arr == null ? "<no array>" : Arrays.toString(arr.shape()));
                    log.trace("--> output {} - {}: array shape {}", x, outputVars[x].getVarName(), arrShape);
                }
            }
        }

        if (log.isTraceEnabled()) {
            log.trace("Execution complete");
        }

        val ret = new Pair<>(opMap, ops);
        exec_cache = ret;
        if (parent != null) {
            parent.exec_cache = exec_cache;
        }


        return ret;
    }


    /**
     * Print the given function for debugging (will not print functions)
     *
     * @param differentialFunction the function to print
     */
    public void printFunction(DifferentialFunction differentialFunction) {
        if (!logExecution)
            return;
        if (differentialFunction instanceof SDVariable)
            return;

        StringBuilder argShapes = new StringBuilder();
        for (val arg : differentialFunction.args()) {
            argShapes.append(" Variable " + arg.getVarName() +
                    " Shape for " + Arrays.toString(arg.getShape()));
        }

        for (val func : differentialFunction.outputVariables()) {
            argShapes.append("  Output variable " + func.getVarName() + " is " +
                    Arrays.toString(func.getShape()));
        }


        StringBuilder realShapes = new StringBuilder();
        for (val arg : differentialFunction.args()) {
            realShapes.append(" Input shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }

        for (val arg : differentialFunction.outputVariables()) {
            realShapes.append(" Output shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }


//        log.info(realShapes.toString());
    }


    /**
     * Permute indices for the samediff/dl4j format.
     * Due to the dl4j format being NCHW, this is a
     * simple routine for returning permute indices.
     * This is typically used for model import.
     *
     * @param dataFormat the data format to permute
     * @return the permuted indices
     */
    public static int[] permuteDataFormatForSameDiff(String dataFormat, boolean weights) {
        val dl4jFormat = "NCHW";
        dataFormat = dataFormat.toUpperCase();
        //TF: filter_height, filter_width, in_channels, out_channels
        /**
         * N: filter_height
         * H: filter_width
         * W: in_channels
         * C: out_channels
         */


        /**
         *
         *
         */
        //DL4J: filter_height,out_channels,filter_width,in_channels
        // Weights should be: out channels, in channels, height,width
        int[] ret = new int[4];
        if (weights) {
            ret[0] = dataFormat.indexOf('W');
            ret[1] = dataFormat.indexOf('C');
            ret[2] = dataFormat.indexOf('N');
            ret[3] = dataFormat.indexOf('H');
            return ret;
        }


        //NHWC
        //DL4J: NCHW
        for (int i = 0; i < dataFormat.length(); i++) {
            if (dl4jFormat.indexOf(dataFormat.charAt(i)) < 0) {
                throw new ND4JIllegalStateException("Illegal convolution data format string passed in " + dataFormat + " must be some variant of NCHW");
            }
        }

        for (int i = 0; i < dl4jFormat.length(); i++) {
            ret[i] = dl4jFormat.indexOf(dataFormat.charAt(i));
        }

        return ret;
    }

    /**
     * Update the {@link INDArray}
     * ndarray for the given variable name
     *
     * @param variableName the variable to update
     * @param arr          the array to update with
     */
    public void updateVariable(String variableName, INDArray arr) {
        if (!variableNameToArr.containsKey(variableName))
            putArrayForVarName(variableName, arr);
        else
            updateArrayForVarName(variableName, arr);
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
                0.0f, 0, 0);

        return flatNode;
    }

    /**
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

    protected int asFlatNode(@NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables, Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter) {
        val opName = node.opName();
        val hash = getOpNum(node.opName(), node.opType());
        //log.info("Exporting node: [{}:<{}> ; OpType: {}; Hash/opNum: {}]", node.opName(), node.tensorflowName(), node.opType(), hash);

        double[] extras = node.getExtraArgs() != null ? new double[node.getExtraArgs().length] : new double[0];
        for (int e = 0; e < extras.length; e++) {
            extras[e] = ((Number) node.getExtraArgs()[e]).doubleValue();
        }

        long[] extraBits = null;
        if (node.opType() == Op.Type.CUSTOM) {
            DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) node;
            extraBits = dynamicCustomOp.iArgs();
        } else if (node instanceof Enter) {
            // in case of Enter node we'll be storing unique frame reference
            val frameName = ((Enter) node).getFrameName();
            if (!framesMap.containsKey(frameName))
                framesMap.put(frameName, idCounter.incrementAndGet());

            extraBits = new long[]{framesMap.get(frameName).intValue()};
        } else
            extraBits = new long[]{};

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


        val inputs = node.args();
        log.trace("");
        for (val input : inputs) {
            //for (int i = 0; i < outputVertexId.length; i++) {
            val pair = parseVariable(input.getVarName());
            if (!reverseMap.containsKey(pair.getFirst())) {
                if (pair.getFirst().contains("NextIteration")) {
                    // forward declaration: Merge node in case of loop will be referring to NextIteration node, which wasn't announced yet
                    int fwdNodeId = idCounter.incrementAndGet();
                    forwardMap.put(pair.getFirst(), fwdNodeId);
                    reverseMap.put(pair.getFirst(), fwdNodeId);
                } else {
                    throw new ND4JIllegalStateException("Unknown variable used in input: [" + pair.getFirst() + "]");
                }
            }

            int nodeId = reverseMap.get(pair.getFirst());
            int outputIndex = pair.getSecond();

            inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outputIndex));
            //}
        }

        log.debug("Own Name: {}", node.getOwnName());
        int ownId = forwardMap.containsKey(node.getOwnName()) ? forwardMap.get(node.getOwnName()) : idCounter.incrementAndGet();
        reverseMap.put(node.getOwnName(), ownId);

        val dims = node.opType() == Op.Type.REDUCE && inPaired.size() == 1 && node.getDimensions() != null ? node.getDimensions() : new int[]{};
        // TODO: Adam, just put your props here, instead of empty list, and they will be saved
        List<FunctionProperties> props = new ArrayList<>();
        int properties = FunctionProperties.asFlatProperties(bufferBuilder, props);

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, outputIds);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, dims);
        int fname = bufferBuilder.createString(
                outputVertexId == null ||
                        outputVertexId.length < 1 ||
                        outputVertexId[0] == null ? "" :
                        outputVertexId[0].getVarName());
        int scopeName = bufferBuilder.createString("");

        if (node.opType() == null)
            log.warn("Null-op node: {}", node);

        int flatNode = FlatNode.createFlatNode(
                bufferBuilder,
                ownId,
                fname,
                getFlatOpType(node.opType()),
                hash,
                properties,
                nodesIn,
                nodesInPaired,
                (byte) 0,
                nodesOut,
                extraz,
                integerArgs,
                dimensions,
                -1,
                node.opType() == Op.Type.SCALAR && node.getScalarValue() != null ? node.getScalarValue().floatValue() : 0.0f, 0, scopeName);

        return flatNode;
    }


    /**
     * This method exports given SameDiff instance into FlatBuffers
     *
     * @param configuration - ExecutorConfiguration to be embedded into serialized graph
     * @return
     */
    public ByteBuffer asFlatBuffers(@NonNull ExecutorConfiguration configuration) {
        Nd4j.getExecutioner().commit();
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);
        val idCounter = new AtomicInteger(0);

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();

        // first of all we build VariableSpace dump
        List<SDVariable> variableList = new ArrayList<>(variables());
        val reverseMap = new LinkedHashMap<String, Integer>();
        val forwardMap = new LinkedHashMap<String, Integer>();
        val framesMap = new LinkedHashMap<String, Integer>();

        int idx = 0;
        for (val variable : variables()) {
            log.debug("Exporting variable: [{}]", variable.getVarName());
            if (variable.getArr() == null || variable.getShape() == null) {
                //putArrayForVarName(variable.getVarName(), Nd4j.scalar(1.0));
                //addAsPlaceHolder(variable.getVarName());
                continue;
            }


            val pair = parseVariable(variable.getVarName());
            reverseMap.put(pair.getFirst(), idCounter.incrementAndGet());
            log.debug("Adding [{}] as [{}]", pair.getFirst(), idCounter.get());

            val arr = variable.getArr();

            int name = bufferBuilder.createString(variable.getVarName());
            int array = arr.toFlatArray(bufferBuilder);
            int id = IntPair.createIntPair(bufferBuilder, idCounter.get(), 0);


            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
            flatVariables.add(flatVariable);
        }

        //add functions
        for (val func : functionInstancesById.values()) {
            flatNodes.add(asFlatNode(func, bufferBuilder, variableList, reverseMap, forwardMap, framesMap, idCounter));
        }

        // we're dumping scopes now
        for (val scope : sameDiffFunctionInstances.entrySet()) {
            flatNodes.add(asFlatNode(scope.getKey(), scope.getValue(), bufferBuilder));
            val currVarList = new ArrayList<SDVariable>(scope.getValue().variables());
            // converting all ops from node
            for (val node : scope.getValue().variables()) {
                INDArray arr = node.getArr();
                if (arr == null) {
                    //val otherArr = Nd4j.scalar(1.0);
                    //scope.getValue().putArrayForVarName(node.getVarName(), otherArr);
                    //log.warn("Adding placeholder for export for var name {}", node.getVarName());
                    //arr = otherArr;
                    continue;
                }

                int name = bufferBuilder.createString(node.getVarName());
                int array = arr.toFlatArray(bufferBuilder);
                int id = IntPair.createIntPair(bufferBuilder, ++idx, 0);

                val pair = parseVariable(node.getVarName());
                reverseMap.put(pair.getFirst(), idx);

                log.debug("Adding [{}] as [{}]", pair.getFirst(), idx);

                int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
                flatVariables.add(flatVariable);
            }

            //add functions
            for (val func : scope.getValue().functionInstancesById.values()) {
                flatNodes.add(asFlatNode(func, bufferBuilder, currVarList, reverseMap, forwardMap, framesMap, idCounter));
            }

        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        synchronized (this) {
            if (this.reverseMap == null)
                this.reverseMap = reverseMap;
        }

        return bufferBuilder.dataBuffer();
    }

    /**
     * This method exports given SameDiff instance into FlatBuffers
     *
     * @return
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
     * This method just converts enums
     *
     * @param val
     * @return
     */
    public static ByteOrder getOrderFromByte(byte val) {
        if (val == org.nd4j.graph.ByteOrder.LE)
            return ByteOrder.LITTLE_ENDIAN;
        else
            return ByteOrder.BIG_ENDIAN;
    }

    /**
     * This method returns current byte order for this JVM as libnd4j enum
     *
     * @return
     */
    public static byte getOrderAsByte() {
        if (ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN))
            return org.nd4j.graph.ByteOrder.BE;
        else
            return org.nd4j.graph.ByteOrder.LE;
    }

    /**
     * This method converts SameDiff instance to FlatBuffers and saves it to file which can be restored later
     *
     * @param file
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
     * @param file
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
     * This method returns "flattened" graph.
     *
     * @return
     */
    public String asFlatPrint() {
        val sb = new StringBuilder();
        val fb = asFlatBuffers();

        val graph = FlatGraph.getRootAsFlatGraph(fb);

        sb.append("\nExternal variables:\n\n");
        for (int e = 0; e < graph.variablesLength(); e++) {
            val var = graph.variables(e);
            val ndarray = Nd4j.createFromFlatArray(var.ndarray());

            sb.append(var.id().first())
                    .append(":<").append(var.name()).append("> ")
                    .append(Arrays.toString(ndarray.shapeInfoDataBuffer().asInt())).append("; Values: ").append(Arrays.toString(ndarray.data().asFloat())).append(";\n");
        }

        val map = Nd4j.getExecutioner().getCustomOperations();


        sb.append("\nOps sequence:\n\n");
        for (int e = 0; e < graph.nodesLength(); e++) {
            val node = graph.nodes(e);

            log.info("{}:<{}>", node.id(), node.name());
            sb.append(node.id())
                    .append(":<").append(node.name()).append("> ").append(SameDiff.getTypeFromByte(node.opType()));

            if (SameDiff.getTypeFromByte(node.opType()) != Op.Type.CUSTOM)
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
     * This method converts enums for DataType
     *
     * @param val
     * @return
     */
    public static DataBuffer.Type getDataTypeFromByte(byte val) {
        if (val == DataType.FLOAT)
            return DataBuffer.Type.FLOAT;
        else if (val == DataType.DOUBLE)
            return DataBuffer.Type.DOUBLE;
        else if (val == DataType.HALF)
            return DataBuffer.Type.HALF;

        throw new UnsupportedOperationException("Unsupported DataType: [" + val + "]");
    }

    /**
     * This method converts enums for DataType
     *
     * @param type
     * @return
     */
    public static byte getDataTypeAsByte(DataBuffer.Type type) {
        switch (type) {
            case FLOAT:
                return DataType.FLOAT;
            case DOUBLE:
                return DataType.DOUBLE;
            case HALF:
                return DataType.HALF;
            case INT:
                return DataType.INT32;
            case LONG:
                return DataType.INT64;
            default:
                throw new ND4JIllegalStateException("Unknown or unsupported DataType used: [" + type + "]");
        }
    }

    /**
     * This method return operation ID for given op name/type pair.
     *
     * @param name
     * @param type
     * @return
     */
    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.LOOP) {
            return 0;
        } else if (type == Op.Type.RETURN) {
            return 40;
        } else if (type == Op.Type.IF) {
            return 30;
        } else if (type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.MERGE) {
            return 60L;
        } else if (type == Op.Type.LOOP_COND) {
            return 70L;
        } else if (type == Op.Type.NEXT_ITERATION) {
            return 80L;
        } else if (type == Op.Type.EXIT) {
            return 90L;
        } else if (type == Op.Type.ENTER) {
            return 100L;
        } else if (type == Op.Type.CUSTOM) {
            val name2 = Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase());
            if (name2 == null)
                return 0;
            return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();

        } else
            return (long) Nd4j.getOpFactory().getOpNumByName(name);
    }

    /**
     * This method converts enums for DataType
     *
     * @param type
     * @return
     */
    public static Op.Type getTypeFromByte(byte type) {
        switch (type) {
            case OpType.SCALAR:
                return Op.Type.SCALAR;
            case OpType.BROADCAST:
                return Op.Type.BROADCAST;
            case OpType.TRANSFORM:
                return Op.Type.TRANSFORM;
            case OpType.ACCUMULATION:
                return Op.Type.REDUCE;
            case OpType.ACCUMULATION3:
                return Op.Type.REDUCE3;
            case OpType.INDEX_ACCUMULATION:
                return Op.Type.INDEXREDUCE;
            case OpType.RANDOM:
                return Op.Type.RANDOM;
            case OpType.LOGIC:
                return Op.Type.META;
            case OpType.CUSTOM:
                return Op.Type.CUSTOM;
            case OpType.SHAPE:
                return Op.Type.SHAPE;
            case OpType.PAIRWISE:
                return Op.Type.PAIRWISE;
            case OpType.SUMMARYSTATS:
                return Op.Type.SUMMARYSTATS;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }

    /**
     * This method converts enums for DataType
     *
     * @param type
     * @return
     */
    public static byte getFlatOpType(Op.Type type) {
        switch (type) {
            case SCALAR:
                return OpType.SCALAR;
            case BROADCAST:
                return OpType.BROADCAST;
            case TRANSFORM:
            case SPECIAL:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case REDUCE3:
                return OpType.ACCUMULATION3;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case RANDOM:
                return OpType.RANDOM;
            case MERGE:
            case CONDITIONAL:
            case LOOP:
            case RETURN:
            case ENTER:
            case EXIT:
            case NEXT_ITERATION:
            case LOOP_COND:
            case IF:
                return OpType.LOGIC;
            case CUSTOM:
                return OpType.CUSTOM;
            case SHAPE:
                return OpType.SHAPE;
            case PAIRWISE:
                return OpType.PAIRWISE;
            case SUMMARYSTATS:
                return OpType.SUMMARYSTATS;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }


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
        for (String s : varMap.keySet()) {
            String outputOf = null;
            for (Map.Entry<String, String[]> dfToArgs : outgoingArgsReverse.entrySet()) {
                if (dfToArgs.getValue() != null && ArrayUtils.contains(dfToArgs.getValue(), s)) {
                    outputOf = dfToArgs.getKey();
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
        }
        maxLengthOutputOf += 2;

        //Create the output for values:
        format = "%-20s%-20s%-" + maxLengthOutputOf + "s%-20s";
        sb.append(String.format(format, "- Name -", "- Array Shape -", "- Output Of Function -", "- Inputs To Functions -")).append("\n");
        for (String s : varMap.keySet()) {
            INDArray arr = getArrForVarName(s);
            String arrayShape = "-";
            if (arr != null) {
                arrayShape = Arrays.toString(arr.shape());
            }

            List<DifferentialFunction> dfs = functionsArgsFor.get(s);
            String dfArrStr = "";
            if (dfs != null) {
                String[] dfArr = new String[dfs.size()];
                for (int i = 0; i < dfs.size(); i++) {
                    dfArr[i] = dfs.get(i).getOwnName();
                }
                dfArrStr = Arrays.toString(dfArr);
            }

            String outputOfStr = outputOfFn.get(s);

            sb.append(String.format(format, s, arrayShape, outputOfStr, dfArrStr)).append("\n");
        }

        sb.append("\n\n--- Functions ---\n");

        //First: work out the amount of space we need for inputs and outputs...
        List<String> dfInputStr = new ArrayList<>();
        List<String> dfOutputStr = new ArrayList<>();
        int maxInLength = 10;       //Length of "- Inputs -"
        int maxOutLength = 11;      //Length of "- Outputs -"
        int maxOpNameLength = 10;   //Default to min of 10
        int maxDfClassNameLength = 10;  //Default to min of 10
        for (DifferentialFunction df : functions) {
            SDVariable[] args = df.args();
            SDVariable[] outputs = df.outputVariables();

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
}
