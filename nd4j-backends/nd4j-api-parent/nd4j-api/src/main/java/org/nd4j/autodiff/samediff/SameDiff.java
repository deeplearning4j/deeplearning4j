package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import com.rits.cloning.Cloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.FunctionProperties;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * SameDiff is the
 * entrypoint for
 * nd4j's autodiff.
 *
 * You define a graph symbolically.
 *
 * That graph accumulates operations.
 *
 * In order to execute the graph, you run
 * {@link #exec()} to get all the operations
 * {@link #exec(List)} for an already created set of ops
 * {@link #execAndEndResult()} for the end result only
 * {@link #execAndEndResult(List)} for a cached set of ops
 *
 *
 */
@AllArgsConstructor
@Builder
@Slf4j
public class SameDiff {

    private Map<String[],DifferentialFunction> incomingArgs;
    private Map<String[],DifferentialFunction> outgoingArgs;
    private Map<String,String[]> incomingArgsReverse;
    private Map<String,String[]> ougoingArgsReverse;
    private boolean shouldBootStrap = true;
    private Set<String> importedVarName;
    //map a function's instance id to a base name, used for propagating variable names
    //for output during import
    private Map<String,String> baseNameForFunctionInstanceId;

    private DifferentialFunctionFactory functionFactory;
    private Map<String,SDVariable> variableMap;
    private Map<String,int[]> variableNameToShape;
    //gradient information
    private Map<String,SDVariable> gradients;
    private Map<String,SDVariable> forwardVarForGrad;

    private Map<String,INDArray> variableNameToArr;

    //individual index for variable names
    private Map<String,List<DifferentialFunction>> functionsArgsFor;
    private Map<String,List<DifferentialFunction>> functionOutputFor;


    private Map<String,List<String[]>> placeHolderMap;
    private Map<String,int[]> placeHolderOriginalShapes;
    private Set<String> placeHolderVarNames;
    private IdentityHashMap<INDArray,SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;
    private Set<String> placeHolderFunctions;
    private static Cloner cloner = new Cloner();
    private static Map<String,Method> opMethods;

    private  Map<String,DifferentialFunction> functionInstancesById;

    // flag, shows if graph was already registered with libnd4j
    private transient AtomicBoolean wasRegistered = new AtomicBoolean(false);




    //debug mode variables
    @Getter
    private boolean debugMode;
    private Map<int[],Op> opsForResult;
    private boolean resolvedVariables = false;





    static {
        opMethods = new HashMap<>();
        Method[] methods = SameDiff.class.getDeclaredMethods();
        for(Method method : methods) {
            if(method.getReturnType().equals(SDVariable.class)) {
                opMethods.put(method.getName(),method);
            }
        }
    }



    /**
     * Update the opName for the variable
     * with the given vertex id
     * @param varName the vertex id to update
     * @param withName thew new opName
     */
    public void updateVariableName(String varName, String withName) {
        SDVariable oldVarNameRef = getVariable(varName);
        variableMap.remove(oldVarNameRef.getVarName());
        val oldVarName = varName;
        variableMap.put(withName,oldVarNameRef);


        for(val reverseValues : ougoingArgsReverse.entrySet()) {
            for(int i = 0; i < reverseValues.getValue().length; i++) {
                if(reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }


        for(val reverseValues : incomingArgsReverse.entrySet()) {
            for(int i = 0; i < reverseValues.getValue().length; i++) {
                if(reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }

        if(variableNameToArr.containsKey(oldVarName)) {
            val arr = variableNameToArr.remove(oldVarName);
            variableNameToArr.put(withName,arr);
        }


        if(variableNameToShape.containsKey(oldVarName)) {
            val shape = variableNameToShape.remove(oldVarName);
            variableNameToShape.put(withName,shape);
        }


        if(gradients.containsKey(oldVarName)) {
            val grad = gradients.remove(oldVarName);
            gradients.put(withName,grad);
        }

        if(forwardVarForGrad.containsKey(oldVarName)) {
            val forwardGrad = forwardVarForGrad.remove(oldVarName);
            forwardVarForGrad.put(withName,forwardGrad);
        }

        if(placeHolderMap.containsKey(oldVarName)) {
            val placeholders = placeHolderMap.remove(oldVarName);
            placeHolderMap.put(withName,placeholders);
        }


        if(functionsArgsFor.containsKey(oldVarName)) {
            val funcs = functionsArgsFor.remove(oldVarName);
            for(val func : funcs) {
                if(func instanceof BaseOp) {
                    BaseOp baseOp = (BaseOp) func;
                    if(baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                        baseOp.setXVertexId(withName);
                    }

                    if(baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                        baseOp.setYVertexId(withName);
                    }

                    if(baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                        baseOp.setZVertexId(withName);
                    }

                }
            }

            functionsArgsFor.put(withName,funcs);
        }


        if(functionOutputFor.containsKey(oldVarName)) {
            val funcs = functionOutputFor.remove(oldVarName);
            for(val func : funcs) {
                if(func instanceof BaseOp) {
                    BaseOp baseOp = (BaseOp) func;
                    if(baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                        baseOp.setXVertexId(withName);
                    }

                    if(baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                        baseOp.setYVertexId(withName);
                    }

                    if(baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                        baseOp.setZVertexId(withName);
                    }

                }
            }

            functionOutputFor.put(withName,funcs);
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
     * @return
     */
    public DifferentialFunctionFactory f() {
        return functionFactory;
    }



    /**
     *
     * @param sameDiff
     * @return
     */
    public SDVariable invokeGraphOn(SameDiff sameDiff) {
        //map the new vertices on to the old ones
        Map<Integer,Integer> thisVertexIdToNew = new HashMap<>();
        int idx = 1;
        for(val var : variables()) {
            val clone = cloner.deepCloneDontCloneInstances(var,var.getSameDiff());
            val newVar = sameDiff.var(clone);
            if(var.getArr() != null) {
                sameDiff.associateArrayWithVariable(var.getArr(),newVar);
            }


            thisVertexIdToNew.put(idx,idx);
            clone.setSameDiff(sameDiff);
            idx++;

        }




        val newFunctions = new LinkedHashMap<String,DifferentialFunction>();
        for(DifferentialFunction function :functionInstancesById.values())  {
            DifferentialFunction clone = cloner.deepCloneDontCloneInstances(
                    function,
                    function.getSameDiff());
            clone.setSameDiff(sameDiff);
            clone.setInstanceId(function.getInstanceId());
            sameDiff.putFunctionForId(function.getInstanceId(),function);
            newFunctions.put(function.getInstanceId(),clone);

            val argsForFunction = function.args();
            val outputsForFunction = function.outputVariables();


            //note that these have the same variable names
            sameDiff.addArgsFor(argsForFunction,clone);
            sameDiff.addOutgoingFor(outputsForFunction,function);

            for(val arg : clone.args()) {
                arg.setSameDiff(sameDiff);
            }

            for(val output : clone.outputVariables()) {
                output.setSameDiff(sameDiff);
            }

        }

        for(val reverseArrayEntry : reverseArrayLookup.entrySet()) {
            sameDiff.reverseArrayLookup.put(reverseArrayEntry.getKey(),sameDiff.getVariable(reverseArrayEntry.getValue().getVarName()));
        }

        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }




    /**
     * Get the function by the {@link DifferentialFunction#getInstanceId()}
     * @param id the id of the function
     * @return the function for the given id if it exists
     */
    public DifferentialFunction getFunctionById(String id) {
        if(!functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("No function with id " + id + " found!");
        }
        return functionInstancesById.get(id);
    }


    /**
     * Put the function for id
     * @param id the id
     * @param function the function
     */
    public void putFunctionForId(String id,DifferentialFunction function) {
        if(functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("Function by id already exists!");
        }

        functionInstancesById.put(id,function);
    }




    /**
     * Returns the inputs for the given function
     * @param function the function to get the
     *                 inputs for
     * @return the input ids for a given function
     */
    public String[] getInputsForFunction(DifferentialFunction function) {
        if(!incomingArgsReverse.containsKey(function.getInstanceId()))
            throw new ND4JIllegalStateException("Illegal function instance id found " + function.getInstanceId());
        return incomingArgsReverse.get(function.getInstanceId());
    }

    /**
     * Returns the outputs for the given function
     * @param function the function to get the
     *                 inputs for
     * @return the outputs ids for a given function
     */
    public String[] getOutputsForFunction(DifferentialFunction function) {
        return ougoingArgsReverse.get(function.getInstanceId());
    }


    /**
     * Get the output variables given a set of ids
     * from {@link #getOutputsForFunction(DifferentialFunction)}
     * @param function the function reference to get the id for
     * @return the output variables for the given function
     */
    public SDVariable[] getOutputVariablesForFunction(DifferentialFunction function) {
        val inputs = getOutputsForFunction(function);
        if(inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
        }

        return vars;
    }


    /**
     * Get the input variables given a set of ids
     * from {@link #getInputVariablesForFunction(DifferentialFunction)}
     * @param function the function reference to get the id for
     * @return the output variables for the given function
     */
    public SDVariable[] getInputVariablesForFunction(DifferentialFunction function) {
        val inputs = getInputsForFunction(function);
        if(inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
            if(vars[i] == null) {
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            }
        }

        return vars;
    }



    /**
     * Update the ndarray for the given vertex id.
     * @throws {@link ND4JIllegalStateException} when the array does not exist.
     * @param varName
     * @param arr
     */
    public void updateArrayForVarName(String varName, INDArray arr) {
        if(!variableNameToArr.containsKey(varName)) {
            throw new ND4JIllegalStateException("Array for " + varName + " does not exist. Please use putArrayForVertexId instead.");
        }

        variableNameToArr.put(varName,arr);
        reverseArrayLookup.put(arr,getVariable(varName));
    }

    /**
     * Adds an ndarray for a given vertex id.
     * Use {@link #updateArrayForVarName(String, INDArray)}
     * if the array already exists.
     *
     * @param varName the vertex id to add
     * @param arr the array to add
     *
     * @throws {@link ND4JIllegalStateException} when the array already exists.
     */
    public void putArrayForVarName(String varName, INDArray arr) {
        if(varName == null)
            throw new ND4JIllegalStateException("No null names allowed!");

        if(variableNameToArr.containsKey(varName)) {
            throw new ND4JIllegalStateException("Array for " + varName + " already exists!");
        }

        variableNameToArr.put(varName,arr);
    }

    /**
     * Get the shape for the given vertex id.
     * Note that if an array is defined, it will use that shape instead.
     *
     * A shape *and* an array should not be defined at the same time.
     * This wastes memory. The internal map used for tracking shapes for particular
     * vertex ids should also delete redundant shapes stored to avoid redundant sources of information.
     * @param varName the vertex id to get the shape for
     * @return the shape for the given vertex if if any.
     */
    public int[] getShapeForVarName(String varName) {
        if(variableNameToArr.containsKey(varName)) {
            return variableNameToArr.get(varName).shape();
        }



        return variableNameToShape.get(varName);
    }



    /**
     * Update a vertex id with the given shape.
     * Note that you should use {@link #putShapeForVarName(String, int[])}
     * if you want to add a new shape.
     * Update is meant to be an in place replacement
     * of the shape for the vertex id *only*.
     * @param varName the vertex id to associate
     * @param shape the shape to associate with
     */
    public void updateShapeForVarName(String varName, int[] shape) {
        if(shape == null || shape.length < 2) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }


        if(shape == null) {
            throw new ND4JIllegalStateException("Null shapes not allowed!");
        }

        if(variableNameToArr.containsKey(varName) && !Arrays.equals(variableNameToArr.get(varName).shape(),shape)) {
            throw new ND4JIllegalStateException("Already found an existing array!");
        }

        variableNameToShape.put(varName,shape);
    }


    /**
     * Associate a vertex id with the given shape.
     * @param varName the vertex id to associate
     * @param shape the shape to assciate with
     */
    public void putShapeForVarName(String varName, int[] shape) {
        if(shape == null || shape.length < 2) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }

        if(variableNameToShape.containsKey(varName)) {
            throw new ND4JIllegalStateException("Shape for " + varName + " already exists!");
        }

        variableNameToShape.put(varName,shape);
    }




    /**
     * Returns true if the given vertex id
     * and shape already exist.
     * @param varName the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean shapeAlreadyExistsForVarName(String varName) {
        return variableNameToShape.containsKey(varName) || arrayAlreadyExistsForVarName(varName);
    }



    /**
     * Returns true if the given vertex id
     * and {@link INDArray} already exist.
     * @param varName the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean arrayAlreadyExistsForVarName(String varName) {
        return variableNameToArr.containsKey(varName);
    }

    /**
     * Get an {@link INDArray}
     * for a given vertex id
     * @param varName
     * @return
     */
    public INDArray getArrForVarName(String varName) {
        return variableNameToArr.get(varName);
    }

    /**
     * Associate the array with the given variable.
     * @param arr the array to get the variable for
     * @param variable the variable to associate
     */
    public void associateArrayWithVariable(INDArray arr, SDVariable variable) {
        reverseArrayLookup.put(arr,variable);
        variableNameToArr.put(variable.getVarName(),arr);
    }





    /**
     * Associate a {@link SameDiff}
     * namespace as a sub function.
     * @param name the opName of the function
     * @param nameSpace the namespace
     */
    public void putSubFunction(String name,SameDiff nameSpace) {
        if(sameDiffFunctionInstances.containsKey(name) && sameDiffFunctionInstances.get(name) != nameSpace) {
            throw new ND4JIllegalStateException("Unable to replace samediff namespace. Please choose another opName");
        }

        sameDiffFunctionInstances.put(name,nameSpace);
    }


    /**
     * Return the internal variable map
     * @return
     */
    public Map<String,SDVariable> variableMap() {
        return variableMap;
    }


    /**
     * Invoke an op by opName
     * @param op the op
     * @param x the first input
     * @param y the second input
     * @return the result variable
     */
    public SDVariable invoke(Op op,SDVariable x,SDVariable y) {
        if(!opMethods.containsKey(op.opName())) {
            throw new ND4JIllegalStateException("Illegal method opName " + op.opName());
        }

        if(x != null && y != null) {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x, y);
            }catch(Exception e) {

            }
        }
        else {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x);
            }catch(Exception e) {

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
     * @param arr the array reference
     * @return the variable if one exists
     */
    public SDVariable getVariableForArray(INDArray arr) {
        return reverseArrayLookup.get(arr);
    }


    /**
     * The set of defined function names
     * @return
     */
    public Collection<String> definedFunctionNames() {
        return this.sameDiffFunctionInstances.keySet();
    }


    /**
     * Returns the number of bytes
     * for the graph
     * @return
     */
    public long memoryForGraph() {
        return numElements() * DataTypeUtil.lengthForDtype(Nd4j.dataType());
    }

    /**
     * Invoke an op by opName
     * @param op the op
     * @param x the first input
     * @return the result variable
     */
    public SDVariable invoke(Op op,SDVariable x) {
        return invoke(op,x,null);
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
        incomingArgs = new LinkedHashMap<>();
        outgoingArgs = new LinkedHashMap<>();
        incomingArgsReverse = new LinkedHashMap<>();
        ougoingArgsReverse = new LinkedHashMap<>();
        this.functionInstancesById = new LinkedHashMap<>();
        placeHolderFunctions = new LinkedHashSet<>();
        functionsArgsFor = new LinkedHashMap<>();
        functionOutputFor = new LinkedHashMap<>();
        baseNameForFunctionInstanceId = new LinkedHashMap<>();
        importedVarName = new LinkedHashSet<>();

    }

    /**
     * Returns true if the variable name is imported
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
     * @param varName the var name to add.
     */
    public void addVarNameForImport(String varName) {
        importedVarName.add(varName);
    }

    /**
     * Sets a base name for the function id.
     * This is used for when calling {@link #generateOutputVariableForOp(DifferentialFunction,String)}
     * for ensuring original names for model import map to current samediff names
     * when names are generated.
     * @param baseName the base name to add
     * @param function the function to declare a base name for.
     */
    public void setBaseNameForFunctionInstanceId(String baseName,DifferentialFunction function) {
        baseNameForFunctionInstanceId.put(function.getInstanceId(),baseName);
    }

    /**
     * Returns the base name for the given function
     * if any (may return null)
     * @param function the function to get the base name for
     * @return the base name for the given function (if any) based
     * on the function's instance id.
     */
    public String getBaseNameForFunction(DifferentialFunction function) {
        return baseNameForFunctionInstanceId.get(function.getInstanceId());
    }


    /**
     * Attempts to insert the {@link DifferentialFunction}
     * reference in to this {@link SameDiff}
     * instance.
     * If the given array field with the given
     * index already exists, it will do a reference
     * check to ensure that the 2 array fields are the same.
     *
     * If not, an exception is thrown.
     * If the instances are the same (by semantics, not reference)
     * then it will just return the original instance.
     * This is to ensure that instances that are created are unique
     * and reference checked.
     * @param function the array field to attempt to create
     * @return
     */
    public <X extends  SDVariable> X setupFunction(X  function) {
        Preconditions.checkNotNull(function,"Passed in function must not be null!");
        if(function instanceof SDVariable) {
            if(function.getSameDiff() != this) {
                function.setSameDiff(this);
            }
            return function;
        }


        return function;
    }


    /**
     * Adds outgoing args to the graph
     * @param variables
     * @param function
     */
    public void addOutgoingFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for(int i = 0; i < varNames.length; i++) {
            varNames[i] = variables[i].getVarName();
        }

        addOutgoingFor(varNames, function);
    }



    /**
     * Adds outgoing arguments to the graph.
     * Also checks for input arguments
     * and updates the graph adding an appropriate edge
     * when the full graph is declared.
     * @param varNames
     * @param function
     */
    public void addOutgoingFor(String[] varNames, DifferentialFunction function) {

        if(function.getInstanceId() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        if(ougoingArgsReverse.containsKey(function.getInstanceId())) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for " + function);
        }

        if(varNames == null)
            throw new ND4JIllegalStateException("Var names can not be null!");


        for(int i = 0; i < varNames.length; i++) {
            if(varNames[i] == null)
                throw new ND4JIllegalStateException("Variable name elements can not be null!");
        }

        ougoingArgsReverse.put(function.getInstanceId(),varNames);
        outgoingArgs.put(varNames,function);

        for(val resultName : varNames) {
            List<DifferentialFunction> funcs = functionOutputFor.get(resultName);
            if(funcs == null) {
                funcs = new ArrayList<>();
                functionOutputFor.put(resultName,funcs);
            }

            funcs.add(function);
        }

    }

    /**
     * Adds incoming args to the graph
     * @param variables
     * @param function
     */
    public void addArgsFor(String[] variables, DifferentialFunction function) {
        if(function.getInstanceId() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        //double check if function contains placeholder args
        for(val varName : variables) {
            if(isPlaceHolder(varName)) {
                placeHolderFunctions.add(function.getInstanceId());
            }
        }


        incomingArgs.put(variables,function);
        incomingArgsReverse.put(function.getInstanceId(),variables);
        for(val variableName : variables) {
            List<DifferentialFunction> funcs = functionsArgsFor.get(variableName);
            if(funcs == null) {
                funcs = new ArrayList<>();
                functionsArgsFor.put(variableName,funcs);
            }

            funcs.add(function);
        }

    }



    /**
     * Adds incoming args to the graph
     * @param variables
     * @param function
     */
    public void addArgsFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for(int i = 0; i < varNames.length; i++) {
            if(variables[i] == null)
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            varNames[i] = variables[i].getVarName();
        }
        addArgsFor(varNames,function);
    }



    /**
     * Returns true if this function already
     * has defined arguments
     * @param function the function to check
     * @return true if the function has args false otherwise
     */
    public boolean hasArgs(int[] function) {
        return incomingArgs.containsKey(function);
    }


    /**
     * Returns true if this function already
     * has defined arguments
     * @param function the function to check
     * @return true if the function has args false otherwise
     */
    public boolean hasArgs(DifferentialFunction function) {
        val vertexIdArgs = incomingArgsReverse.get(function.getInstanceId());
        if(vertexIdArgs != null) {
            val args = incomingArgs.get(vertexIdArgs);
            if(args != null)
                return true;
        }
        return false;
    }


    public DifferentialFunction[] functions() {
        val ret =  functionInstancesById.values();
        return ret.toArray(new DifferentialFunction[ret.size()]);
    }




    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (variableMap != null ? variableMap.hashCode() : 0);
        return result;
    }




    /**
     *
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
     *
     * @return
     */
    public static SameDiff create() {
        return new SameDiff();
    }



    /**
     * Evaluate the given inputs
     * based on the current graph
     * @param inputs the inputs to evaluate
     * @return
     */
    public INDArray[] eval(Map<String,INDArray> inputs) {

        SameDiff execPipeline = dup();

        List<DifferentialFunction> opExecAction = execPipeline.exec().getRight();
        if(opExecAction.isEmpty())
            throw new IllegalStateException("No ops found to execute.");
        INDArray[] ret = new INDArray[opExecAction.size()];
        for(int i = 0; i < ret.length; i++) {
            val varName = opExecAction.get(i).outputVariables()[0].getVarName();
            ret[i] = execPipeline.getArrForVarName(varName);
        }
        return ret;
    }

    /**
     *
     * @return
     */
    public SameDiff dup() {
        Cloner cloner = new Cloner();
        return cloner.deepClone(this);
    }


    /**
     *
     * @return
     */
    public long numElements() {
        long ret = 0;
        for(SDVariable variable : variables()) {
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
     * @return
     */
    public List<SDVariable> variables() {
        return new ArrayList<>(variableMap.values());
    }

    /**
     * Variable initialization
     * with 1.0
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable one(String name, int[] shape) {
        return var(name,shape,new ConstantInitScheme('f',1.0));

    }


    /**
     * Variable initialization
     * with 0.0
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, int[] shape) {
        return var(name,shape,new ZeroInitScheme('f'));

    }






    /**
     * Variable initialization
     * with a specified {@link WeightInitScheme}
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @param weightInitScheme the weight init scheme
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape, WeightInitScheme weightInitScheme) {
        if(variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            return variableMap.get(name);


        if(name == null || name.length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if(workspace == null)
            initWorkspace();


        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(shape).weightInitScheme(weightInitScheme)
                .varName(name)
                .build();



        addVariable(ret);
        variableMap.put(name,ret);
        return ret;

    }



    /**
     * Creates a {@link SDVariable}
     * with the given shape
     * and a depth of 0.
     *
     * @param name the opName of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape) {
        return var(name,shape,new ZeroInitScheme('f'));

    }


    /**
     * Initialize a {@link SDVariable}
     * reference tying this variable to this
     * samediff instance.
     *
     * {@link NDArraySupplierInitScheme} is used
     * to ensure that if the array is allocated anywhere
     * and {@link SameDiff} instance to exist as a copy of the variable.
     *
     * @param arr
     * @return
     */
    public SDVariable var(final SDVariable arr) {
        if(variableMap.containsKey(arr.getVarName()) && variableMap.get(arr.getVarName()).getArr() != null)
            return variableMap.get(arr.getVarName());


        if(arr.getVarName() == null || arr.getVarName().length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if(arr == null)
            throw new IllegalArgumentException("Array for " + arr.getVarName() + " must not be null");

        if(workspace == null)
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
                        if(arr.getArr() == null) {
                            INDArray retArr =  arr.getWeightInitScheme().create(arr.getShape());
                            associateArrayWithVariable(retArr,arr);
                        }
                        return arr.getArr();
                    }
                }))
                .build();
        variableMap.put(arr.getVarName(),ret);
        return ret;

    }




    /**
     *
     *
     * @param name
     * @param arr
     * @return
     */
    public SDVariable var(String name, INDArray arr) {
        if(variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            return variableMap.get(name);


        if(name == null || name.length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if(arr == null)
            throw new IllegalArgumentException("Array for " + name + " must not be null");

        if(workspace == null)
            initWorkspace();

        val arrRef  = arr.migrate();
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


        associateArrayWithVariable(arr,ret);
        if(ArrayUtil.prod(arr.shape()) == 1)
            ret.setScalarValue(arr.getDouble(0));

        addVariable(ret);
        if(getShapeForVarName(name) == null)
            putShapeForVarName(name,arr.shape());
        //ensure there is a reference to the array in the integer index
        //this is used later for op creation
        reverseArrayLookup.put(arr, ret);
        variableMap.put(name,ret);
        return ret;

    }

    /**
     * Get the variable based on the opName
     * @param name the opName of the variable
     * @return the variabel instance if there is one
     *
     */
    public SDVariable getVariable(String name) {
        return variableMap.get(name);
    }


    /**
     * Get the gradient for the given vertex id
     * @param varName the vertex id
     * @return the gradient for this variable or null
     */
    public SDVariable getGradForVariable(String varName) {
        return gradients.get(varName);
    }


    /**
     * Assign a vertex id
     * to a gradient
     * @param variableName the vertex id
     *                 to assign
     * @param variable the variable
     */
    public void setGradientForVariableName(String variableName, SDVariable variable) {
        if(variable == null) {
            throw new ND4JIllegalStateException("Unable to set null gradient for variable name " + variableName);
        }
        gradients.put(variableName,variable);
    }


    /**
     * Get the forward variable for gradient
     * based on the gradient's vertex id
     * @param vertexId the vertex id
     * @return the gradient for the variable or null
     */
    public SDVariable getForwardVariableForVertexId(int vertexId) {
        return forwardVarForGrad.get(vertexId);
    }


    /**
     *  @param varName
     * @param forwardVariable
     */
    public void setForwardVariableForVarName(String varName, SDVariable forwardVariable) {
        forwardVarForGrad.put(varName,forwardVariable);
    }

    /**
     * Gradient with respect
     * to the given variable opName.
     * Note that in order to run this function,
     * {@link #execBackwards()} must be executed first.
     * All gradient functions are obtained within that time.
     * @param varName the variable opName to get the gradient for.
     * @return
     */
    public SDVariable grad(String varName) {
        if(!sameDiffFunctionInstances.containsKey("grad")) {
            throw new IllegalStateException("Unable to obtain gradient. Please run execBackwards() first.");
        }

        SameDiff grad = getFunction("grad");
        SDVariable var = grad.getVariable(varName);
        return getFunction("grad").getGradForVariable(var.getVarName());
    }


    /**
     * Conv2d operation.
     * @param inputs  the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable conv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        Conv2D conv2D = Conv2D.builder()
                .inputFunctions(inputs)
                .sameDiff(this)
                .conv2DConfig(conv2DConfig)
                .build();

        val outputVertexId = conv2D.outputVariables()[0];
        updateVariableName(outputVertexId.getVarName(),generateVariableName(conv2D.opName(),false,inputs));
        return outputVertexId;
    }


    /**
     * Conv2d operation.
     * @param inputs  the inputs to conv2d
     * @param conv3DConfig the configuration
     * @return
     */
    public SDVariable conv3d(SDVariable[] inputs, Conv3DConfig conv3DConfig) {
        Conv3D conv3D = Conv3D.builder()
                .inputFunctions(inputs)
                .conv3DConfig(conv3DConfig)
                .sameDiff(this)
                .build();

        val outputVars = conv3D.outputVariables();
        updateVariableName(outputVars[0].getVarName(),generateVariableName(conv3D.opName(),false,inputs));
        return outputVars[0];
    }




    /**
     *
     * @param name
     * @param value
     * @return
     */
    public SDVariable scalar(String name, double value) {
        return var(name,Nd4j.scalar(value));
    }







    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(SDVariable iX, double iy) {
        return gte(null,iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, double iy) {
        return lte(null,iX,iy);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, double iy) {
        return lt(null,iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, double iy) {
        return lt(null,iX,iy);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, double iy) {
        return neq(null,iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, double iy) {
        return eq(null,iX,iy);
    }







    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(SDVariable iX, SDVariable iy) {
        return gte(null,iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, SDVariable iy) {
        return lte(null,iX,iy);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, SDVariable iy) {
        return lt(null,iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, SDVariable iy) {
        return lt(null,iX,iy);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        return neq(null,iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, SDVariable iy) {
        return eq(null,iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(SDVariable iX, SDVariable iy) {
        return or(null,iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(SDVariable iX) {
        return neg(null,iX);
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(SDVariable iX) {
        return cos(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(SDVariable iX) {
        return sin(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(SDVariable iX) {
        return tan(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(SDVariable iX) {
        return acos(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(SDVariable iX) {
        return asin(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(SDVariable iX) {
        return atan(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(SDVariable iX) {
        return cosh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(SDVariable iX) {
        return sinh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(SDVariable iX) {
        return tanh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(SDVariable iX) {
        return acosh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(SDVariable iX) {
        return asin(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(SDVariable iX) {
        return atanh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(SDVariable iX) {
        return exp(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(SDVariable iX) {
        return log(null,iX);
    }

    /**
     *
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(SDVariable iX,double value) {
        return pow(null,iX,value);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(SDVariable iX) {
        return sqrt(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(SDVariable iX) {
        return square(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(SDVariable iX) {
        return floor(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(SDVariable iX,double cutoff) {
        return relu(null,iX,cutoff);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(SDVariable iX) {
        return softmax(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return gradientBackwardsMarker(generateVariableName(new GradientBackwardsMarker().opName(),true,iX),iX);
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanh(SDVariable iX) {
        return hardTanh(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(SDVariable iX) {
        return hardTanhDerivative(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(SDVariable iX) {
        return sigmoid(null,iX);
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(SDVariable iX,SDVariable wrt) {
        return sigmoidDerivative(null,iX,wrt);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(SDVariable iX) {
        return sign(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(SDVariable iX) {
        return softsign(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(SDVariable iX) {
        return softsignDerivative(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(SDVariable iX) {
        return softplus(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(SDVariable iX) {
        return elu(null,iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(SDVariable iX) {
        return eluDerivative(null,iX);
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return leakyRelu(null,iX,cutoff);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable mean(SDVariable iX) {
        return mean(null,iX);
    }

    /**
     *
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable standardDeviation(SDVariable iX,
                                        boolean biasCorrected,
                                        int...dimensions) {
        return standardDeviation(null,iX,biasCorrected,dimensions);
    }

    /**
     *
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable variance(SDVariable iX,
                               boolean biasCorrected,
                               int...dimensions) {
        return variance(null,iX,biasCorrected,dimensions);
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(SDVariable iX,
                          int...dimensions) {
        return sum(null,iX,dimensions);
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(SDVariable iX,
                           int...dimensions) {
        return prod(null,iX,dimensions);
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(SDVariable iX, int...dimensions) {
        return max(null,iX,dimensions);

    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(SDVariable iX,
                          int...dimensions) {
        return min(null,iX,dimensions);
    }


    /**
     *
     * @param iX
     * @param shape
     * @return
     */
    public SDVariable reshape(SDVariable iX,
                              int...shape) {
        return reshape(null,iX,shape);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(SDVariable iX) {
        return transpose(null,iX);
    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(SDVariable x, int axis) {
        return rollAxis(null,x,axis);
    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(SDVariable x, SDVariable y) {
        return mmul(null,x,y);
    }

    /**
     *
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        return tensorMmul(null,x,y,dimensions);
    }


    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int...dimensions) {
        return cosineSimilarity(generateVariableName("cosineSimilarity",false,iX,i_y),iX,i_y,dimensions);
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int...dimensions) {
        return euclideanDistance(generateVariableName("euclideandistance",false,iX,i_y),iX,i_y,dimensions);
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int...dimensions) {
        return manhattanDistance(generateVariableName("manhattanDistance",false,iX,i_y),iX,i_y,dimensions);
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossBinaryXENT(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossBinaryXENT(generateVariableName("lossBinaryXENT",false,iX,i_y),iX,i_y,dimensions);
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossCosineSimilarity(null,iX,i_y,dimensions);
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossHinge(generateVariableName("lossHinge",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossKLD(generateVariableName("lossKKLD",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossL1(generateVariableName("lossL1",false,iX),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossL2(generateVariableName("lossL2",false,iX),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossMAE(generateVariableName("lossMAE",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossMSE(generateVariableName("lossMSE",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossMCXENT(generateVariableName("lossMCXENT",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossMSLE(null,iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossNegativeLogLikelihood(generateVariableName("lossNegativeLogLikelihood",false,iX,i_y),iX,i_y,dimensions);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossPoisson(generateVariableName("lossPoisson",false,iX,i_y),iX,i_y,dimensions);

    }


    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int...dimensions) {
        return lossSquaredHinge(generateVariableName("lossPoisson",false,iX,i_y),iX,i_y,dimensions);
    }




    /**
     *
     * @param name
     * @param iX
     * @return
     */
    public SDVariable gradientBackwardsMarker(String name, SDVariable iX) {
        SDVariable result = functionFactory.gradientBackwardsMarker(iX);
        return updateVariableNameAndReference(result,name);
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.neq(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.eq(iX,iy);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX,double iy) {
        SDVariable result = functionFactory.gte(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX,double iy) {
        SDVariable result = functionFactory.lte(iX,iy);
        return updateVariableNameAndReference(result,name);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.gt(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.lt(iX,iy);
        return updateVariableNameAndReference(result,name);

    }





    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.neq(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.eq(iX,iy);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gte(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lte(iX,iy);
        return updateVariableNameAndReference(result,name);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gt(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lt(iX,iy);
        return updateVariableNameAndReference(result,name);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.or(iX,iy);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(String name,SDVariable iX) {
        SDVariable result = functionFactory.neg(iX);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(String name,SDVariable iX) {
        SDVariable result = functionFactory.cos(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(String name,SDVariable iX) {
        SDVariable result = functionFactory.sin(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(String name,SDVariable iX) {
        SDVariable result = functionFactory.tan(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(String name,SDVariable iX) {
        SDVariable result = functionFactory.acos(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(String name,SDVariable iX) {
        SDVariable result = functionFactory.asin(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(String name,SDVariable iX) {
        SDVariable result = functionFactory.atan(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(String name,SDVariable iX) {
        SDVariable result = functionFactory.cosh(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(String name,SDVariable iX) {
        SDVariable result = functionFactory.sinh(iX);
        return updateVariableNameAndReference(result,name);


    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(String name,SDVariable iX) {
        SDVariable
                result = functionFactory.tanh(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(String name,SDVariable iX) {
        SDVariable result = functionFactory.acosh(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(String name,SDVariable iX) {
        SDVariable result = functionFactory.asinh(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(String name,SDVariable iX) {
        SDVariable result = functionFactory.atanh(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(String name,SDVariable iX) {
        SDVariable result = functionFactory.exp(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(String name,SDVariable iX) {
        SDVariable result = functionFactory.log(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(String name,SDVariable iX,double value) {
        SDVariable result = functionFactory.pow(iX,value);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(String name,SDVariable iX) {
        SDVariable result = functionFactory.sqrt(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(String name,SDVariable iX) {
        SDVariable result = functionFactory.square(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(String name,SDVariable iX) {
        SDVariable result = functionFactory.floor(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(String name,SDVariable iX,double cutoff) {
        SDVariable result = functionFactory.relu(iX,cutoff);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(String name,SDVariable iX) {
        SDVariable result = functionFactory.softmax(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmaxDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable result = functionFactory.softmaxDerivative(iX,wrt);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanh(String name,SDVariable iX) {
        SDVariable result = functionFactory.hardTanh(iX);
        return updateVariableNameAndReference(result, name);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.hardTanhDerivative(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(String name,SDVariable iX) {
        SDVariable result = functionFactory.sigmoid(iX);
        return updateVariableNameAndReference(result,name);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable result = functionFactory
                .sigmoidDerivative(iX,wrt);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(String name,SDVariable iX) {
        SDVariable result = functionFactory
                .sign(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(String name,SDVariable iX) {
        SDVariable result = functionFactory.softsign(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.softsignDerivative(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(String name,SDVariable iX) {
        SDVariable result = functionFactory.softplus(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(String name,SDVariable iX) {
        SDVariable result = functionFactory.elu(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.eluDerivative(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(String name,SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.leakyRelu(iX,cutoff);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param wrt
     * @param cutoff
     * @return
     */
    public SDVariable leakyReluDerivative(String name,SDVariable iX, SDVariable wrt,double cutoff) {
        SDVariable result = functionFactory.leakyReluDerivative(iX,
                wrt,
                cutoff);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable mean(String name,SDVariable iX) {
        SDVariable result = functionFactory.mean(iX);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable standardDeviation(String name,SDVariable iX,
                                        boolean biasCorrected,
                                        int...dimensions) {
        SDVariable result = functionFactory.std(
                iX,
                biasCorrected ,
                dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param biasCorrected
     * @param dimensions
     * @return
     */
    public SDVariable variance(String name,SDVariable iX,
                               boolean biasCorrected,
                               int...dimensions) {
        SDVariable result = functionFactory.variance(iX,
                biasCorrected ,
                dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(String name,SDVariable iX,
                          int...dimensions) {
        SDVariable result = functionFactory.sum(iX,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(String name,SDVariable iX,
                           int...dimensions) {
        SDVariable result = functionFactory.prod(iX,dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(String name,SDVariable iX, int...dimensions) {
        SDVariable result = functionFactory.max(iX,dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(String name,SDVariable iX,
                          int...dimensions) {
        SDVariable result = functionFactory.min(iX,dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @param shape
     * @return
     */
    public SDVariable reshape(String name,SDVariable iX,
                              int...shape) {
        shape = Shape.resolveNegativeShapeIfNeccessary(shape,iX.getShape());
        SDVariable result = functionFactory
                .reshape(iX,shape);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(String name,SDVariable iX) {
        SDVariable result = functionFactory.transpose(iX);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(String name,SDVariable x, int axis) {
        SDVariable result = functionFactory.rollAxis(x,axis);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(String name,SDVariable x, SDVariable y) {
        SDVariable result = functionFactory.mmul(x, y);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(String name,
                                 SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        SDVariable result = functionFactory.tensorMmul(x,y, dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable cosineSimilarity(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable cosim = functionFactory.cosineSimilarity(
                iX,
                i_y,
                dimensions);
        return updateVariableNameAndReference(cosim,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable euclideanDistance(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.euclideanDistance(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable manhattanDistance(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.manhattanDistance(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossBinaryXENT(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossBinaryXENT(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossCosineSimilarity(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossCosineSimilarity(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossHinge(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossHinge(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossKLD(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossKLD(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL1(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossL1(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL2(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossL2(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMAE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossMAE(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }



    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossMSE(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMCXENT(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossMCXENT(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSLE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossMSLE(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossNegativeLogLikelihood(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossNegativeLogLikelihood(iX, i_y, dimensions);
        return updateVariableNameAndReference(result,name);

    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossPoisson(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossPoisson(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossSquaredHinge(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        SDVariable result = functionFactory.lossSquaredHinge(iX,i_y,dimensions);
        return updateVariableNameAndReference(result,name);

    }


    /**
     *
     * @param variable
     */
    public void addVariable(SDVariable variable) {
        if(variableMap == null)
            variableMap = new HashMap<>();

        Preconditions.checkState(variable.getSameDiff() == this,"Samediff instance must be the same.");


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
        if(variableMap.containsKey(variable.getVarName()) && !variableMap.get(variable.getVarName()).equals(variable)) {
            throw new IllegalArgumentException("Variable already found with variable opName " + variable.getVarName());
        }

        Preconditions.checkState(variable.getSameDiff() == this,"Same diff instance for variable must be the same!");
        variableMap.put(variable.getVarName(),variable);

    }


    /**
     * Generate a new variable name
     * based on the uniqueness
     * of thebase name and arg index
     * @param baseName the base name to use (use function.opName() where function is a {@link DifferentialFunction}
     * @param argIndex the arg index
     * @return the new generated name
     */
    public String generateNewVarName(String  baseName,int argIndex) {
        //need to find a new name
        int count = 1;
        String name = baseName + "_" + count   + (argIndex > 0 ? ":" + argIndex : "");
        while(getVariable(name) != null) {
            count++;
            name = baseName + "_" + count   + (argIndex > 0 ? ":" + argIndex : "");
        }

        if(getVariable(name) != null) {
            throw new ND4JIllegalStateException("Converged on already generated variable!");
        }


        return name;
    }


    /**
     * LSTM unit
     * @param baseName the base name for outputs
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable lstm(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(this,configuration).outputVariables(baseName)[0];
    }




    /**
     * An sru cell
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell( SRUCellConfiguration configuration) {
        return new SRUCell(this,configuration).outputVariables()[0];
    }


    /**
     * Simiple recurrent  unit
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru( SRUConfiguration configuration) {
        return new SRU(this,configuration).outputVariables()[0];
    }

    /**
     * The gru cell
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(GRUCellConfiguration configuration) {
        return new GRUCell(this,configuration).outputVariables()[0];
    }



    /**
     * An sru cell
     * @param baseName the base name to  use for the output variables
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(String baseName, SRUCellConfiguration configuration) {
        return new SRUCell(this,configuration).outputVariables(baseName)[0];
    }


    /**
     * Simiple recurrent  unit
     * @param baseName the base name to use for output variables
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(String baseName, SRUConfiguration configuration) {
        return new SRU(this,configuration).outputVariables(baseName)[0];
    }

    /**
     * The gru cell
     * @param baseName the base name for the gru cell
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(String baseName, GRUCellConfiguration configuration) {
        return new GRUCell(this,configuration).outputVariables(baseName)[0];
    }


    /**
     * Generate the variables based on the given input op
     * and return the output variable names.
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function,String baseName) {
        //xyz ops only have 1 output
        //if there is already a base name defined, use that
        if(baseName == null || baseName.isEmpty()   && getBaseNameForFunction(function) != null)
            baseName = getBaseNameForFunction(function);

        if(baseName == null)
            baseName = function.opName();


        val outputShape = function.calculateOutputShape();
        if(outputShape == null || outputShape.isEmpty()) {
            if(function instanceof CustomOp) {
                CustomOp customOp = (CustomOp) function;
                val descriptor = customOp.getDescriptor();
                //can't guess number of outputs, variable
                if(descriptor == null || descriptor.getNumOutputs() <= 0) {
                    return new SDVariable[0];
                }
                else {
                    SDVariable[] ret = new SDVariable[descriptor.getNumOutputs()];
                    //dynamic shapes
                    for(int i = 0; i < ret.length; i++) {
                        SDVariable checkGet = getVariable(baseName);
                        if(checkGet == null) {
                            checkGet = var(generateNewVarName(baseName,i),null,new ZeroInitScheme('f'));
                        }
                        else if(!importedVarName.contains(baseName)) {
                            //need to find a new name
                            String newName  = generateNewVarName(baseName,i);
                            checkGet = getVariable(newName);
                        }


                        if(checkGet == null) {
                            String newName  = generateNewVarName(baseName,i);
                            checkGet = var(newName,null,new ZeroInitScheme('f'));
                        }


                        ret[i] = checkGet;
                    }

                    return ret;

                }
            }

            //this is for unresolved shapes, we know xyz is always 1 outputu
            else if(function instanceof BaseOp && outputShape.isEmpty()) {
                SDVariable[] ret = new SDVariable[1];
                SDVariable checkGet = getVariable(baseName);
                if(checkGet == null) {
                    checkGet = var(baseName ,null,new ZeroInitScheme('f'));
                }
                else if(!importedVarName.contains(baseName)) {
                    //need to find a new name
                    String newName  = generateNewVarName(baseName,0);
                    checkGet = var(newName,null,new ZeroInitScheme('f'));
                }


                if(checkGet == null) {
                    checkGet = var(baseName,null,new ZeroInitScheme('f'));
                }

                ret[0] = checkGet;
                return ret;

            }
        }




        SDVariable[] ret = new SDVariable[outputShape.size()];

        for(int i = 0; i < ret.length; i++) {
            val shape = outputShape.get(i);
            SDVariable checkGet = getVariable(baseName);
            if(checkGet == null) {
                checkGet = var(baseName + (i > 0 ? ":" +  i : ""),shape);
            }
            else if(!importedVarName.contains(baseName)) {
                //need to find a new name
                int count = 1;
                String name = baseName + "_" + count   + (i > 0 ? ":" +  i : "");
                while(getVariable(name) != null) {
                    count++;
                    name = baseName + "_" + count   + (i > 0 ? ":" +  i : "");
                }

                if(getVariable(name) != null) {
                    throw new ND4JIllegalStateException("Converged on already generated variable!");
                }


                checkGet = var(name,shape);
            }

            else if(shape != null)
                putShapeForVarName(checkGet.getVarName(),shape);

            if(checkGet == null) {
                checkGet = var(baseName + (i > 0 ? ":" +  i : ""),shape);
            }


            ret[i] = checkGet;
        }


        return ret;
    }

    /**
     * Generate the variables based on the given input op
     * and return the output variable names.
     * @param function the function to generate the output
     *                 variable names for
     * @return the set of names generated for each output of the function.
     */
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function) {
        return generateOutputVariableForOp(function,function.opName());
    }


    /**
     *
     * @param funcName
     * @param grad
     * @param inputs
     * @return
     */
    public String generateVariableName(String funcName,boolean grad,SDVariable...inputs) {
        StringBuilder sb = new StringBuilder();
        sb.append(funcName).append("(");
        if(inputs != null) {
            for (SDVariable variable : inputs) {
                if(variable == null) {
                    throw new ND4JIllegalStateException("Found null variable when attempting to generate variable name for function " + funcName);
                }
                sb.append(variable.getVarName());
                if (grad) {
                    sb.append("-grad");
                }

                sb.append("-");


                sb.append(",");
            }
        }


        return sb.toString();

    }




    /**
     * Get a function instance
     * given the opName
     * @param functionName the opName of the function
     * @return the same diff function instance
     * defined for the given opName
     */
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }



    /**
     *u
     * @return
     */
    public INDArray execAndEndResult(List<DifferentialFunction> ops) {
        List<DifferentialFunction> exec = exec(ops);
        Op op = (Op) exec.get(exec.size() - 1);
        return op.z();
    }

    /**
     *
     *  @return
     */
    public INDArray execAndEndResult() {
        resolveVariablesWith(Collections.<String, INDArray>emptyMap());
        List<DifferentialFunction> exec = exec().getRight();
        val output =  exec.get(exec.size() - 1).outputVariables()[0];
        return output.getArr();
    }


    public INDArray yetAnotherExecMethod(@NonNull Map<String, INDArray> inputs){
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

        for (val key: keySet) {
            val vx = variableMap.get(key);
            newMap.put(vx.getVarName(), inputs.get(key));
        }

        val result = Nd4j.getExecutioner().executeGraph(this.hashCode(), newMap);
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
     * @param ops the list of already created ops
     * @return the passes in list
     */
    public List<DifferentialFunction> exec(List<DifferentialFunction> ops) {
        for(int i = 0; i < ops.size(); i++) {
            Op op = (Op) ops.get(i);
            Nd4j.getExecutioner().exec(op);
        }
        return ops;
    }


    /**
     * An interface for representing a conditional statement
     */
    public interface SameDiffConditional {


        /**
         *
         * @param context
         * @param body
         * @return
         *
         * * @param inputVars
         * @return
         */
        SDVariable eval(SameDiff context, SameDiffFunctionDefinition body, SDVariable[] inputVars);

    }

    public static class DefaultSameDiffConditional implements SameDiffConditional {

        @Override
        public SDVariable eval(SameDiff context, SameDiff.SameDiffFunctionDefinition body, SDVariable[] inputVars) {
            context.defineFunction("eval",body,inputVars);
            context.invokeFunctionOn("eval",context);
            return new ArrayList<>(context.functionInstancesById.values()).get(context.functionInstancesById.size() - 1).outputVariables()[0];
        }
    }


    /**
     * Creates a while statement
     * @param sameDiffConditional
     * @param loopBody
     * @return
     */
    public While whileStatement(SameDiffConditional sameDiffConditional,
                                SameDiffFunctionDefinition conditionBody,
                                SameDiff.SameDiffFunctionDefinition loopBody
            ,SDVariable[] inputVars) {
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
     *
     * @param conditional
     * @param trueBody
     * @param falseBody
     * @return
     */
    public If ifStatement(SameDiffConditional conditional,
                          SameDiffFunctionDefinition conditionBody,
                          SameDiffFunctionDefinition trueBody,
                          SameDiffFunctionDefinition falseBody
            ,SDVariable[] inputVars) {
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


    /**
     * A function definition for
     * samediff
     */
    public interface SameDiffFunctionDefinition {

        /**
         *
         * @param inputs
         * @param variableInputs
         * @return
         */
        SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs);
    }

    /**
     *
     * @param functionName
     * @param with
     */

    public SDVariable invokeFunctionOn(String functionName,SameDiff with) {
        SameDiff instance = sameDiffFunctionInstances.get(functionName);
        SDVariable ret = instance.invokeGraphOn(with);

        return ret;
    }




    /**
     *
     * @param function
     */
    public SameDiff defineFunction(String function,SameDiffFunctionDefinition functionDefinition,SDVariable[] variables) {
        if(!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            sub.workspace = (workspace);
            //setup subgraph
            //re execute to populate subgraph
            SDVariable[] ret = new SDVariable[variables.length];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = sub.var(variables[i]);
            }

            functionDefinition.define(sub,null, ret);
            sameDiffFunctionInstances.put(function,sub);
        }

        return sameDiffFunctionInstances.get(function);
    }

    /**
     *
     * @param function
     */
    public void defineFunction(String function,SameDiffFunctionDefinition functionDefinition) {
        defineFunction(function,functionDefinition, new LinkedHashMap<String,INDArray>());
    }

    /**
     *
     * @param function
     * @param functionDefinition
     * @param inputs
     */
    public void defineFunction(String function,
                               SameDiffFunctionDefinition functionDefinition,
                               Map<String,INDArray> inputs) {
        if(!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            sub.workspace = (workspace);
            //setup subgraph
            //re execute to populate subgraph
            functionDefinition.define(sub,inputs, null);

            sameDiffFunctionInstances.put(function,sub);
        }

    }




    /**
     * Exec a given function
     * @param functionName the opName of the function
     *                     to invoke
     * @return
     */
    public INDArray execAndEndResult(String functionName) {
        return sameDiffFunctionInstances.get(functionName).execAndEndResult();
    }


    /**
     * Exec a given function
     * @param functionName the opName of the function
     *                     to invoke
     * @return
     */
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec(String functionName) {
        if(debugMode) {
            return sameDiffFunctionInstances.get(functionName).enableDebugMode().exec();

        }
        else
            return sameDiffFunctionInstances.get(functionName).exec();
    }

    /**
     * Exec the given function
     * given the ops
     * @param functionName the opName of the function to
     *                     exec
     * @param cachedOps the cached operations
     * @return
     */
    public List<DifferentialFunction> exec(String functionName,List<DifferentialFunction> cachedOps) {
        return sameDiffFunctionInstances.get(functionName).exec(cachedOps);
    }


    /**
     * Builds a backwards graph
     * and executes the operations
     * on that graph.
     * @return
     */
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards() {

        final SameDiff outer = this;
        if(getFunction("grad") == null)
            defineFunction("grad", new SameDiffFunctionDefinition() {

                @Override
                public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                    //propagate graph to this samediff instance
                    //which wil also contain the backward
                    if(SameDiff.this.debugMode) {
                        sameDiff.enableDebugMode();
                    }

                    outer.invokeGraphOn(sameDiff);

                    List<DifferentialFunction> allFunctions = new ArrayList<DifferentialFunction>(sameDiff.functionInstancesById.values());
                    if(allFunctions.isEmpty()) {
                        throw new ND4JIllegalStateException("No ops found!");
                    }


                    for(val func : allFunctions) {
                        val args = func.args();
                        for(val arg : args)
                            arg.setSameDiff(sameDiff);
                        val outputs = func.outputVariables();
                        for(val output : outputs)
                            output.setSameDiff(sameDiff);
                        func.setSameDiff(sameDiff);
                    }

                    val initialOuts =  allFunctions.get(allFunctions.size() - 1).outputVariables();
                    val firstBackward = initialOuts[0];

                    //start with scalar backprop
                    SDVariable initialGrad = sameDiff.var("one-var",Nd4j.scalar(1.0));
                    sameDiff.forwardVarForGrad.put(firstBackward.getVarName(),initialGrad);
                    sameDiff.gradients.put(firstBackward.getVarName(),initialGrad);

                    SDVariable gradientBackwardsMarker = sameDiff.gradientBackwardsMarker(firstBackward);

                    //reinitialize list with all declared variables
                    allFunctions = new ArrayList<DifferentialFunction>(sameDiff.functionInstancesById.values());
                    Collections.reverse(allFunctions);


                    for(DifferentialFunction action : allFunctions) {
                        if(action instanceof GradientBackwardsMarker) {
                            log.warn("Action op state is null");
                            continue;
                        }

                        DifferentialFunction currFunction = action;
                        Preconditions.checkState(currFunction.getSameDiff() == sameDiff,"Wrong samediff instance found!");
                        //Preconditions.checkNotNull("Gradient for " + currFunction.opName() + " was null ! " + sameDiff.getVariableForVertexId(currFunction.getVertexId()).getGradient());
                        val args = currFunction.outputVariables();
                        for(val arg : args) {
                            if(arg.getSameDiff() != sameDiff) {
                                arg.setSameDiff(sameDiff);
                            }
                        }


                        List<SDVariable> grads = new ArrayList<>();
                        for(val varToGrad : args) {
                            val grad = varToGrad.gradient();
                            if(grad == null)
                                throw new ND4JIllegalStateException("No gradient found for " + varToGrad.getVarName());
                            grads.add(grad);
                        }

                        currFunction.diff(grads);


                    }


                    if(sameDiff.isDebugMode()) {
                        //ensure all gradients are present for all variables
                        for(SDVariable sdVariable : variables()) {
                            sdVariable.gradient();
                        }
                    }


                    return new   SDVariable[] {sameDiff.var("grad",new int[] {1,1})};
                }
            });


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> forward = exec("grad");
        SameDiff grad = getFunction("grad");
        if(grad.isDebugMode()) {
            //ensure all gradients are present for all variables
            for(SDVariable sdVariable : grad.variables()) {
                sdVariable.gradient();
            }
        }

        return forward;
    }


    /**
     * Exec a backwards operation
     * and return the end result
     * @return
     */
    public INDArray execBackwardAndEndResult() {
        List<DifferentialFunction> backwards = execBackwards().getRight();
        Op op = (Op) backwards.get(backwards.size() - 1);
        return op.z();
    }




    /**
     * Creates and executes a list of operations
     * @return
     */
    public INDArray execWithPlaceHolderAndEndResult(Map<String,INDArray> inputs) {
        resolveVariablesWith(inputs);
        return execAndEndResult();
    }


    /**
     * Set the original shape for a given place holder.
     * This is used to track original shapes of place holder variables.
     * The reason we track original shapes is to validate
     * possible candidate arrays coming in (especially with -1
     * as the expected shapes).
     *
     * Note that if {@link #isPlaceHolder(String)}
     * returns false for the passed in vertex id,
     * a {@link ND4JIllegalStateException} is thrown.
     *
     * A vertex id must be added first. You can
     * do this with {@link #addAsPlaceHolder(String)}
     *  @param variableName the vertex id for the original shape
     * @param shape the shape of the place holder
     */
    public void setOriginalPlaceHolderShape(String variableName, int[] shape) {
        if(!isPlaceHolder(variableName)) {
            throw  new ND4JIllegalStateException("Vertex id " + variableName + " does not appear to be a place holder. Did you forget to call addPlaceHolder?");
        }

        if(shape == null) {
            throw new ND4JIllegalStateException("Null and 0 length shape arrays not allowed");
        }


        if(placeHolderOriginalShapes.containsKey(variableName) && !Arrays.equals(placeHolderOriginalShapes.get(variableName),shape)) {
            throw new ND4JIllegalStateException("Unable to add a new shape for vertex id " + variableName);
        }

        //after validation now only set once
        placeHolderOriginalShapes.put(variableName,shape);

    }


    /**
     * Get the original shape for the vertex id if one was set
     * (other wise returns null).
     * This is mainly for use in validating passed in arrays
     * as arguments to {@link #resolveVariablesWith(Map)}
     * usually when executing using {@link #execWithPlaceHolder(Map)}
     * @param varName the vertex id to get the original shape for.
     *
     * @return the set vertex
     */
    public int[] getOriginalShapeForPlaceHolder(String varName) {
        return placeHolderOriginalShapes.get(varName);
    }

    /**
     * Returns true if this vertex id
     * is a place holder variable or not
     * @param varName the vertex id to test
     * @return
     */
    public boolean isPlaceHolder(String varName) {
        return placeHolderVarNames.contains(varName);
    }


    /**
     * Add  this vertex id as a place holder
     * @param varName the vertex id to add
     */
    public void addAsPlaceHolder(String varName) {
        placeHolderVarNames.add(varName);
    }


    /**
     * Resolve all ndarrays by updating the variables
     * for each array specified in the given map.
     * An {@link IllegalStateException} will be thrown
     * if not all arrays are specified for resolution.
     * @param arrays the arrays to resolve.
     */
    public void resolveVariablesWith(Map<String,INDArray> arrays) {
        for(val arrayEntry : arrays.entrySet()) {
            val varForName = getVariable(arrayEntry.getKey());
            if(varForName == null) {
                throw new ND4JIllegalStateException("No variable name found for " + arrayEntry.getKey());
            }

            if(placeHolderOriginalShapes.containsKey(arrayEntry.getKey())) {
                val originalShape = placeHolderOriginalShapes.get(arrayEntry.getKey());
                for(int i = 0; i < originalShape.length; i++) {
                    if(originalShape[i] != arrayEntry.getValue().shape()[i] && originalShape[i] >= 1) {
                        throw new ND4JIllegalStateException("Incompatible shape passed for variable. " + Arrays.toString(arrayEntry.getValue().shape()));
                    }
                }
            }
        }



        for(val entry : arrays.entrySet()) {
            if(!placeHolderVarNames.contains(entry.getKey())) {
                throw new ND4JIllegalStateException("Illegal variable " + entry.getKey() + " passed in. Variable found not to be a place holder variable");
            }

            updateShapeForVarName(entry.getKey(),entry.getValue().shape());
            associateArrayWithVariable(entry.getValue(),getVariable(entry.getKey()));
            updateArrayForVarName(entry.getKey(),entry.getValue());

        }

        //extra init after we know aray shape
     /*   for(val func : functionInstancesById.values()) {
            func.initWithArrays(arrays);
        }*/
/*

        //propagate variable names, sometimes shapes depend on  variables
        //that have place holders
        for(val func : placeHolderFunctions) {
            getFunctionById(func).outputVariables();
            val calcOutputShape = getFunctionById(func).calculateOutputShape();
            val outputs = getOutputVariablesForFunction(getFunctionById(func));
            for(int i = 0; i < calcOutputShape.size(); i++) {
                if(getShapeForVarName(outputs[i].getVarName()) == null)
                    putShapeForVarName(outputs[i].getVarName(),calcOutputShape.get(i));
                if(getArrForVarName(outputs[i].getVarName()) == null)
                    outputs[i].storeAndAllocateNewArray();
            }
        }
*/



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
     * @return true if all place holder variables are resolved.
     */
    public boolean allPlaceHolderVariablesResolved() {
        for(val vertexId : placeHolderVarNames) {
            val var = getVariable(vertexId);
            if(var.getArr() == null) {
                return false;
            }
        }

        return true;
    }

    /**
     * Add one or or more place holder variables
     * for the given vertex id.
     *
     * Note that if a vertex id in placeHolderVariables
     * isn't present in this samediff instance anyways,
     * an {@link ND4JIllegalStateException} is thrown
     *  @param varName the vertex id to add place holders for
     * @param placeHolderVariables the place holder variables
     */
    public void putPlaceHolderForVariable(String varName, String... placeHolderVariables) {
        for(val placeHolderVariable : placeHolderVariables) {
            if(!variableMap.containsKey(placeHolderVariable)) {
                throw new ND4JIllegalStateException("No variable found for " + placeHolderVariable);
            }
        }


        List<String[]> placeHolders = placeHolderMap.get(varName);
        if(placeHolders == null) {
            placeHolders = new ArrayList<>();
            placeHolderMap.put(varName,placeHolders);
        }

        placeHolders.add(placeHolderVariables);
    }


    /**
     * Returns true if the given vertex id
     * has any placeholder variables
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
     *
     * Consider using {@link #hasPlaceHolderVariables(String)}
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
     * @return
     */
    public Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> execWithPlaceHolder(Map<String,INDArray> inputs) {
        resolveVariablesWith(inputs);
        return exec();
    }

    /**
     * Get the {@link SDVariable}
     * associated with each function
     * based on the {@link DifferentialFunction#outputVariables()} ()}
     * @param functions the functions to get the variables for
     * @return the list of variables associated with the given {@link DifferentialFunction}
     */
    public List<SDVariable> getVariablesAssociatedWithFunctions(List<DifferentialFunction> functions) {
        List<SDVariable> ret = new ArrayList<>(functions.size());
        for(DifferentialFunction function : functions) {
            ret.addAll(Arrays.asList(function.outputVariables()));
        }

        return ret;
    }


    /**
     * Updates the variable name
     * property on the passed in variable,
     * the reference in samediff,
     * and returns the variable.
     *
     * Note that if null for the new variable is passed in,
     * it will just return the original input variable.
     *
     * @param varToUpdate the variable to update
     * @param newVarName the new variable name
     * @return the passed in variable
     */
    public SDVariable updateVariableNameAndReference(SDVariable varToUpdate,String newVarName) {
        if(newVarName == null || varToUpdate.getVarName().equals(newVarName)) {
            return varToUpdate;
        }

        if(varToUpdate == null) {
            throw new ND4JIllegalStateException("No variable found for updating!");
        }

        val oldVarName = varToUpdate.getVarName();
        varToUpdate.setVarName(newVarName);
        updateVariableName(oldVarName,newVarName);
        return varToUpdate;
    }

    /**
     * Creates and executes a list of operations
     * @return
     */
    public Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> exec() {
        if(!resolvedVariables)
            resolveVariablesWith(new LinkedHashMap<String, INDArray>());

        List<DifferentialFunction> ops = new ArrayList<>();

        Map<SDVariable,DifferentialFunction> opMap = new HashMap<>();
        val funcs = new ArrayList<DifferentialFunction>(functionInstancesById.values());
        boolean onBackward = false;
        for(int i = 0; i < funcs.size(); i++) {
            val opName = funcs.get(i).opName();
            if(!onBackward && opName.equals(new GradientBackwardsMarker().opName())) {
                onBackward = true;
            }

            if(opName.equals(new GradientBackwardsMarker().opName()))
                continue;

            DifferentialFunction differentialFunction = funcs.get(i);

            if(differentialFunction instanceof If) {
                If ifOp = (If) differentialFunction;
                if(!onBackward) {
                    ifOp.getPredicateExecution().exec();
                    //depending on the block add the proper graph body to this for persistence
                    //and possible later processing.
                    if(ifOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                        ifOp.getLoopBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(true);
                    }
                    else {
                        ifOp.getFalseBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(false);

                    }
                }
                else {
                    if(ifOp.getTrueBodyExecuted() != null) {
                        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = null;
                        List<SDVariable> variablesForFunctions =  null;
                        if(ifOp.getTrueBodyExecuted()) {
                            execBackwards = ifOp.getLoopBodyExecution().execBackwards();

                            variablesForFunctions = ifOp.getLoopBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        }
                        else {
                            execBackwards = ifOp.getFalseBodyExecution().execBackwards();
                            variablesForFunctions = ifOp.getFalseBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        }

                        /**
                         * Maps the variables from the child namespace body to
                         * the parent. This allows access to the underlying ndarray
                         * and returning a valid variable reference for autodiff.
                         */
                        for(SDVariable variable : variablesForFunctions) {
                            SDVariable proxyVar = var(variable);
                        }


                    }

                    else
                        throw new ND4JIllegalStateException("No body was run.");

                }


                ops.add(differentialFunction);

            }
            else if(differentialFunction instanceof While) {
                While whileOp = (While) differentialFunction;

                if(!onBackward) {
                    SameDiff execBody = whileOp.getLoopBodyExecution();
                    //depending on the block add the proper graph body to this for persistence
                    //and possible later processing.
                    //note that we need to update the graph predicate by running the execution
                    whileOp.getPredicateExecution().exec();
                    while(whileOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                        //run the body
                        execBody.exec();
                        //update the predicate
                        whileOp.getPredicateExecution().exec();
                        whileOp.incrementLoopCounter();

                    }

                    List<SDVariable> outputs = new ArrayList<>();
                    val outputFuncArgs =  new ArrayList<>(execBody.functionInstancesById.values()).get(execBody.functionInstancesById.values() .size() -1).outputVariables();
                    outputs.addAll(Arrays.asList(outputFuncArgs));

                    whileOp.setOutputVars(outputs.toArray(new SDVariable[outputs.size()]));
                    ops.add(differentialFunction);
                }

                else {
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
                    for(SDVariable variable : mapListPair.getFirst().keySet()) {
                        variable.getArr().muli(whileOp.getNumLooped());
                    }


                }



            }
            else if(differentialFunction instanceof CustomOp) {
                DynamicCustomOp customOp = (DynamicCustomOp) differentialFunction;
                customOp.populateInputsAndOutputsFromSameDiff();
                customOp.assertValidForExecution();
                Nd4j.getExecutioner().exec(customOp);
                ops.add(customOp);
            }

            else if(differentialFunction instanceof Op) {
                Op op = (Op) differentialFunction;
                if(differentialFunction.getDimensions() == null)
                    Nd4j.getExecutioner().exec(op);
                else if(op.isExecSpecial()) {
                    op.exec();
                }

                else {
                    int[] axes = differentialFunction.getDimensions();
                    if(differentialFunction instanceof Accumulation) {
                        Accumulation accumulation = (Accumulation) differentialFunction;
                        Nd4j.getExecutioner().exec(accumulation,axes);

                    }

                    else if(differentialFunction instanceof BroadcastOp) {
                        BroadcastOp broadcastOp = (BroadcastOp) differentialFunction;
                        Nd4j.getExecutioner().exec(broadcastOp,axes);
                    }
                    else if(differentialFunction instanceof GradientOp) {
                        Nd4j.getExecutioner().exec(op);
                    }
                    else if(differentialFunction instanceof IndexAccumulation) {
                        IndexAccumulation indexAccumulation = (IndexAccumulation) differentialFunction;
                        Nd4j.getExecutioner().exec(indexAccumulation,axes);

                    }
                }



                ops.add(differentialFunction);
            }


            //debug
            printFunction(differentialFunction);
        }

        return new Pair<>(opMap,ops);
    }


    public void printFunction(DifferentialFunction differentialFunction) {
        StringBuilder argShapes = new StringBuilder();
        for(val arg : differentialFunction.args()) {
            argShapes.append(" Variable " + arg.getVarName() +
                    " Shape for " + Arrays.toString(arg.getShape()));
        }

        for(val func : differentialFunction.outputVariables()) {
            argShapes.append("  Output variable " + func.getVarName() + " is " +
                    Arrays.toString(func.getShape()));
        }


        log.info("Executing op " + differentialFunction.opName());

        StringBuilder realShapes = new StringBuilder();
        for(val arg: differentialFunction.args()) {
            realShapes.append(" Input shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }

        for(val arg: differentialFunction.outputVariables()) {
            realShapes.append(" Output shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }


        log.info(realShapes.toString());
    }

    /**
     * Update the {@link INDArray}
     * ndarray for the given variable name
     * @param variableName the variable to update
     * @param arr the array to update with
     */
    public void updateVariable(String variableName,INDArray arr) {
        if(!variableNameToArr.containsKey(variableName))
            putArrayForVarName(variableName,arr);
        else
            updateArrayForVarName(variableName,arr);
    }







    protected int asFlatNode(String name,@NonNull SameDiff scope, @NonNull FlatBufferBuilder bufferBuilder) {
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
     *
     * @param varName
     * @return
     */
    public static Pair<String, Integer> parseVariable(@NonNull String varName) {
        if (!varName.contains(":")) {
            return Pair.pairOf(varName, 0);
        } else {
            val split = varName.split(":");
            val index = Integer.valueOf(split[split.length-1]);
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

    protected int asFlatNode(@NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder,List<SDVariable> variables, Map<String, Integer> reverseMap) {
        val hash = getOpNum(node.opName(), node.opType());
        //log.info("Exporting node: [{}:<{}> ; OpType: {}; Hash/opNum: {}]", node.opName(), node.tensorflowName(), node.opType(), hash);

        float[] extras = node.getExtraArgs() != null ? new float[node.getExtraArgs().length] : new float[0];
        for (int e = 0; e < extras.length; e++) {
            extras[e] = ((Number) node.getExtraArgs()[e]).floatValue();
        }


        int[] extraBits = null;
        if(node.opType() == Op.Type.CUSTOM) {
            DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) node;
            extraBits = dynamicCustomOp.iArgs();
        }
        else
            extraBits = new int[]{};

        val inPaired = new ArrayList<Integer>();

        val outputVertexId = node.outputVariables();
        val outputIds = new int[outputVertexId.length];
        for(int i = 0; i < outputIds.length; i++) {
            outputIds[i] = variables.indexOf(outputVertexId[i]);
        }


        val inputs = node.args();
        for(val input : inputs) {
            for(int i = 0; i < outputVertexId.length; i++) {
                val pair = parseVariable(input.getVarName());
                if (!reverseMap.containsKey(pair.getFirst()))
                    throw new ND4JIllegalStateException("Unknown variable used in input: [" +  pair.getFirst() + "]");

                int nodeId = reverseMap.get(pair.getFirst());
                int outputIndex = pair.getSecond();

                inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outputIndex));
            }
        }

        log.info("Own Name: {}", node.getOwnName());
        int ownId = reverseMap.size() + 1;
        reverseMap.put(node.getOwnName(), ownId);

        // TODO: Adam, just put your props here, instead of empty list, and they will be saved
        List<FunctionProperties> props = new ArrayList<>();
        int properties = FunctionProperties.asFlatProperties(bufferBuilder, props);

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder,outputIds);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, node.getDimensions() != null ? node.getDimensions() : new int[]{});
        int fname = bufferBuilder.createString(
                outputVertexId == null  ||
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
                node.opType() == Op.Type.SCALAR && node.getScalarValue() != null ?  node.getScalarValue().floatValue() : 0.0f, 0, scopeName);

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

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();

        // first of all we build VariableSpace dump
        List<SDVariable> variableList = new ArrayList<>(variables());
        val reverseMap = new LinkedHashMap<String, Integer>();

        int idx = 0;
        for (val variable: variables()) {
            log.info("Exporting variable: [{}]", variable.getVarName());
            if(variable.getArr() == null || variable.getShape() == null)
                continue;

            val pair = parseVariable(variable.getVarName());
            reverseMap.put(pair.getFirst(), ++idx);
            log.info("Adding [{}] as [{}]", pair.getFirst(), idx);

            val arr = variable.getArr();

            int name = bufferBuilder.createString(variable.getVarName());
            int array = arr.toFlatArray(bufferBuilder);
            int id = IntPair.createIntPair(bufferBuilder, idx, 0);


            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
            flatVariables.add(flatVariable);
        }

        //add functions
        for(val func : functionInstancesById.values()) {
            flatNodes.add(asFlatNode(func,bufferBuilder,variableList, reverseMap));
        }

        // we're dumping scopes now
        for (val scope: sameDiffFunctionInstances.entrySet()) {
            flatNodes.add(asFlatNode(scope.getKey(),scope.getValue(), bufferBuilder));
            val currVarList = new ArrayList<SDVariable>(scope.getValue().variables());
            // converting all ops from node
            for (val node: scope.getValue().variables()) {
                val arr = node.getArr();

                int name = bufferBuilder.createString(node.getVarName());
                int array = arr.toFlatArray(bufferBuilder);
                int id = IntPair.createIntPair(bufferBuilder, ++idx, 0);

                val pair = parseVariable(node.getVarName());
                reverseMap.put(pair.getFirst(), idx);

                log.info("Adding [{}] as [{}]", pair.getFirst(), idx);

                int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
                flatVariables.add(flatVariable);
            }

            //add functions
            for(val func : scope.getValue().functionInstancesById.values()) {
                flatNodes.add(asFlatNode(func,bufferBuilder,currVarList, reverseMap));
            }

        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

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
        for (int e = 0; e <graph.nodesLength(); e++) {
            val node = graph.nodes(e);

            log.info("{}:<{}>", node.id(), node.name());
            sb.append(node.id())
                    .append(":<").append(node.name()).append("> ").append(SameDiff.getTypeFromByte(node.opType()));

            if (SameDiff.getTypeFromByte(node.opType()) != Op.Type.CUSTOM)
                sb.append(": ").append(node.opNum());
            else {
                val keys = map.keySet();
                String opName = null;
                for (val k: keys) {
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

            sb.append("}\n");
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
            case FLOAT: return DataType.FLOAT;
            case DOUBLE: return DataType.DOUBLE;
            case HALF: return DataType.HALF;
            case INT: return DataType.INT32;
            case LONG: return DataType.INT64;
            default: throw new ND4JIllegalStateException("Unknown or unsupported DataType used: ["+ type +"]");
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
        } else if (type == Op.Type.IF || type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.CUSTOM) {
            val name2 = Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase());
            if(name2 == null)
                return 0;
            return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();

        }
        else
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
            case LOOP:
                return OpType.LOGIC;
            case RETURN:
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

}