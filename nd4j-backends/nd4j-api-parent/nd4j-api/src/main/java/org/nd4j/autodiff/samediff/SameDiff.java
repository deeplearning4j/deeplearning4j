package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import com.rits.cloning.Cloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.EdgeId;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpExecOrder;
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
    private SDGraph graph;

    //
    private Map<int[],DifferentialFunction> incomingArgs;
    private Map<int[],DifferentialFunction> outgoingArgs;
    private Map<String,int[]> incomingArgsReverse;
    private Map<String,int[]> ougoingArgsReverse;
    //index of edges
    private Table<IntArrayKeyMap.IntArray,IntArrayKeyMap.IntArray,DifferentialFunction> fromToTable;

    private DifferentialFunctionFactory functionFactory;
    private Map<String,SDVariable> variableMap;
    private Map<Integer,SDVariable> vertexIdToVariable;
    private Map<Integer,int[]> vertexIdToShape;
    //gradient information
    private Map<Integer,SDVariable> gradients;
    private Map<Integer,SDVariable> forwardVarForGrad;

    private Map<Integer,INDArray> vertexIdToArr;

    private Map<Integer,List<int[]>> placeHolderMap;
    private Map<Integer,int[]> placeHolderOriginalShapes;
    private Set<Integer> placeHolderVertexIds;
    private IdentityHashMap<INDArray,SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;

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

    private Map<OpExecAction,ForwardBackwardState> forwardBackwardStates;

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
     * Clears debugging state
     * and disables debug mode.
     */
    public SameDiff disableDebugging() {
        forwardBackwardStates.clear();
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

        //map new vertex ids and create new vertices
        for(int i = 0; i < graph().numVertices(); i++) {
            int nextVertexId = sameDiff.graph.nextVertexId();
            SDVariable clone = cloner.deepClone(graph.getVertex(i + 1).getValue());
            NDArrayVertex info = new NDArrayVertex(
                    sameDiff,
                    nextVertexId,
                    graph.getVertex(i + 1).depth(),
                    clone);
            thisVertexIdToNew.put(graph.getVertex(i + 1).vertexID(),nextVertexId);
            sameDiff.graph().addVertex(info);

        }

        for(Set<Edge<DifferentialFunction>> edgeList : graph().getEdges().values()) {
            for(Edge<DifferentialFunction> edge : edgeList) {
                EdgeId newEdge = new EdgeId(
                        new int[]{thisVertexIdToNew.get(edge.getFrom()[0])},
                        new int[]{thisVertexIdToNew.get(edge.getTo()[0])},
                        cloner.deepCloneDontCloneInstances(edge.getValue()),true);

                sameDiff.graph().addEdge(newEdge);
            }
        }


        for(val edgeList : graph().getIncomingEdges().values()) {
            for(val edge : edgeList) {
                val newFrom = new int[]{thisVertexIdToNew.get(edge.getFrom()[0])};
                val newTo =  new int[]{thisVertexIdToNew.get(edge.getTo()[0])};
                EdgeId newEdge = new EdgeId(
                        newFrom,
                        newTo,
                        cloner.deepCloneDontCloneInstances(edge.getValue()),true);
                sameDiff.graph().addEdge(newEdge);
                sameDiff.addArgsFor(newFrom,newEdge.getValue());
                sameDiff.addOutgoingFor(newTo,newEdge.getValue());

            }
        }



        List<SDVariable> variables = variables();

        //copy over variables
        for(SDVariable variable : variables) {
            SDVariable deepClone = cloner.deepCloneDontCloneInstances(
                    variable,
                    variable.getArr(),
                    variable.getSameDiff(),
                    variable.getShape());
            Preconditions.checkState(thisVertexIdToNew.containsKey(variable.getVertexId()),variable.getVertexId() + " not found in mapped vertices!");
            int newVertexMap = thisVertexIdToNew.get(variable.getVertexId());

            deepClone.setDepth(variable.depth());
            deepClone.setVertexId(newVertexMap);
            deepClone.setSameDiff(sameDiff);
            sameDiff.addVariable(deepClone);

            if(variable.getArr() != null) {
                sameDiff.associateArrayWithVariable(variable.getArr(),deepClone);
            }


        }

        val newFunctions = new LinkedHashMap<String,DifferentialFunction>();
        for(DifferentialFunction function : this.fromToTable.values())  {
            DifferentialFunction clone = cloner.deepCloneDontCloneInstances(
                    function,
                    function.getSameDiff());
            clone.setSameDiff(sameDiff);
            newFunctions.put(function.getInstanceId(),clone);
            for(DifferentialFunction clonedArgs : clone.args()) {
                clonedArgs.setSameDiff(sameDiff);
            }


        }

        for(val entry : this.fromToTable.rowMap().entrySet()) {
            val oldFrom = entry.getKey().getBackingArray();
            val oldTo = entry.getValue().keySet().iterator().next().getBackingArray();

            val currFunc = entry.getValue().values().iterator().next();
            val reMappedFrom = new IntArrayKeyMap.IntArray(new int[]{thisVertexIdToNew.get(oldFrom[0])});
            val reMappedTo =  new IntArrayKeyMap.IntArray(new int[]{thisVertexIdToNew.get(oldTo[0])});
            currFunc.setSameDiff(sameDiff);
            sameDiff.fromToTable.put(reMappedFrom,reMappedTo,newFunctions.get(currFunc.getInstanceId()));
            sameDiff.functionInstancesById.put(currFunc.getInstanceId(),currFunc);
        }

        sameDiff.reverseArrayLookup.putAll(reverseArrayLookup);
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
     * Returns the inputs for the given function
     * @param function the function to get the
     *                 inputs for
     * @return the input ids for a given function
     */
    public int[] getInputsForFunction(DifferentialFunction function) {
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
    public int[] getOutputsForFunction(DifferentialFunction function) {
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
            vars[i] = getVariableForVertexId(inputs[i]);
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
            vars[i] = getVariableForVertexId(inputs[i]);
        }

        return vars;
    }


    /**
     * Returns the arguments for a given output {@link DifferentialFunction}
     * @param output the output to get the arguments for
     * @return the arguments for the target function
     */
    public SDVariable[] getArgsFor(SDVariable output) {
        Set<SDVariable> ret = new LinkedHashSet<>();
        val from = graph().getFromFor(output.inputVertexIds());
        for (int i = 0; i < from.length; i++) {
            val currFunc = getVariableForVertexId(from[i]);
            if (currFunc != null && !ret.contains(currFunc)) {
                ret.add(currFunc);
            } else if (getVariableForVertexId(from[i]) != null) {
                ret.add(getVariableForVertexId(from[i]));
            } else
                throw new ND4JIllegalStateException("No function or variable found for " + Arrays.toString(new int[]{from[i]}));
        }


        return ret.toArray(new SDVariable[ret.size()]);
    }


    /**
     * Get the variable for the given vertex id.
     * WARNING: This function also has side effects.
     * {@link SDVariable} instances will be created
     * if a valid {@link DifferentialFunction} is present.
     * Otherwise an {@link ND4JIllegalStateException} is thrown
     * for no vertex id.
     * @param vertexId the vertex id (usually a 1 length array but can be multiple)
     * @return the variable for this vertex
     */
    public SDVariable getVariableForVertexId(int vertexId) {
        if(!vertexIdToVariable.containsKey(vertexId)) {
            throw new IllegalArgumentException("No vertex id of " + vertexId + " found!");
        }

        return vertexIdToVariable.get(vertexId);
    }


    /**
     * Update the ndarray for the given vertex id.
     * @throws {@link ND4JIllegalStateException} when the array does not exist.
     * @param vertexId
     * @param arr
     */
    public void updateArrayForVertexId(int vertexId,INDArray arr) {
        if(!vertexIdToArr.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Array for " + vertexId + " does not exist. Please use putArrayForVertexId instead.");
        }
        vertexIdToArr.put(vertexId,arr);
        reverseArrayLookup.put(arr,getVariableForVertexId(vertexId));
    }

    /**
     * Adds an ndarray for a given vertex id.
     * Use {@link #updateArrayForVertexId(int, INDArray)}
     * if the array already exists.
     *
     * @param vertexId the vertex id to add
     * @param arr the array to add
     *
     * @throws {@link ND4JIllegalStateException} when the array already exists.
     */
    public void putArrayForVertexId(int vertexId,INDArray arr) {
        if(vertexIdToArr.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Array for " + vertexId + " already exists!");
        }

        vertexIdToArr.put(vertexId,arr);
    }

    /**
     * Get the shape for the given vertex id.
     * Note that if an array is defined, it will use that shape instead.
     *
     * A shape *and* an array should not be defined at the same time.
     * This wastes memory. The internal map used for tracking shapes for particular
     * vertex ids should also delete redundant shapes stored to avoid redundant sources of information.
     * @param vertexId the vertex id to get the shape for
     * @return the shape for the given vertex if if any.
     */
    public int[] getShapeForVertexId(int vertexId) {
        //first check that the shape doesn't already exists.
        //if it does, remove it
        if(vertexIdToArr.containsKey(vertexId) && vertexIdToShape.containsKey(vertexId)) {
            vertexIdToShape.remove(vertexId);
        }

        if(vertexIdToArr.containsKey(vertexId)) {
            return vertexIdToArr.get(vertexId).shape();
        }



        return vertexIdToShape.get(vertexId);
    }



    /**
     * Update a vertex id with the given shape.
     * Note that you should use {@link #putShapeForVertexId(int, int[])}
     * if you want to add a new shape.
     * Update is meant to be an in place replacement
     * of the shape for the vertex id *only*.
     * @param vertexId the vertex id to associate
     * @param shape the shape to associate with
     */
    public void updateShapeForVertexId(int vertexId,int[] shape) {
        if(shape == null || shape.length < 2) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }


        if(shape == null) {
            throw new ND4JIllegalStateException("Null shapes not allowed!");
        }

        if(vertexIdToArr.containsKey(vertexId) && !Arrays.equals(vertexIdToArr.get(vertexId).shape(),shape)) {
            throw new ND4JIllegalStateException("Already found an existing array!");
        }

        vertexIdToShape.put(vertexId,shape);
    }


    /**
     * Associate a vertex id with the given shape.
     * @param vertexId the vertex id to associate
     * @param shape the shape to assciate with
     */
    public void putShapeForVertexId(int vertexId,int[] shape) {
        if(shape == null || shape.length < 2) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }

        if(vertexIdToShape.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Shape for " + vertexId + " already exists!");
        }

        vertexIdToShape.put(vertexId,shape);
    }




    /**
     * Returns true if the given vertex id
     * and shape already exist.
     * @param vertexId the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean shapeAlreadyExistsForVertexId(int vertexId) {
        return vertexIdToShape.containsKey(vertexId) || arrayAlreadyExistsForVertexId(vertexId);
    }



    /**
     * Returns true if the given vertex id
     * and {@link INDArray} already exist.
     * @param vertexId the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean arrayAlreadyExistsForVertexId(int vertexId) {
        return vertexIdToArr.containsKey(vertexId);
    }

    /**
     * Get an {@link INDArray}
     * for a given vertex id
     * @param vertexId
     * @return
     */
    public INDArray getArrForVertexId(int vertexId) {
        return vertexIdToArr.get(vertexId);
    }

    /**
     * Associate the array with the given variable.
     * @param arr the array to get the variable for
     * @param variable the variable to associate
     */
    public void associateArrayWithVariable(INDArray arr, SDVariable variable) {
        reverseArrayLookup.put(arr,variable);
        vertexIdToArr.put(variable.getVertexId(),arr);
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
        graph = new SDGraph();
        graph.setSameDiff(this);
        functionFactory = new DifferentialFunctionFactory(this);
        variableMap = new LinkedHashMap<>();
        sameDiffFunctionDefinitionMap = new LinkedHashMap<>();
        sameDiffFunctionInstances = new LinkedHashMap<>();
        vertexIdToVariable = new LinkedHashMap<>();
        gradients = new LinkedHashMap<>();
        forwardVarForGrad = new LinkedHashMap<>();
        forwardBackwardStates = new HashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        reverseArrayLookup = new IdentityHashMap<>();
        vertexIdToArr = new LinkedHashMap<>();
        vertexIdToShape = new LinkedHashMap<>();
        placeHolderMap = new LinkedHashMap<>();
        placeHolderVertexIds = new LinkedHashSet<>();
        placeHolderOriginalShapes = new LinkedHashMap<>();

        incomingArgs = new IntArrayKeyMap<>();
        outgoingArgs = new IntArrayKeyMap<>();
        incomingArgsReverse = new HashMap<>();
        ougoingArgsReverse = new HashMap<>();
        this.functionInstancesById = new HashMap<>();
        fromToTable = HashBasedTable.create();

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
    public void addOutgoingFor(SDVariable[] variables,DifferentialFunction function) {
        int[] vertexIds = new int[variables.length];
        for(int i = 0; i < vertexIds.length; i++) {
            vertexIds[i] = variables[i].getVertexId();
        }

        addOutgoingFor(vertexIds,function);
    }



    /**
     * Adds outgoing arguments to the graph.
     * Also checks for input arguments
     * and updates the graph adding an appropriate edge
     * when the full graph is declared.
     * @param vertexIds
     * @param function
     */
    public void addOutgoingFor(int[] vertexIds,DifferentialFunction function) {
        if(outgoingArgs.containsKey(vertexIds)) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for "  + function);
        }

        if(function.getInstanceId() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        if(ougoingArgsReverse.containsKey(function.getInstanceId())) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for " + function);
        }


        ougoingArgsReverse.put(function.getInstanceId(),vertexIds);
        outgoingArgs.put(vertexIds,function);

        val incomingArgs = incomingArgsReverse.get(function.getInstanceId());
        graph().addEdge(incomingArgs,vertexIds,function,true);
        fromToTable.put(new IntArrayKeyMap.IntArray(incomingArgs),new IntArrayKeyMap.IntArray(vertexIds),function);

    }



    /**
     * Adds incoming args to the graph
     * @param variables
     * @param function
     */
    public void addArgsFor(SDVariable[] variables,DifferentialFunction function) {
        int[] vertexIds = new int[variables.length];
        for(int i = 0; i < vertexIds.length; i++) {
            vertexIds[i] = variables[i].getVertexId();
        }

        addArgsFor(vertexIds,function);
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

    /**
     * Adds incoming args to the graph
     * @param vertexIds
     * @param function
     */
    public void addArgsFor(int[] vertexIds,DifferentialFunction function) {

        if(incomingArgs.containsKey(vertexIds)) {
            throw new ND4JIllegalStateException("Incoming arguments already declared for "  + function);
        }

        if(function.getInstanceId() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");


        if(incomingArgsReverse.containsKey(function.getInstanceId())) {
            throw new ND4JIllegalStateException("Attempting to add duplicate function for function id " + function.getInstanceId());
        }


        incomingArgsReverse.put(function.getInstanceId(),vertexIds);
        incomingArgs.put(vertexIds,function);
    }

    /**
     * Get the function matching
     * the specified inputs and outputs
     * @param inputArgs the inputs
     * @param outputArgs the outputs
     * @return the function for the given inputs
     * and outputs or null
     */
    public DifferentialFunction getFunction(int[] inputArgs,int[] outputArgs) {
        return fromToTable.get(new IntArrayKeyMap.IntArray(inputArgs),new IntArrayKeyMap.IntArray(outputArgs));
    }

    public DifferentialFunction[] functions() {
        val values  = fromToTable.values();
        return values.toArray(new DifferentialFunction[values.size()]);
    }


    /**
     * The same diff graph
     * @return
     */
    public SDGraph graph() {
        if(graph.getGraphApply() != null) {
            return graph.getGraphApply();
        }

        return graph;
    }



    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (graph != null ? graph.hashCode() : 0);
        result = 31 * result + (variableMap != null ? variableMap.hashCode() : 0);
        return result;
    }




    /**
     *
     * @param originalSameDiff
     * @param graph
     * @return
     */
    public static SameDiff create(SameDiff originalSameDiff, SDGraph graph) {
        SDGraph clone = new SDGraph(graph);
        SameDiff ret = SameDiff.builder()
                .variableMap(originalSameDiff.variableMap)
                .sameDiffFunctionInstances(originalSameDiff.sameDiffFunctionInstances)
                .graph(clone)
                .build();
        //ensuring proper sameDiff reference
        clone.setSameDiff(ret);
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

        if (graph != null ? !graph.equals(sameDiff.graph) : sameDiff.graph != null) return false;
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
            ret[i] = getArrForVertexId(opExecAction.get(i).outputVariables()[0].getVertexId());
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
        return var(name,shape,1.0);

    }


    /**
     * Variable initialization
     * with 0.0
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @return the created variable
     */
    public SDVariable zero(String name, int[] shape) {
        return var(name,shape,0.0);

    }



    /**
     * Variable initialization
     * with a  constant
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @param constant the value to be initialized with
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape,double constant) {
        return var(name,shape,new ConstantInitScheme('f',constant),0);

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
        return var(name,shape,weightInitScheme,0);

    }


    /**
     * Variable initialization
     * with a specified {@link WeightInitScheme}
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @param weightInitScheme the weight init scheme
     * @param vertexId  the vertex id to use for the variable
     * @param depth the depth in the graph (default 0)
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape, WeightInitScheme weightInitScheme,int vertexId,int depth) {
        if(variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            return variableMap.get(name);


        if(name == null || name.length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if(workspace == null)
            initWorkspace();


        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .vertexId(vertexId)
                .shape(shape).weightInitScheme(weightInitScheme)
                .varName(name)
                .build();

        if(graph().getVertex(vertexId) == null) {
            NDArrayVertex ndArrayVertex = new NDArrayVertex(this, vertexId, depth, ret);
            graph.addVertex(ndArrayVertex);
        }

        addVariable(ret);
        variableMap.put(name,ret);
        return ret;

    }


    /**
     * Variable initialization
     * with a specified {@link WeightInitScheme}
     * , depth, and vertex id of{@link SDGraph#nextVertexId}
     * @param name the opName of the variable
     * @param shape the shape of the array to be created
     * @param weightInitScheme the weight init scheme
     * @param depth the depth in the graph (default 0)
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape, WeightInitScheme weightInitScheme,int depth) {
        return var(name, shape, weightInitScheme, graph.nextVertexId(),depth);

    }



    /**
     *
     *
     * @param name
     * @param shape
     * @return
     */
    public SDVariable var(String name, int[] shape,int depth) {
        return var(name,shape,new ZeroInitScheme('f'),depth);
    }


    /**
     * Creates a {@link SDVariable}
     * ,{@link NDArrayVertex}
     * with the given shape
     * and a depth of 0.
     *
     * @param name the opName of the variable
     * @param shape the shape of the variable
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape) {
        return var(name,shape,0);

    }


    /**
     * Initialize a {@link SDVariable}
     * reference tying this variable to this
     * samediff instance.
     *
     * {@link NDArraySupplierInitScheme} is used
     * to ensure that if the array is allocated anywhere
     * in any setting, the same array reference will be preserved
     * while allowing a separate {@link NDArrayVertex}
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

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,graph.nextVertexId(), 0,arr);
        graph.addVertex(ndArrayVertex);
        final SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .vertexId(ndArrayVertex.getIdx())
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
        addVariable(ret);
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

        arr = arr.migrate();
        int vertexIdx = this.graph.nextVertexId();
        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .vertexId(vertexIdx)
                .shape(arr.shape())
                .varName(name)
                .build();
        associateArrayWithVariable(arr,ret);
        if(ArrayUtil.prod(arr.shape()) == 1)
            ret.setScalarValue(arr.getDouble(0));

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,vertexIdx, 0,ret);
        graph.addVertex(ndArrayVertex);

        addVariable(ret);
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
     * @param vertexId the vertex id
     * @return the gradient for this variable or null
     */
    public SDVariable getGradForVertexId(int...vertexId) {
        return gradients.get(vertexId[0]);
    }


    /**
     * Assign a vertex id
     * to a gradient
     * @param vertexId the vertex id
     *                 to assign
     * @param variable the variable
     */
    public void setGradientForVertexId(int vertexId, SDVariable variable) {
        gradients.put(vertexId,variable);
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
     *
     * @param vertexId
     * @param forwardVariable
     */
    public void setForwardVariableForVertexId(int vertexId,SDVariable forwardVariable) {
        forwardVarForGrad.put(vertexId,forwardVariable);
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
        return getFunction("grad").getGradForVertexId(var.getVertexId());
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

        val outputVertexId = conv2D.outputVertexIds()[0];
        updateVariableName(outputVertexId,generateVariableName(conv2D.opName(),false,inputs));
        return getVariableForVertexId(outputVertexId);
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

        updateVariableName(conv3D.outputVertexIds()[0],generateVariableName(conv3D.opName(),false,inputs));
        return getVariableForVertexId(conv3D.outputVertexIds()[0]);
    }


    public String createName(SDVariable...inputs) {
        StringBuilder stringBuilder = new StringBuilder();
        for(int i = 0; i < inputs.length; i++) {
            stringBuilder.append(inputs[i].getVarName());
            if(i < inputs.length - 1)
                stringBuilder.append(",");
        }

        return stringBuilder.toString();
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
        return gte(generateVariableName("gte",false,iX),iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, double iy) {
        return lte(generateVariableName("lte",false,iX),iX,iy);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, double iy) {
        return lt(generateVariableName("gt",false,iX),iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, double iy) {
        return lt(generateVariableName("lt",false,iX),iX,iy);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, double iy) {
        return neq(generateVariableName("neq",false,iX),iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, double iy) {
        return eq(generateVariableName("eq",false,iX),iX,iy);
    }







    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(SDVariable iX, SDVariable iy) {
        return gte(generateVariableName("gte",false,iX,iy),iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(SDVariable iX, SDVariable iy) {
        return lte(generateVariableName("lte",false,iX,iy),iX,iy);

    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(SDVariable iX, SDVariable iy) {
        return lt(generateVariableName("gt",false,iX,iy),iX,iy);

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(SDVariable iX, SDVariable iy) {
        return lt(generateVariableName("lt",false,iX,iy),iX,iy);

    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        return neq(generateVariableName("neq",false,iX,iy),iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(SDVariable iX, SDVariable iy) {
        return eq(generateVariableName("eq",false,iX,iy),iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(SDVariable iX, SDVariable iy) {
        return or(generateVariableName("or",false,iX,iy),iX,iy);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(SDVariable iX) {
        return neg(generateVariableName("neg",false,iX),iX);
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(SDVariable iX) {
        return cos(generateVariableName("cos",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(SDVariable iX) {
        return sin(generateVariableName("sin",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(SDVariable iX) {
        return tan(generateVariableName("tan",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(SDVariable iX) {
        return acos(generateVariableName("acos",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(SDVariable iX) {
        return asin(generateVariableName("asin",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(SDVariable iX) {
        return atan(generateVariableName("atan",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(SDVariable iX) {
        return cosh(generateVariableName("cosh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(SDVariable iX) {
        return sinh(generateVariableName("sinh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(SDVariable iX) {
        return tanh(generateVariableName("tanh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(SDVariable iX) {
        return acosh(generateVariableName("acosh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(SDVariable iX) {
        return asin(generateVariableName("asin",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(SDVariable iX) {
        return atanh(generateVariableName("atanh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(SDVariable iX) {
        return exp(generateVariableName("exp",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(SDVariable iX) {
        return log(generateVariableName("log",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(SDVariable iX,double value) {
        return pow(generateVariableName("pow",false,iX),iX,value);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(SDVariable iX) {
        return sqrt(generateVariableName("sqrt",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(SDVariable iX) {
        return square(generateVariableName("square",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(SDVariable iX) {
        return floor(generateVariableName("floor",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(SDVariable iX,double cutoff) {
        return relu(generateVariableName("relu",false,iX),iX,cutoff);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(SDVariable iX) {
        return softmax(generateVariableName("softmax",false,iX),iX);
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
        return hardTanh(generateVariableName("hardTanh",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(SDVariable iX) {
        return hardTanhDerivative(generateVariableName("hardTanhDerivative",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(SDVariable iX) {
        return sigmoid(generateVariableName("sigmoid",false,iX),iX);
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(SDVariable iX,SDVariable wrt) {
        return sigmoidDerivative(generateVariableName("sigmoidDerivative",false,iX),iX,wrt);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(SDVariable iX) {
        return sign(generateVariableName("sign",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(SDVariable iX) {
        return softsign(generateVariableName("softsign",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(SDVariable iX) {
        return softsignDerivative(generateVariableName("softsignDerivative",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(SDVariable iX) {
        return softplus(generateVariableName("softplus",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(SDVariable iX) {
        return elu(generateVariableName("elu",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(SDVariable iX) {
        return eluDerivative(generateVariableName("eluDerivative",false,iX),iX);
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return leakyRelu(generateVariableName("leakyRelu",false,iX),iX,cutoff);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable mean(SDVariable iX) {
        return mean(generateVariableName("mean",false,iX),iX);
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
        return standardDeviation(generateVariableName("std",false,iX),iX,biasCorrected,dimensions);
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
        return variance(generateVariableName("variance",false,iX),iX,biasCorrected,dimensions);
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(SDVariable iX,
                          int...dimensions) {
        return sum(generateVariableName("sum",false,iX),iX,dimensions);
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(SDVariable iX,
                           int...dimensions) {
        return prod(generateVariableName("prod",false,iX),iX,dimensions);
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(SDVariable iX, int...dimensions) {
        return max(generateVariableName("max",false,iX),iX,dimensions);

    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(SDVariable iX,
                          int...dimensions) {
        return min(generateVariableName("min",false,iX),iX,dimensions);
    }


    /**
     *
     * @param iX
     * @param shape
     * @return
     */
    public SDVariable reshape(SDVariable iX,
                              int...shape) {
        return reshape(generateVariableName("reshape",false,iX),iX,shape);
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(SDVariable iX) {
        return transpose(generateVariableName("transpose",false,iX),iX);
    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(SDVariable x, int axis) {
        return rollAxis(generateVariableName("rollAxis",false,x),x,axis);
    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(SDVariable x, SDVariable y) {
        return mmul(generateVariableName("mmul",false,x,y),x,y);
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
        return tensorMmul(generateVariableName("tensorMmul",false,x,y),x,y,dimensions);
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
        return lossCosineSimilarity(generateVariableName("lossCosineSimilarity",false,iX),iX,i_y,dimensions);
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
        return lossMSLE(generateVariableName("lossMLE",false,iX),iX,i_y,dimensions);

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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.neq(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.eq(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX,double iy) {
        SDVariable result = functionFactory.gte(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX,double iy) {
        SDVariable result = functionFactory.lte(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.gt(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX,double iy) {
        SDVariable result = functionFactory.lt(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }





    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.neq(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.eq(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gte(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lte(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gt(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lt(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(String name,SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.or(iX,iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(String name,SDVariable iX) {
        SDVariable result = functionFactory.neg(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(String name,SDVariable iX) {
        SDVariable result = functionFactory.cos(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(String name,SDVariable iX) {
        SDVariable result = functionFactory.sin(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(String name,SDVariable iX) {
        SDVariable result = functionFactory.tan(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(String name,SDVariable iX) {
        SDVariable result = functionFactory.acos(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(String name,SDVariable iX) {
        SDVariable result = functionFactory.asin(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(String name,SDVariable iX) {
        SDVariable result = functionFactory.atan(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(String name,SDVariable iX) {
        SDVariable result = functionFactory.cosh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(String name,SDVariable iX) {
        SDVariable result = functionFactory.sinh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(String name,SDVariable iX) {
        SDVariable
                result = functionFactory.tanh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(String name,SDVariable iX) {
        SDVariable result = functionFactory.acosh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(String name,SDVariable iX) {
        SDVariable result = functionFactory.asinh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(String name,SDVariable iX) {
        SDVariable result = functionFactory.atanh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(String name,SDVariable iX) {
        SDVariable result = functionFactory.exp(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(String name,SDVariable iX) {
        SDVariable result = functionFactory.log(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(String name,SDVariable iX,double value) {
        SDVariable result = functionFactory.pow(iX,value);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(String name,SDVariable iX) {
        SDVariable result = functionFactory.sqrt(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(String name,SDVariable iX) {
        SDVariable result = functionFactory.square(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(String name,SDVariable iX) {
        SDVariable result = functionFactory.floor(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(String name,SDVariable iX,double cutoff) {
        SDVariable result = functionFactory.relu(iX,cutoff);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(String name,SDVariable iX) {
        SDVariable result = functionFactory.softmax(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmaxDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable result = functionFactory.softmaxDerivative(iX,wrt);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanh(String name,SDVariable iX) {
        SDVariable result = functionFactory.hardTanh(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.hardTanhDerivative(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(String name,SDVariable iX) {
        SDVariable result = functionFactory.sigmoid(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable result = functionFactory
                .sigmoidDerivative(iX,wrt);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(String name,SDVariable iX) {
        SDVariable result = functionFactory
                .sign(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(String name,SDVariable iX) {
        SDVariable result = functionFactory.softsign(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.softsignDerivative(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(String name,SDVariable iX) {
        SDVariable result = functionFactory.softplus(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(String name,SDVariable iX) {
        SDVariable result = functionFactory.elu(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(String name,SDVariable iX) {
        SDVariable result = functionFactory.eluDerivative(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(String name,SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.leakyRelu(iX,cutoff);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable mean(String name,SDVariable iX) {
        SDVariable result = functionFactory.mean(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(String name,SDVariable iX, int...dimensions) {
        SDVariable result = functionFactory.max(iX,dimensions);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(String name,SDVariable iX) {
        SDVariable result = functionFactory.transpose(iX);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(String name,SDVariable x, int axis) {
        SDVariable result = functionFactory.rollAxis(x,axis);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(String name,SDVariable x, SDVariable y) {
        SDVariable result = functionFactory.mmul(x, y);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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

        int[] shape = ArrayUtil.getTensorMmulShape(x.getShape(), y.getShape(), dimensions);
        SDVariable result = functionFactory.tensorMmul(x,y, dimensions);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(cosim.getVertexId(),name);
        return getVariableForVertexId(cosim.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
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
        vertexIdToVariable.put(variable.getVertexId(),variable);
        variableMap.put(variable.getVarName(),variable);

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
                sb.append(variable.getVarName());
                if (grad) {
                    sb.append("-grad");
                }

                sb.append("-");
                if ((variable != null)) {
                    sb.append(variable.getVertexId());
                    sb.append("-");
                }

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
     *
     * @param opExecAction
     * @return
     */
    public DifferentialFunction createOp(OpExecAction opExecAction) {
        DifferentialFunction differentialFunction = getFunction(opExecAction.getInputsIds(),opExecAction.getOutputId());
        if(differentialFunction instanceof Op) {
            if (differentialFunction instanceof ScalarOp) {
                ScalarOp scalarOp = (ScalarOp) differentialFunction;
                scalarOp.setScalar(differentialFunction.getScalarValue());
            }

            Op op = (Op) differentialFunction;
            differentialFunction.fillInArrays();
            val argZero = ArrayUtil.prod(getShapeForVertexId(differentialFunction.outputVariables()[0].getVertexId()));
            op.setN(argZero);

        }
        else if(differentialFunction instanceof DynamicCustomOp) {
            DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) differentialFunction;

        }
        //if and while are special
        if(differentialFunction instanceof  If || differentialFunction instanceof While) {
            return differentialFunction;
        }

        Preconditions.checkNotNull(differentialFunction,"Unable to return null function");

        return differentialFunction;
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
     * @return
     */
    public INDArray execAndEndResult() {
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

        val newMap = new LinkedHashMap<Integer, INDArray>();
        val keySet = inputs.keySet();

        for (val key: keySet) {
            val vx = variableMap.get(key);
            newMap.put(vx.getVertexId(), inputs.get(key));
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
        /**
         * Need to ensure op references
         * are consistent across a graph.
         *
         * This means having a central repository for
         * the variables allocated and using those with in ndarrays
         *
         * We also need a mechanism of validating op structure.
         * Exceptions thrown during calculation should happen
         * in the graph very similar to nd4j.
         */
        if(graph().numVertices() == 0)
            throw new ND4JIllegalStateException("Unable to run exec pipeline. No vertices in graph");


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
            OpExecOrder opExecOrder = context.graph().getOpOrder();
            int finalId = opExecOrder.getActions().get(opExecOrder.getActions().size() - 1).getOutputId()[0];
            return context.getVariableForVertexId(finalId);
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
        defineFunction(function,functionDefinition,new HashMap<String, INDArray>());
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
                    List<OpExecAction> opOrder = sameDiff.graph().getOpOrder(true).getActions();
                    List<OpExecAction> exec = new ArrayList<>();
                    SDVariable gradientBackwardsMarker = sameDiff.gradientBackwardsMarker(sameDiff.getVariableForVertexId(opOrder.get(0).getOutputId()[0]));

                    //start with scalar backprop
                    SDVariable initialGrad = sameDiff.var("one-var",Nd4j.scalar(1.0));
                    SDVariable firstBackward = sameDiff.getVariableForVertexId(opOrder.get(0).getOutputId()[0]);
                    sameDiff.forwardVarForGrad.put(firstBackward.getVertexId(),initialGrad);
                    sameDiff.gradients.put(firstBackward.getVertexId(),initialGrad);



                    Set<String> seen = new LinkedHashSet<>();

                    for(OpExecAction action : opOrder) {
                        if(action == null) {
                            log.warn("Action op state is null");
                            continue;
                        }

                        DifferentialFunction currFunction = sameDiff.getFunction(action.getInputsIds(),action.getOutputId());
                        Preconditions.checkState(currFunction.getSameDiff() == sameDiff,"Wrong samediff instance found!");
                        //Preconditions.checkNotNull("Gradient for " + currFunction.opName() + " was null ! " + sameDiff.getVariableForVertexId(currFunction.getVertexId()).getGradient());
                        val args = currFunction.outputVariables();
                        for(val arg : args) {
                            SDVariable currVar = arg;
                            SDVariable inputGrad = currVar.gradient();
                            Preconditions.checkState(inputGrad.getSameDiff() == sameDiff);
                            List<SDVariable> backwardResult = currFunction.diff(Arrays.asList(inputGrad));
                            //clear out all the variables
                            List<SDVariable> functionVars = debugMode ? new ArrayList<SDVariable>(2) : null;

                            for(int i = 0; i < currFunction.args().length; i++) {
                                DifferentialFunction differentialFunction = sameDiff.setupFunction(backwardResult.get(i));
                                DifferentialFunction x  = sameDiff.setupFunction(currFunction.args()[i]);
                                if(!seen.contains(x.getInstanceId())) {
                                    seen.add(x.getInstanceId());

                                    if (isDebugMode()) {
                                        SDVariable[] add = x.outputVariables();
                                        for(val sdVar : add)
                                            if (sdVar.gradient() != null) {
                                                sameDiff.addVariable(sdVar.gradient());
                                                functionVars.add(sdVar);
                                            }
                                    }
                                }

                            }

                        }

                        if(isDebugMode()) {
                            exec.add(action);
                        }

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
     * Note that if {@link #isPlaceHolder(int)}
     * returns false for the passed in vertex id,
     * a {@link ND4JIllegalStateException} is thrown.
     *
     * A vertex id must be added first. You can
     * do this with {@link #addAsPlaceHolder(int)}
     *
     * @param vertexId the vertex id for the original shape
     * @param shape the shape of the place holder
     */
    public void setOriginalPlaceHolderShape(int vertexId,int[] shape) {
        if(!isPlaceHolder(vertexId)) {
            throw  new ND4JIllegalStateException("Vertex id " + vertexId + " does not appear to be a place holder. Did you forget to call addPlaceHolder?");
        }

        if(shape == null) {
            throw new ND4JIllegalStateException("Null and 0 length shape arrays not allowed");
        }


        if(placeHolderOriginalShapes.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Unable to add a new shape for vertex id " + vertexId);
        }

        //after validation now only set once
        placeHolderOriginalShapes.put(vertexId,shape);

    }


    /**
     * Get the original shape for the vertex id if one was set
     * (other wise returns null).
     * This is mainly for use in validating passed in arrays
     * as arguments to {@link #resolveVariablesWith(Map)}
     * usually when executing using {@link #execWithPlaceHolder(Map)}
     * @param vertexId the vertex id to get the original shape for.
     *
     * @return the set vertex
     */
    public int[] getOriginalShapeForPlaceHolder(int vertexId) {
        return placeHolderOriginalShapes.get(vertexId);
    }

    /**
     * Returns true if this vertex id
     * is a place holder variable or not
     * @param vertexId the vertex id to test
     * @return
     */
    public boolean isPlaceHolder(int vertexId) {
        return placeHolderVertexIds.contains(vertexId);
    }


    /**
     * Add  this vertex id as a place holder
     * @param vertexId the vertex id to add
     */
    public void addAsPlaceHolder(int vertexId) {
        placeHolderVertexIds.add(vertexId);
    }


    /**
     * Resolve all ndarrays by updating the variables
     * for each array specified in the given map.
     * An {@link IllegalStateException} will be thrown
     * if not all arrays are specified for resolution.
     * @param arrays the arrays to resolve.
     */
    public void resolveVariablesWith(Map<String,INDArray> arrays) {
        Preconditions.checkState(arrays.size() == placeHolderVertexIds.size(),"Not all variables specified. " + arrays.size() + " variables were specified, but needed " + placeHolderVertexIds.size());

        for(val arrayEntry : arrays.entrySet()) {
            val varForName = getVariable(arrayEntry.getKey());
            if(varForName == null) {
                throw new ND4JIllegalStateException("No variable name found for " + arrayEntry.getKey());
            }

            if(placeHolderOriginalShapes.containsKey(varForName.getVertexId())) {
                val originalShape = placeHolderOriginalShapes.get(varForName.getVertexId());
                for(int i = 0; i < originalShape.length; i++) {
                    if(originalShape[i] != arrayEntry.getValue().shape()[i] && originalShape[i] >= 1) {
                        throw new ND4JIllegalStateException("Incompatible shape passed for variable. " + Arrays.toString(arrayEntry.getValue().shape()));
                    }
                }
            }
        }



        for(val entry : arrays.entrySet()) {
            val arrVertexId = getVariable(entry.getKey()).getVertexId();
            if(!placeHolderVertexIds.contains(arrVertexId)) {
                throw new ND4JIllegalStateException("Illegal variable " + entry.getKey() + " passed in. Variable found not to be a place holder variable");
            }

            associateArrayWithVariable(entry.getValue(),getVariable(entry.getKey()));
            updateArrayForVertexId(arrVertexId,entry.getValue());

        }

        for(val function : fromToTable.values()) {
            val inputVertexIds = incomingArgsReverse.get(function.getInstanceId());
            int maxDepth = -1;
            for(int i = 0; i < inputVertexIds.length; i++) {
                maxDepth = Math.max(maxDepth,getVariableForVertexId(inputVertexIds[i]).depth());
            }

            val outputArgs = ougoingArgsReverse.get(function.getInstanceId());
            if(outputArgs == null) {
                val shapes = function.calculateOutputShape();
                val outgoingVertexIds = new int[shapes.size()];
                int outgoingVertexIdx = 0;
                for(val shape : shapes) {
                    val newVertexId = graph().nextVertexId();
                    val var = var("output-" + UUID.randomUUID().toString(),shape,new ZeroInitScheme('f'),newVertexId,maxDepth + 1);
                    outgoingVertexIds[outgoingVertexIdx++] = newVertexId;
                    addVariable(var);
                    if(getArrForVertexId(var.getVertexId()) == null)
                        var.storeAndAllocateNewArray();
                }

                addOutgoingFor(outgoingVertexIds,function);
                function.initWithArrays(arrays);
                function.initOutputWithArrays(arrays);
            }

            if(function instanceof CustomOp) {
                CustomOp customOp = (CustomOp) function;
                if(customOp.numInputArguments() < 1) {
                    val args = function.args();
                    for(int i = 0; i < args.length; i++) {
                        val arr = getArrForVertexId(args[i].getVertexId());
                        if(arr == null) {
                            throw new ND4JIllegalStateException("Op " + function.opName() + " not initialized!");
                        }

                        customOp.addInputArgument(arr);
                    }
                }


                if(customOp.numOutputArguments() < 1) {
                    val args = function.outputVariables();
                    for(int i = 0; i < args.length; i++) {
                        val arr = getArrForVertexId(args[i].getVertexId());
                        if(arr == null) {
                            throw new ND4JIllegalStateException("Op " + function.opName() + " not initialized!");
                        }

                        customOp.addOutputArgument(arr);
                    }
                }
            }
        }



        //declare resolved
        resolvedVariables = true;
    }

    /**
     * Returns true if all place holder variables
     * are resolved.
     * A place holder variable is resolved when
     * {@link #getVariableForVertexId(int)}
     * getArr() does not return null and
     * the shape is properly resolved.
     * @return true if all place holder variables are resolved.
     */
    public boolean allPlaceHolderVariablesResolved() {
        for(val vertexId : placeHolderVertexIds) {
            val var = getVariableForVertexId(vertexId);
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
     *
     * @param vertexId the vertex id to add place holders for
     * @param placeHolderVariables the place holder variables
     *                             to add
     */
    public void putPlaceHolderForVertex(int vertexId,int...placeHolderVariables) {
        for(int placeHolderVariable : placeHolderVariables) {
            if(!vertexIdToVariable.containsKey(placeHolderVariable)) {
                throw new ND4JIllegalStateException("No variable found for " + placeHolderVariable);
            }
        }


        List<int[]> placeHolders = placeHolderMap.get(vertexId);
        if(placeHolders == null) {
            placeHolders = new ArrayList<>();
            placeHolderMap.put(vertexId,placeHolders);
        }

        placeHolders.addAll(Arrays.asList(placeHolderVariables));
    }


    /**
     * Returns true if the given vertex id
     * has any placeholder variables
     * @param vertexId the vertex id to check for
     * @return true if this vertex has any place holder
     * variables or not
     */
    public boolean hasPlaceHolderVariables(int vertexId) {
        return placeHolderMap.containsKey(vertexId);
    }

    /**
     * Get the place holders for a given
     * vertex id. May return null.
     *
     * Consider using {@link #hasPlaceHolderVariables(int)}
     * @param vertexId the vertex id to get the place holders for
     * @return the place holder variables for the given vertex
     * id or null
     */
    public List<int[]> getPlaceHoldersFor(int[] vertexId) {
        return placeHolderMap.get(vertexId);
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
        //resolve the place holders
        for(DifferentialFunction function : fromToTable.values()) {
            function.initWithArrays(inputs);
        }


        for(val entry : inputs.entrySet()) {
            associateArrayWithVariable(entry.getValue(),getVariable(entry.getKey()));
        }

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
     * Creates and executes a list of operations
     * @return
     */
    public Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> exec() {
        if(!allPlaceHolderVariablesResolved()) {
            throw new ND4JIllegalStateException("Undefined variables found.");
        }

        if(!resolvedVariables)
            resolveVariablesWith(Collections.emptyMap());

        List<DifferentialFunction> ops = new ArrayList<>();
        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();

        Map<SDVariable,DifferentialFunction> opMap = new HashMap<>();

        boolean onBackward = false;
        for(int i = 0; i < opExecActions.size(); i++) {
            OpExecAction opExecAction = opExecActions.get(i);
            val opName = getFunction(opExecAction.getInputsIds(),opExecAction.getOutputId()).opName();
            if(!onBackward && opName.equals(new GradientBackwardsMarker().opName())) {
                onBackward = true;
            }

            if(opName.equals(new GradientBackwardsMarker().opName()))
                continue;

            DifferentialFunction differentialFunction = createOp(
                    opExecAction);

            //debug
            printFunction(differentialFunction);
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

                    List<int[]> list = execBody.graph().getOutputIds();
                    List<SDVariable> outputs = new ArrayList<>();
                    /**
                     * Find why this is null.
                     */
                    for(int[] output : list) {
                        outputs.add(execBody.getVariableForVertexId(output[0]));
                    }

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


                if(debugMode) {
                    opsForResult.put(opExecAction.getOutputId(),op);
                }

                ops.add(differentialFunction);


                SDVariable currVariable = getVariableForVertexId(opExecAction.getOutputId()[0]);
                if(currVariable ==  null) {
                    List<SDVariable> functions = new ArrayList<>(opExecAction.getInputsIds().length);
                    SDVariable add = SDVariable.builder()
                            .sameDiff(this)
                            .varName(!functions.isEmpty() ? generateVariableName(opName,true,
                                    functions.toArray(new SDVariable[functions.size()])) : opName + "-" + UUID.randomUUID().toString())
                            .shape(op.z().shape())
                            .vertexId(opExecAction.getOutputId()[0])
                            .build();

                    addVariable(add);
                    currVariable = add;

                }

                opMap.put(currVariable,differentialFunction);
            }

        }

        return new Pair<>(opMap,ops);
    }


    public void printFunction(DifferentialFunction differentialFunction) {
        StringBuilder argShapes = new StringBuilder();
        for(val arg : differentialFunction.args()) {
            argShapes.append(" Variable " + getVariableForVertexId(arg.getVertexId()).getVarName() +
                    " Shape for " + Arrays.toString(arg.getShape()));
        }

        for(val func : differentialFunction.outputVariables()) {
            argShapes.append("  Output variable " + getVariableForVertexId(func.getVertexId()) + " is " +
                    Arrays.toString(getVariableForVertexId(func.getVertexId()).getShape()));
        }


        log.info("Executing op " + differentialFunction.opName());

        StringBuilder realShapes = new StringBuilder();
        for(val arg: differentialFunction.args()) {
            val var = getVariableForVertexId(arg.getVertexId());
            realShapes.append(" Input shape for " + var.getVarName() + " is  " + Arrays.
                    toString(getShapeForVertexId(arg.getVertexId())));
        }

        for(val arg: differentialFunction.outputVariables()) {
            val var = getVariableForVertexId(arg.getVertexId());
            realShapes.append(" Output shape for " + var.getVarName() + " is  " + Arrays.
                    toString(getShapeForVertexId(arg.getVertexId())));
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
        val vertexId = getVariable(variableName).getVertexId();
        if(!vertexIdToArr.containsKey(vertexId))
            putArrayForVertexId(vertexId,arr);
        else
            updateArrayForVertexId(vertexId,arr);
    }

    /**
     * Update the opName for the variable
     * with the given vertex id
     * @param vertexId the vertex id to update
     * @param withName thew new opName
     */
    public void updateVariableName(int vertexId,String withName) {
        SDVariable oldVarNameRef = getVariableForVertexId(vertexId);
        variableMap.remove(oldVarNameRef.getVarName());
        variableMap.put(withName,oldVarNameRef);

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
                (byte) 0,
                0,
                0,
                0,
                0,
                -1,
                0.0f, 0, 0);

        return flatNode;
    }

    protected int asFlatNode(@NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder) {
        val hash = getOpNum(node.opName(), node.opType());
        log.info("Exporting node: [{}:<{}>; OpType: {}; Hash/opNum: {}]", node.opName(), node.tensorflowName(), node.opType(), hash);

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

        val outputVertexId = node.outputVertexIds();
        val inputs = graph().getFromFor(outputVertexId);
        for(int input : inputs) {
            for(int i = 0; i < outputVertexId.length; i++) {
                inPaired.add(IntPair.createIntPair(bufferBuilder,input,i));
            }
        }


        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder,outputVertexId);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, node.getDimensions() != null ? node.getDimensions() : new int[]{});
        int fname = bufferBuilder.createString(getVariableForVertexId(outputVertexId[0]).getVarName());
        int scopeName = bufferBuilder.createString("");

        if (node.opType() == null)
            log.warn("Null-op node: {}", node);

        int flatNode = FlatNode.createFlatNode(
                bufferBuilder,
                outputVertexId[0],
                fname,
                getFlatOpType(node.opType()),
                hash,
                nodesIn,
                nodesInPaired,
                (byte) 0,
                nodesOut,
                extraz,
                integerArgs,
                dimensions,
                -1,
                node.opType() == Op.Type.SCALAR ? node.getScalarValue().floatValue() : 0.0f, 0, scopeName);

        return flatNode;
    }


    public ByteBuffer asFlatBuffers(@NonNull ExecutorConfiguration configuration) {
        Nd4j.getExecutioner().commit();
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();

        // first of all we build VariableSpace dump

        for (val variable: variables()) {
            log.info("Exporting variable: [{}]", variable.getVarName());
            if(variable.getArr() == null || variable.getShape() == null)
                continue;

            val arr = variable.getArr();

            int name = bufferBuilder.createString(variable.getVarName());
            int array = arr.toFlatArray(bufferBuilder);
            int id = IntPair.createIntPair(bufferBuilder, variable.getVertexId(), 0);

            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
            flatVariables.add(flatVariable);
        }

        //add functions
        for(val func : fromToTable.values()) {
            flatNodes.add(asFlatNode(func,bufferBuilder));
        }

        // we're dumping scopes now
        for (val scope: sameDiffFunctionInstances.entrySet()) {
            flatNodes.add(asFlatNode(scope.getKey(),scope.getValue(), bufferBuilder));

            // converting all ops from node
            for (val node: scope.getValue().variables()) {
                flatNodes.add(asFlatNode(node, bufferBuilder));
            }
        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        return bufferBuilder.dataBuffer();
    }

    public ByteBuffer asFlatBuffers() {
        val configuration = ExecutorConfiguration.builder()
                .outputMode(org.nd4j.autodiff.execution.conf.OutputMode.IMPLICIT)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .build();

        return asFlatBuffers(configuration);
    }

    public static ByteOrder getOrderFromByte(byte val) {
        if (val == org.nd4j.graph.ByteOrder.LE)
            return ByteOrder.LITTLE_ENDIAN;
        else
            return ByteOrder.BIG_ENDIAN;
    }

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

    public static DataBuffer.Type getDataTypeFromByte(byte val) {
        if (val == DataType.FLOAT)
            return DataBuffer.Type.FLOAT;
        else if (val == DataType.DOUBLE)
            return DataBuffer.Type.DOUBLE;
        else if (val == DataType.HALF)
            return DataBuffer.Type.HALF;

        throw new UnsupportedOperationException("Unsupported DataType: [" + val + "]");
    }

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