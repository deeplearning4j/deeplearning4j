package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.*;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
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
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.lang.reflect.Method;
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
    private DifferentialFunctionFactory functionFactory;
    private Map<String,SDVariable> variableMap;
    private Map<int[],SDVariable> vertexIdToVariable;
    //gradient information
    private Map<int[],SDVariable> gradients;
    private Map<int[],SDVariable> forwardVarForGrad;
    private Map<int[],INDArray> vertexIdToArr;
    private IdentityHashMap<INDArray,SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;
    private Map<int[],DifferentialFunction> functionInstances;
    private static Cloner cloner = new Cloner();
    private static Map<String,Method> opMethods;


    //debug mode variables
    @Getter
    private boolean debugMode;
    private Map<int[],Op> opsForResult;
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

        for(List<Edge<OpState>> edgeList : graph().getEdges().values()) {
            for(Edge<OpState> edge : edgeList) {
                OpStateEdge newEdge = new OpStateEdge(
                        new int[]{thisVertexIdToNew.get(edge.getFrom()[0])},
                        new int[]{thisVertexIdToNew.get(edge.getTo()[0])},
                        cloner.deepCloneDontCloneInstances(edge.getValue()),true);
                newEdge.getValue().setVertexIds(sameDiff.generateVertexIds(newEdge.getFrom()[0],newEdge.getTo()[0]));

                sameDiff.graph().addEdge(newEdge);
            }
        }


        for(List<Edge<OpState>> edgeList : graph().getIncomingEdges().values()) {
            for(Edge<OpState> edge : edgeList) {
                OpStateEdge newEdge = new OpStateEdge(
                        new int[]{thisVertexIdToNew.get(edge.getFrom()[0])},
                        new int[]{thisVertexIdToNew.get(edge.getTo()[0])},
                        cloner.deepCloneDontCloneInstances(edge.getValue()),true);
                newEdge.getValue().setVertexIds(sameDiff.generateVertexIds(newEdge.getFrom()[0],newEdge.getTo()[0]));
                sameDiff.graph().addEdge(newEdge);

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
            Preconditions.checkState(thisVertexIdToNew.containsKey(variable.getVertexId()[0]),variable.getVertexId()[0] + " not found in mapped vertices!");
            int newVertexMap = thisVertexIdToNew.get(variable.getVertexId()[0]);


            deepClone.setVertexId(new int[]{newVertexMap});
            deepClone.setSameDiff(sameDiff);
            sameDiff.addVariable(deepClone);

            if(variable.getArr() != null) {
                sameDiff.associateArrayWithVariable(variable.getArr(),deepClone);
            }


        }


        for(DifferentialFunction function : functionInstances.values())  {
            int[] newVertexId = {thisVertexIdToNew.get(function.getVertexId()[0])};
            DifferentialFunction clone = cloner.deepCloneDontCloneInstances(
                    function,
                    function.getSameDiff(),
                    function.getVertexId());
            clone.setSameDiff(sameDiff);
            clone.setVertexId(newVertexId);
            sameDiff.putFunction(newVertexId,clone);
            for(DifferentialFunction clonedArgs : clone.args()) {
                clonedArgs.setSameDiff(sameDiff);
            }


        }

        sameDiff.reverseArrayLookup.putAll(reverseArrayLookup);
        return sameDiff.variables().get(sameDiff.variables().size() - 1);

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
    public SDVariable getVariableForVertexId(int[] vertexId) {
        if(!vertexIdToVariable.containsKey(vertexId)) {
            if(functionInstances.containsKey(vertexId)) {
                DifferentialFunction func = getFunctionForVertexId(vertexId);
                if(func == null)
                    throw new IllegalArgumentException("No vertex id of " + Arrays.toString(vertexId) + " function or variable found!");
                SDVariable newVar = var(generateVariableName(func.opName(),false,func.args()),func.getShape(),func.depth() + 1);
                Preconditions.checkState(newVar.getSameDiff() == this,"Same diff instance for variable must be the same!");
                vertexIdToVariable.put(vertexId,newVar);
                addVariable(newVar);
            }
            else
                throw new IllegalArgumentException("No vertex id of " + Arrays.toString(vertexId) + " found!");
        }

        return vertexIdToVariable.get(vertexId);
    }


    /**
     * Update the ndarray for the given vertex id.
     * @throws {@link ND4JIllegalStateException} when the array does not exist.
     * @param vertexId
     * @param arr
     */
    public void updateArrayForVertexId(int[] vertexId,INDArray arr) {
        if(vertexIdToArr.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Array for " + Arrays.toString(vertexId) + " does not exist. Please use putArrayForVertexId instead.");
        }
        vertexIdToArr.put(vertexId,arr);
    }

    /**
     * Adds an ndarray for a given vertex id.
     * Use {@link #updateArrayForVertexId(int[], INDArray)}
     * if the array already exists.
     *
     * @param vertexId the vertex id to add
     * @param arr the array to add
     *
     * @throws {@link ND4JIllegalStateException} when the array already exists.
     */
    public void putArrayForVertexId(int[] vertexId,INDArray arr) {
        if(vertexIdToArr.containsKey(vertexId)) {
            throw new ND4JIllegalStateException("Array for " + Arrays.toString(vertexId) + " already exists!");
        }
        vertexIdToArr.put(vertexId,arr);
    }


    /**
     * Returns true if the given vertex id
     * and {@link INDArray} already exist.
     * @param vertexId the vertex id
     * @return true if the ndarray and vertex id already exist
     */
    public boolean arrayAlreadyExistsForVertexId(int...vertexId) {
        return vertexIdToArr.containsKey(vertexId);
    }

    /**
     * Get an {@link INDArray}
     * for a given vertex id
     * @param vertexId
     * @return
     */
    public INDArray getArrForVertexId(int...vertexId) {
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
     * Get the {@link SDVariable}
     * associated with each function
     * based on the {@link DifferentialFunction#resultVertexId()}
     * @param functions the functions to get the variables for
     * @return the list of variables associated with the given {@link DifferentialFunction}
     */
    public List<SDVariable> getVariablesAssociatedWithFunctions(List<DifferentialFunction> functions) {
        List<SDVariable> ret = new ArrayList<>(functions.size());
        for(DifferentialFunction function : functions) {
            ret.add(getVariableForVertexId(function.getVertexId()));
        }

        return ret;
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
     * Returns the {@link SDGraph}
     * asociated with this samediff instance.
     * @return
     */
    public SDGraph getGraph() {
        return graph();
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
        variableMap = new HashMap<>();
        sameDiffFunctionDefinitionMap = new HashMap<>();
        sameDiffFunctionInstances = new HashMap<>();
        functionInstances = new IntArrayKeyMap<>();
        vertexIdToVariable = new IntArrayKeyMap<>();
        gradients = new IntArrayKeyMap<>();
        forwardVarForGrad = new IntArrayKeyMap<>();
        forwardBackwardStates = new HashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        reverseArrayLookup = new IdentityHashMap<>();
        vertexIdToArr = new IntArrayKeyMap<>();
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
    public <X extends DifferentialFunction> X setupFunction(X  function) {
        Preconditions.checkNotNull(function,"Passed in function must not be null!");
        int[] idx = function.getVertexId();
        if(function instanceof SDVariable) {
            if(function.getSameDiff() != this) {
                function.setSameDiff(this);
            }
            return function;
        }
        DifferentialFunction get = null;

        if(idx != null && functionInstances.containsKey(idx)) {
            get = functionInstances.get(idx);
            //note that we check if the graph is frozen
            //if the graph is frozen this reference is disposable
            if(!graph().isFrozen() && !function.equals(get)) {

                throw new IllegalStateException("Attempted to override Differential Function instance with idx " + idx + " with instance " + function);
            }
        }
        else if(idx != null) {
            get = function;
            if(!(get instanceof SDVariable) && !(function instanceof GradientBackwardsMarker))
                functionInstances.put(idx,function);
        }
        else {
            get = function;
        }



        return (X) get;
    }

    /**
     * Generates a set of strings
     * based on int vertex ids
     * @param vertexIds
     * @return
     */
    public String[] generateVertexIds(int[]...vertexIds) {
        List<String> ret = new ArrayList<>();
        for(int i = 0; i < vertexIds.length; i++) {
            for(int j = 0;j < vertexIds[i].length; j++)
                ret.add(String.valueOf(vertexIds[i][j]));
        }

        return ret.toArray(new String[ret.size()]);
    }

    /**
     * Generates a set of strings
     * based on int vertex ids
     * @param vertexIds
     * @return
     */
    public String[] generateVertexIds(int...vertexIds) {
        String[] ret = new String[vertexIds.length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = String.valueOf(vertexIds[i]);
        return ret;
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
            Op op = (Op) opExecAction.get(i);
            ret[i] = op.z();
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
     * @return the created variable
     */
    public SDVariable var(String name, int[] shape, WeightInitScheme weightInitScheme,int[] vertexId) {
        return var(name,shape,weightInitScheme,vertexId,0);
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
    public SDVariable var(String name, int[] shape, WeightInitScheme weightInitScheme,int[] vertexId,int depth) {
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

        if(graph().getVertex(vertexId[0]) == null) {
            NDArrayVertex ndArrayVertex = new NDArrayVertex(this, vertexId[0], depth, ret);
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
        return var(name, shape, weightInitScheme, new int[]{graph.nextVertexId()},depth);

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
                .vertexId(new int[]{ndArrayVertex.getIdx()})
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
                .vertexId(new int[]{vertexIdx})
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
        return gradients.get(vertexId);
    }


    /**
     * Assign a vertex id
     * to a gradient
     * @param vertexId the vertex id
     *                 to assign
     * @param variable the variable
     */
    public void setGradientForVertexId(int[] vertexId, SDVariable variable) {
        gradients.put(vertexId,variable);
    }


    /**
     * Get the forward variable for gradient
     * based on the gradient's vertex id
     * @param vertexId the vertex id
     * @return the gradient for the variable or null
     */
    public SDVariable getForwardVariableForVertexId(int...vertexId) {
        return forwardVarForGrad.get(vertexId);
    }


    /**
     *
     * @param vertexId
     * @param forwardVariable
     */
    public void setForwardVariableForVertexId(int[] vertexId,SDVariable forwardVariable) {
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
                .inputFunctions(getInputs(inputs))
                .sameDiff(this)
                .conv2DConfig(conv2DConfig)
                .build();
        updateVariableName(conv2D.getVertexId(),generateVariableName(conv2D.opName(),false,inputs));
        return getVariableForVertexId(conv2D.getVertexId());
    }


    /**
     * Conv2d operation.
     * @param inputs  the inputs to conv2d
     * @param conv3DConfig the configuration
     * @return
     */
    public SDVariable conv3d(SDVariable[] inputs, Conv3DConfig conv3DConfig) {
        Conv3D conv3D = Conv3D.builder()
                .inputFunctions(getInputs(inputs))
                .conv3DConfig(conv3DConfig)
                .sameDiff(this)
                .build();
        updateVariableName(conv3D.getVertexId(),generateVariableName(conv3D.opName(),false,inputs));
        return getVariableForVertexId(conv3D.getVertexId());
    }


    public DifferentialFunction[] getInputs(SDVariable...inputs) {
        DifferentialFunction[] ret = new DifferentialFunction[inputs.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getFunctionInput(inputs[i]);
        }

        return ret;
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
        DifferentialFunction result = functionFactory.gradientBackwardsMarker(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.neq(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.eq(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.gte(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.lte(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.gt(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX,double iy) {
        DifferentialFunction result = functionFactory.lt(getFunctionInput(iX),iy);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }





    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.neq(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.eq(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.gte(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.lte(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.gt(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.lt(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(String name,SDVariable iX, SDVariable iy) {
        DifferentialFunction result = functionFactory.or(getFunctionInput(iX),getFunctionInput(iy));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.neg(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.cos(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.sin(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.tan(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.acos(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.asin(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.atan(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.cosh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.sinh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());

    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(String name,SDVariable iX) {
        DifferentialFunction
                result = functionFactory.tanh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.acosh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.asinh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.atanh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.exp(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.log(getFunctionInput(iX));
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
        DifferentialFunction result = functionFactory.pow(getFunctionInput(iX),value);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.sqrt(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.square(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.floor(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(String name,SDVariable iX,double cutoff) {
        DifferentialFunction result = functionFactory.relu(getFunctionInput(iX),cutoff);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.softmax(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmaxDerivative(String name,SDVariable iX,SDVariable wrt) {
        DifferentialFunction result = functionFactory.softmaxDerivative(getFunctionInput(iX),getFunctionInput(wrt));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanh(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.hardTanh(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.hardTanhDerivative(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.sigmoid(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(String name,SDVariable iX,SDVariable wrt) {
        DifferentialFunction result = functionFactory
                .sigmoidDerivative(getFunctionInput(iX), getFunctionInput(wrt));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory
                .sign(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.softsign(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.softsignDerivative(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.softplus(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.elu(getFunctionInput(iX));
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.eluDerivative(getFunctionInput(iX));
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
        DifferentialFunction result = functionFactory.leakyRelu(getFunctionInput(iX),cutoff);
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
        DifferentialFunction result = functionFactory.leakyReluDerivative(getFunctionInput(iX),
                getFunctionInput(wrt),
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
        DifferentialFunction result = functionFactory.mean(getFunctionInput(iX));
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
        DifferentialFunction result = functionFactory.std(
                getFunctionInput(iX),
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
        DifferentialFunction result = functionFactory.variance(getFunctionInput(iX),
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
        DifferentialFunction result = functionFactory.sum(getFunctionInput(iX),dimensions);
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
        DifferentialFunction result = functionFactory.prod(getFunctionInput(iX),dimensions);
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
        DifferentialFunction result = functionFactory.max(getFunctionInput(iX),dimensions);
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
        DifferentialFunction result = functionFactory.min(getFunctionInput(iX),dimensions);
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
        DifferentialFunction result = functionFactory
                .reshape(getFunctionInput(iX),shape);
        updateVariableName(result.getVertexId(),name);
        return getVariableForVertexId(result.getVertexId());
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(String name,SDVariable iX) {
        DifferentialFunction result = functionFactory.transpose(getFunctionInput(iX));
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
        DifferentialFunction result = functionFactory.rollAxis(x,axis);
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
        DifferentialFunction result = functionFactory.mmul(x, y);
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
        DifferentialFunction result = functionFactory.tensorMmul(getFunctionInput(x),getFunctionInput(y), dimensions);
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
        DifferentialFunction cosim = functionFactory.cosineSimilarity(
                getFunctionInput(iX),
                getFunctionInput(i_y),
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
        DifferentialFunction result = functionFactory.euclideanDistance(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.manhattanDistance(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossBinaryXENT(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossCosineSimilarity(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossHinge(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossKLD(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossL1(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossL2(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossMAE(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossMSE(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossMCXENT(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossMSLE(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossMSLE(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossPoisson(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        DifferentialFunction result = functionFactory.lossSquaredHinge(getFunctionInput(iX),getFunctionInput(i_y),dimensions);
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
        vertexIdToVariable.put(variable.resultVertexId(),variable);
        variableMap.put(variable.getVarName(),variable);

    }


    /**
     *
     * @param funcName
     * @param grad
     * @param inputs
     * @return
     */
    public String generateVariableName(String funcName,boolean grad,DifferentialFunction...inputs) {
        StringBuilder sb = new StringBuilder();
        sb.append(funcName).append("(");
        if(inputs != null) {
            for (DifferentialFunction variable : inputs) {
                if (grad) {
                    sb.append("-grad");
                }

                sb.append("-");
                sb.append(variable.resultVertexId());
                sb.append("-");


                if (variable.getOpState() != null)
                    sb.append(Arrays.toString(variable.getOpState().getVertexIds()));
                sb.append(",");
            }
        }


        return sb.toString();

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
                if ((getFunctionInput(variable) != null)) {
                    sb.append(getFunctionInput(variable).resultVertexId());
                    sb.append("-");
                }

                if (getFunctionInput(variable).getOpState() != null)
                    sb.append(Arrays.toString(getFunctionInput(variable).getOpState().getVertexIds()));
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
     * Add a function to this instance for tracking
     * @param vertexId the vertex id
     * @param function the function
     */
    public void putFunction(int[] vertexId,DifferentialFunction function) {
        if(function instanceof GradientBackwardsMarker) {
            return;
        }


        if(function instanceof SDVariable) {
            SDVariable sdVariable = (SDVariable) function;
            Preconditions.checkState(sdVariable.getSameDiff() == this,"Same diff instance for variable must be the same!");
            this.vertexIdToVariable.put(vertexId,sdVariable);
        }
        else {
            this. functionInstances.put(vertexId,function);
        }
    }




    /**
     * Get the function for the given vertex id
     * @param vertexId the vertex id to get the function for
     * @return the
     */
    public DifferentialFunction getFunctionForVertexId(int...vertexId) {
        return functionInstances.get(vertexId);
    }


    /**
     *
     * @param opExecAction
     * @return
     */
    public DifferentialFunction createOp(OpExecAction opExecAction) {
        DifferentialFunction differentialFunction = getFunctionForVertexId(opExecAction.getOutputId());
        if(differentialFunction instanceof Op) {
            if (differentialFunction instanceof ScalarOp) {
                ScalarOp scalarOp = (ScalarOp) differentialFunction;
                scalarOp.setScalar(differentialFunction.getScalarValue());

            }


            Op op = (Op) differentialFunction;
            differentialFunction.fillInArrays();
            op.setN(op.x().length());

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
        Op op = (Op) exec.get(exec.size() - 1);
        return op.z();
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

    private DifferentialFunction getFunctionInput(SDVariable iX) {
        DifferentialFunction ret =  getFunctionForVertexId(iX.getVertexId()) != null ?
                getFunctionForVertexId(iX.getVertexId())  : iX;
        ret = setupFunction(ret);
        Preconditions.checkState(iX.getSameDiff() != null,"Samediff instance must not be null.");
        if(graph().getGraphApply() == null) {
            Preconditions.checkState(ret.getSameDiff() == functionFactory.getSameDiff(), "Function input does not have same samediff instance as get value");
        }
        return ret;
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
            OpExecOrder opExecOrder = context.getGraph().getOpOrder();
            int[] finalId = opExecOrder.getActions().get(opExecOrder.getActions().size() - 1).getOutputId();
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
                    SDVariable gradientBackwardsMarker = sameDiff.gradientBackwardsMarker(sameDiff.getVariableForVertexId(opOrder.get(0).getOutputId()));

                    //start with scalar backprop
                    SDVariable initialGrad = sameDiff.one("one-var",new int[]{1,1});
                    SDVariable firstBackward = sameDiff.getVariableForVertexId(opOrder.get(0).getOutputId());
                    sameDiff.forwardVarForGrad.put(firstBackward.getVertexId(),initialGrad);
                    sameDiff.gradients.put(firstBackward.getVertexId(),initialGrad);



                    Set<DifferentialFunction> seen = new HashSet<>();

                    for(OpExecAction action : opOrder) {
                        if(action == null || action.getOpState() == null) {
                            log.warn("Action op state is null");
                            continue;
                        }

                        DifferentialFunction currFunction = sameDiff.getFunctionForVertexId(action.getOutputId());
                        Preconditions.checkState(currFunction.getSameDiff() == sameDiff,"Wrong samediff instance found!");
                        Preconditions.checkNotNull("Gradient for " + currFunction.opName() + " was null ! " + sameDiff.getVariableForVertexId(currFunction.getVertexId()).getGradient());
                        SDVariable currVar = sameDiff.getVariableForVertexId(currFunction.getVertexId());
                        SDVariable inputGrad = currVar.gradient();
                        Preconditions.checkState(inputGrad.getSameDiff() == sameDiff);
                        List<DifferentialFunction> backwardResult = currFunction.diff(Arrays.<DifferentialFunction>asList(inputGrad));
                        //clear out all the variables
                        List<SDVariable> functionVars = debugMode ? new ArrayList<SDVariable>(2) : null;

                        for(int i = 0; i < currFunction.args().length; i++) {
                            DifferentialFunction differentialFunction = sameDiff.setupFunction(backwardResult.get(i));
                            DifferentialFunction x  = sameDiff.setupFunction(currFunction.args()[i]);
                            if(!seen.contains(x)) {
                                seen.add(x);
                                SDVariable add = sameDiff.getVariableForVertexId(differentialFunction.resultVertexId());

                                if (isDebugMode()) {
                                    if (add.gradient() != null)
                                        sameDiff.addVariable(add.gradient());
                                    functionVars.add(add);
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
    public Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> exec() {
        List<DifferentialFunction> ops = new ArrayList<>();
        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();

        Map<SDVariable,DifferentialFunction> opMap = new HashMap<>();

        boolean onBackward = false;
        for(int i = 0; i < opExecActions.size(); i++) {

            OpExecAction opExecAction = opExecActions.get(i);
            if(!onBackward && opExecAction.getOpState().getOpName().equals(new GradientBackwardsMarker().opName())) {
                onBackward = true;
            }

            if(opExecAction.getOpState().getOpName().equals(new GradientBackwardsMarker().opName()))
                continue;

            DifferentialFunction differentialFunction = createOp(
                    opExecAction);
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
                        outputs.add(execBody.getVariableForVertexId(output));
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

            else if(differentialFunction instanceof Op) {
                Op op = (Op) differentialFunction;
                if(opExecAction.getOpState().getAxes() == null)
                    Nd4j.getExecutioner().exec(op);

                else {
                    int[] axes = opExecAction.getOpState().getAxes();
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


                SDVariable currVariable = getVariableForVertexId(opExecAction.getOutputId());
                if(currVariable ==  null) {
                    List<SDVariable> functions = new ArrayList<>(opExecAction.getInputsIds().length);
                    SDVariable add = SDVariable.builder()
                            .sameDiff(this)
                            .varName(!functions.isEmpty() ? generateVariableName(opExecAction.getOpState().getOpName(),true,
                                    functions.toArray(new SDVariable[functions.size()])) : opExecAction.getOpState().getOpName() + "-" + UUID.randomUUID().toString())
                            .shape(op.z().shape())
                            .vertexId(opExecAction.getOutputId())
                            .build();

                    addVariable(add);
                    currVariable = add;

                }
                else {
                    associateArrayWithVariable(op.z(), currVariable);
                }

                opMap.put(currVariable,differentialFunction);
                putFunction(opExecAction.getOutputId(),differentialFunction);
            }

        }

        return new Pair<>(opMap,ops);
    }


    /**
     * Update the opName for the variable
     * with the given vertex id
     * @param vertexId the vertex id to update
     * @param withName thew new opName
     */
    public void updateVariableName(int[] vertexId,String withName) {
        SDVariable oldVarNameRef = getVariableForVertexId(vertexId);
        variableMap.remove(oldVarNameRef.getVarName());
        variableMap.put(withName,oldVarNameRef);

    }


}