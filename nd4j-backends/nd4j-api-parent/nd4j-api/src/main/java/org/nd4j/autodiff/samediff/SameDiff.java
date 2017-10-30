package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.transforms.Constant;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.*;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
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
@Data
@Builder
@Slf4j
public class SameDiff {
    private SDGraph graph;
    private DifferentialFunctionFactory functionFactory;
    private Map<String,SDVariable> variableMap;
    private Map<int[],SDVariable> vertexIdToVariable;
    private Map<String,INDArray> vertexToArray;
    private IdentityHashMap<INDArray,SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;
    private Map<int[],DifferentialFunction> functionInstances;
    private static Cloner cloner = new Cloner();
    private static Map<String,Method> opMethods;


    //debug mode variables
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
            if(clone.getOpState() != null && clone.getOpState().getDifferentialFunction() != null)
                clone.getOpState().getDifferentialFunction().setSameDiff(sameDiff);
            NDArrayVertex info = new NDArrayVertex(
                    sameDiff,
                    nextVertexId,
                    graph.getVertex(i + 1).depth(),
                    clone);
            thisVertexIdToNew.put(graph.getVertex(i + 1).vertexID(),nextVertexId);
            sameDiff.graph().addVertex(info);
        }



        for(int i = 0; i < graph().numVertices(); i++) {
            /**
             * In this loop also remap the
             * same diff variables
             * with setupFunction,
             */
            List<Edge<OpState>> edgesForVertex = graph.getEdges().get(new int[]{i + 1});
            List<Edge<OpState>> incomingEdgesForVertex = graph.getIncomingEdges()
                    .get(new int[]{i + 1});
            //map to new vertex
            int newVertexMap = thisVertexIdToNew.get(i + 1);
            if(edgesForVertex != null) {
                List<Edge<OpState>> edgesForNewVertex = new ArrayList<>();
                sameDiff.graph().getEdges().put(new int[]{newVertexMap}, edgesForNewVertex);
                for (Edge<OpState> edge : edgesForVertex) {
                    Preconditions.checkState(thisVertexIdToNew.containsKey(edge.getFrom()[0]),"Edge missing from vertex id for copy " + edge.getFrom()[0]);
                    Preconditions.checkState(thisVertexIdToNew.containsKey(edge.getTo()[0]),"Edge missing to vertex id for copy " + edge.getTo()[0]);

                    OpStateEdge newEdge = new OpStateEdge(
                            new int[]{thisVertexIdToNew.get(edge.getFrom()[0])},
                            new int[]{thisVertexIdToNew.get(edge.getTo()[0])},
                            cloner.deepClone(edge.getValue()), true);
                    newEdge.getValue().setVertexIds(sameDiff.generateVertexIds(newEdge.getFrom()[0],newEdge.getTo()[0]));
                    edgesForNewVertex.add(newEdge);

                }
            }

            if(incomingEdgesForVertex != null) {
                List<Edge<OpState>> newIncomingEdges = new ArrayList<>();
                sameDiff.graph().getIncomingEdges().put(new int[]{newVertexMap},newIncomingEdges);
                for(Edge<OpState> edge : incomingEdgesForVertex) {
                    OpStateEdge newEdge = new OpStateEdge(
                            new int[]{thisVertexIdToNew.get(edge.getFrom()[0])},
                            new int[]{thisVertexIdToNew.get(edge.getTo()[0])},
                            cloner.deepCloneDontCloneInstances(edge.getValue()),true);
                    newEdge.getValue().setVertexIds(sameDiff.generateVertexIds(newEdge.getFrom()[0],newEdge.getTo()[0]));

                    newIncomingEdges.add(newEdge);

                    if(newEdge.getValue().getDifferentialFunction() != null) {
                        ensureSameDiffInstance(sameDiff,newEdge.getValue().getDifferentialFunction());
                        newEdge.getValue().setDifferentialFunction(sameDiff.setupFunction(newEdge.getValue().getDifferentialFunction()));
                        newEdge.getValue().getDifferentialFunction().setVertexId(edge.getValue().getDifferentialFunction().resultVertexId());
                    }
                }
            }



            if(functionInstances.containsKey(new int[]{i + 1})) {
                DifferentialFunction function = functionInstances.get(new int[]{i + 1});
                if(function instanceof SDVariable)
                    continue;
                DifferentialFunction clone = sameDiff.setupFunction(cloner.deepClone(function));
                clone.setVertexId(new int[]{newVertexMap});
                sameDiff.functionInstances.put(new int[]{newVertexMap},clone);
                ensureSameDiffInstance(sameDiff,clone);
            }



        }



        List<SDVariable> variables = variables();

        //copy over variables
        for(SDVariable variable : variables) {
            SDVariable deepClone = cloner.deepCloneDontCloneInstances(
                    variable,
                    variable.getDifferentialFunction(),
                    variable.getVertex(),
                    variable.getArr(),
                    variable.getSameDiff(),
                    variable.getShape());
            Preconditions.checkState(thisVertexIdToNew.containsKey(variable.getVertexId()[0]),variable.getVertexId()[0] + " not found in mapped vertices!");
            int newVertexMap = thisVertexIdToNew.get(variable.getVertexId()[0]);

            //change the vertex id to the new value
            //for the graph transition
            if(variable.getDifferentialFunction() != null) {
                DifferentialFunction val = sameDiff.functionInstances.get(new int[]{newVertexMap});
                deepClone.setDifferentialFunction(val);

            }


            if(variable.getVertex() != null)
                deepClone.setVertex((NDArrayVertex) sameDiff.graph().getVertex(newVertexMap));

            deepClone.setVertexId(new int[]{newVertexMap});
            deepClone.setSameDiff(sameDiff);
            sameDiff.addVariable(deepClone);



        }

        sameDiff.reverseArrayLookup.putAll(reverseArrayLookup);
        sameDiff.vertexToArray.putAll(vertexToArray);
        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }


    /**
     * Get the variable for the given vertex id
     * @param vertexId the vertex id (usually a 1 length array but can be multiple)
     * @return the variable for this vertex
     */
    public SDVariable getVariableForVertexId(int[] vertexId) {
        if(!vertexIdToVariable.containsKey(vertexId)) {
            throw new IllegalArgumentException("No vertex id of " + Arrays.toString(vertexId) + " found!");
        }

        return vertexIdToVariable.get(vertexId);
    }

    /**
     * Return the array information
     * for the given array
     * (note that array references
     * are used rather than a clone, so dup() ed arrays
     * will not work here)
     * @param arr the array reference to get the information for
     * @return the {@link SDVariable}
     * for the given array reference
     */
    public SDVariable getInfoFor(INDArray arr) {
        return reverseArrayLookup.get(arr);
    }


    private void ensureSameDiffInstance(SameDiff sameDiff,DifferentialFunction val) {
        val.setSameDiff(sameDiff);
        if(val instanceof SDVariable) {
            SDVariable variable1 = (SDVariable) val;
            variable1.setSameDiff(sameDiff);
            variable1.setVertexId(val.getVertexId());
            sameDiff.setupFunction(variable1);


        }
        else if(val instanceof Constant) {
            Constant constant = (Constant) val;
            constant.setSameDiff(sameDiff);
            sameDiff.setupFunction(constant);
        }

        //recursive case
        else if(val.args() != null) {
            for(DifferentialFunction equation  : val.args()) {
                sameDiff.setupFunction(equation);
                ensureSameDiffInstance(sameDiff,equation);

            }
        }

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
     * Invoke an op by name
     * @param op the op
     * @param x the first input
     * @param y the second input
     * @return the result variable
     */
    public SDVariable invoke(Op op,SDVariable x,SDVariable y) {
        if(!opMethods.containsKey(op.name())) {
            throw new ND4JIllegalStateException("Illegal method name " + op.name());
        }

        if(x != null && y != null) {
            try {
                return (SDVariable) opMethods.get(op.name()).invoke(this, x, y);
            }catch(Exception e) {

            }
        }
        else {
            try {
                return (SDVariable) opMethods.get(op.name()).invoke(this, x);
            }catch(Exception e) {

            }
        }

        throw new ND4JIllegalStateException("Illegal method name " + op.name());

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
     * Invoke an op by name
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
        vertexToArray = new HashMap<>();
        sameDiffFunctionDefinitionMap = new HashMap<>();
        sameDiffFunctionInstances = new HashMap<>();
        functionInstances = new IntArrayKeyMap<>();
        vertexIdToVariable = new IntArrayKeyMap<>();
        forwardBackwardStates = new HashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        reverseArrayLookup = new IdentityHashMap<>();
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
            functionInstances.put(idx,function);
        }
        else {
            get = function;
        }

        if(idx == null || get.getSameDiff() != this || get.getVertex() == null) {
            /**
             * Note that we generate a new id
             * if the intended samediff instance
             * isn't "this" samediff.
             *
             * Otherwise, we assume that the intended index
             * is already being set.
             *
             * The reason for this is to allow for people to
             * set the id externally.
             *
             * We also check if the id is zero (unset, an id can never be < 1)
             */
            NDArrayVertex ndArrayVertex = new NDArrayVertex(this,get.getSameDiff() != this || idx == null || idx[0] == 0 ? graph().nextVertexId() : idx[0],0,get.getResult());
            graph().addVertex(ndArrayVertex);
            get.setVertex(ndArrayVertex);
            get.setSameDiff(this);
            get.setVertexId(new int[]{ndArrayVertex.vertexID()});
        }

        if(!graph().getVertices().containsKey(get.getVertex().vertexID())) {
            graph().addVertex(get.getVertex());
        }

        if(get instanceof SDVariable) {
            SDVariable sdVariable  = (SDVariable) get;
            if(!variableMap.containsKey(sdVariable.getVarName()))
                variableMap.put(sdVariable.getVarName(),sdVariable);
            if(!vertexIdToVariable.containsKey(sdVariable.resultVertexId()))
                vertexIdToVariable.put(sdVariable.getOutputVertexIds(),sdVariable);

            if( sdVariable.getArr() != null) {
                reverseArrayLookup.put(sdVariable.getArr(),sdVariable);
            }

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
                .variableMap(originalSameDiff.getVariableMap())
                .sameDiffFunctionInstances(originalSameDiff.getSameDiffFunctionInstances())
                .vertexToArray(originalSameDiff.getVertexToArray())
                .graph(clone)
                .build();
        //ensuring proper sameDiff reference
        clone.setSameDiff(ret);
        DifferentialFunctionFactory differentialFunctionFactory =
                new
                        DifferentialFunctionFactory(ret);
        ret.setFunctionFactory(differentialFunctionFactory);
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
        if (vertexToArray != null ? !vertexToArray.equals(sameDiff.vertexToArray) : sameDiff.vertexToArray != null)
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



    /**
     * Allocate ndarrays in to memory,
     * linking the {@link SDVariable}
     * {@link INDArray}
     * provided with {@link SDVariable#getArr()}
     * as needed.
     */
    public void allocate() {
        if(workspace != null) {
            workspace.close();
        }
        else {
            initWorkspace();
        }


        for (Integer i : graph().getVertices().keySet()) {
            SDVariable info = graph.getVariableForVertex(i);
            allocateArrayFor(info);
        }

    }


    /**
     * Allocate an individual {@link INDArray}
     * for a given {@link SDVariable}
     * @param sdVariable the variable to allocate
     *                   memory for
     */
    public void allocateArrayFor(SDVariable sdVariable) {
        if(workspace != null) {
            workspace.close();
        }
        else {
            initWorkspace();
        }

        SDVariable info = sdVariable;
        DifferentialFunction func = functionInstances.get(info.getVertexId());

        if(!variableMap.containsKey(info.getVarName())) {
            SDVariable.SDVariableBuilder variableBuilder = SDVariable.builder()
                    .sameDiff(this)
                    .varName(info.getVarName());
            //associate the proper differential function with the given
            //variable
            if(func != null)
                variableBuilder.differentialFunction(func);

            if(func != null)
                variableBuilder.shape(info.getShape());

            variableBuilder.vertexId(info.getVertexId());

            SDVariable variable = variableBuilder.build();
            variableMap.put(info.getVarName(),variable);
        }

        /**
         * Problem:
         * Vertexes are not a unique identifier of an actual array.
         * Duplicate vertices are put in to place
         * to avoid cycles by may point at the same array.
         * NDArrayInformation should somehow be unique
         * and point to an actual array.
         */
        if(!vertexToArray.containsKey(info.getVarName()) || vertexToArray.get(info.getVarName()) == null) {
            //initialize value if it's actually a scalar constant (zero or 1 typically...)
            if(info.getScalarValue() != null && ArrayUtil.prod(info.getShape()) == 1) {
                INDArray arr = Nd4j.valueArrayOf(info.getShape(),
                        info.getScalarValue().doubleValue());
                vertexToArray.put(info.getVarName(),arr);
                reverseArrayLookup.put(arr,info);
                info.setArr(arr);
            }
            else {
                INDArray newAlloc = info.getWeightInitScheme().create(info.getShape(),Nd4j.zeros(info.getShape(),info.getWeightInitScheme().order()));
                vertexToArray.put(info.getVarName(),newAlloc);
                reverseArrayLookup.put(newAlloc,info);
                info.setArr(newAlloc);

            }

        }


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
     *
     *
     * @param name
     * @param shape
     * @return
     */
    public SDVariable var(String name, int[] shape,int depth) {
        if(variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            return variableMap.get(name);


        if(name == null || name.length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if(workspace == null)
            initWorkspace();


        int[] vertexId = {graph.nextVertexId()};
        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .vertexId(vertexId)
                .shape(shape).weightInitScheme(new ZeroInitScheme('f'))
                .varName(name)
                .build();

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,vertexId[0], depth,ret);
        graph.addVertex(ndArrayVertex);
        addVariable(ret);
        variableMap.put(name,ret);
        return ret;

    }


    /**
     * Creates a {@link SDVariable}
     * ,{@link NDArrayVertex}
     * with the given shape
     * and a depth of 0.
     *
     * @param name the name of the variable
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
        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .vertexId(new int[]{ndArrayVertex.getIdx()})
                .shape(arr.getShape())
                .varName(arr.getVarName())
                .differentialFunction(arr.getDifferentialFunction())
                .weightInitScheme(new NDArraySupplierInitScheme(new NDArraySupplierInitScheme.NDArraySupplier() {
                    @Override
                    public INDArray getArr() {
                        /**
                         * Pre allocate the array if it doesn't already exist.
                         * The reason we do this is to avoid race conditions with
                         * {@link #allocate()}
                         */
                        if(arr.getArr() == null) {
                            arr.setArr(arr.getWeightInitScheme().create(arr.getShape()));
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
                .arr(arr).build();
        if(ArrayUtil.prod(arr.shape()) == 1)
            ret.setScalarValue(arr.getDouble(0));

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,vertexIdx, 0,ret);
        graph.addVertex(ndArrayVertex);

        addVariable(ret);
        //ensure there is a reference to the array in the integer index
        //this is used later for op creation
        vertexToArray.put(ret.getVarName(), arr);
        reverseArrayLookup.put(arr, ret);
        variableMap.put(name,ret);
        return ret;

    }

    /**
     * Get the variable based on the name
     * @param name the name of the variable
     * @return the variabel instance if there is one
     *
     */
    public SDVariable getVariable(String name) {
        return getVariableMap().get(name);
    }


    /**
     * Gradient with respect
     * to the given variable name.
     * Note that in order to run this function,
     * {@link #execBackwards()} must be executed first.
     * All gradient functions are obtained within that time.
     * @param varName the variable name to get the gradient for.
     * @return
     */
    public SDVariable grad(String varName) {
        if(!sameDiffFunctionInstances.containsKey("grad")) {
            throw new IllegalStateException("Unable to obtain gradient. Please run execBackwards() first.");
        }

        return getFunction("grad").getVariable(varName).gradient();
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


        SDVariable ret = SDVariable.builder()
                .differentialFunction(conv2D)
                .sameDiff(this)
                .varName(conv2D.opName() + "(" + createName(inputs) + ")")
                .build();
        return ret;
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

        SDVariable ret = SDVariable.builder()
                .differentialFunction(conv3D)
                .sameDiff(this)
                .varName(conv3D.opName() + "(" + createName(inputs) + ")")
                .build();
        return ret;
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
     * Returns the ndarrays
     * allocated for a given
     * {@link SDVariable}
     * @param info the information to get the array for
     * @return
     */
    public INDArray getNDArray(SDVariable  info) {
        return getVertexToArray().get(info.getVarName());
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
        return gradientBackwardsMarker(generateVariableName(new GradientBackwardsMarker().name(),true,iX),iX);
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
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.gradientBackwardsMarker(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neq(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.eq(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.gte(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.lte(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.gt(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX,double iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.lt(getFunctionInput(iX),iy))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }





    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neq(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eq(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.eq(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.gte(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.lte(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }




    /**
     *
     * @param iX
     * @return
     */
    public SDVariable gt(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.gt(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable lt(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.lt(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable or(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.or(getFunctionInput(iX),getFunctionInput(iy)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neg(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neg(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cos(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.cos(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sin(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.sin(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tan(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.tan(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acos(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.acos(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */

    public SDVariable asin(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.asin(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atan(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.atan(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable cosh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.cosh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sinh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.sinh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable tanh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.tanh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable acosh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.acosh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable asinh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.asinh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable atanh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.atanh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable exp(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.exp(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable log(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.log(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param value
     * @return
     */
    public SDVariable pow(String name,SDVariable iX,double value) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.pow(getFunctionInput(iX),value))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sqrt(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.sqrt(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable square(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.square(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable floor(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.floor(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable relu(String name,SDVariable iX,double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.relu(getFunctionInput(iX),cutoff))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmax(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softmax(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softmaxDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softmaxDerivative(getFunctionInput(iX),getFunctionInput(wrt)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanh(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.hardTanh(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable hardTanhDerivative(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.hardTanhDerivative(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoid(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .sameDiff(this)
                .differentialFunction(functionFactory.sigmoid(getFunctionInput(iX)))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }



    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(String name,SDVariable iX,SDVariable wrt) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory
                        .sigmoidDerivative(getFunctionInput(iX), getFunctionInput(wrt)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sign(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory
                        .sign(getFunctionInput(iX))).differentialFunction(functionFactory
                        .sign(iX.getDifferentialFunction()))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsign(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softsign(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softsignDerivative(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softsignDerivative(getFunctionInput(iX)))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable softplus(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softplus(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable elu(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.elu(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable eluDerivative(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.eluDerivative(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public SDVariable leakyRelu(String name,SDVariable iX, double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.leakyRelu(getFunctionInput(iX),cutoff))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param wrt
     * @param cutoff
     * @return
     */
    public SDVariable leakyReluDerivative(String name,SDVariable iX, SDVariable wrt,double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.leakyReluDerivative(getFunctionInput(iX),
                        getFunctionInput(wrt),
                        cutoff))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SDVariable mean(String name,SDVariable iX) {

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.mean(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
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
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .sameDiff(this)
                .differentialFunction(functionFactory.std(
                        getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
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
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.variance(getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable sum(String name,SDVariable iX,
                          int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.sum(getFunctionInput(iX),dimensions))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable prod(String name,SDVariable iX,
                           int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.prod(getFunctionInput(iX),dimensions))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable max(String name,SDVariable iX, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.max(getFunctionInput(iX),dimensions))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public SDVariable min(String name,SDVariable iX,
                          int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.min(getFunctionInput(iX),dimensions))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
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

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory
                        .reshape(getFunctionInput(iX),shape))
                .varName(name)
                .shape(shape)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable transpose(String name,SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.transpose(getFunctionInput(iX)))
                .varName(name)
                .sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public SDVariable rollAxis(String name,SDVariable x, int axis) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .sameDiff(this)
                .differentialFunction(functionFactory.rollAxis(x,axis))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(String name,SDVariable x, SDVariable y) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .sameDiff(this)
                .differentialFunction(functionFactory.mmul(x, y))
                .varName(name)
                .build();
        ret.setShape(Shape.getMatrixMultiplyShape(x.getShape(),y.getShape()));
        addVariable(ret);
        return ret;
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

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.tensorMmul(getFunctionInput(x),getFunctionInput(y), dimensions))
                .varName(name)
                .sameDiff(this)
                .shape(shape)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
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

        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(cosim)
                .varName(name)
                .sameDiff(this).shape(arrayReduceShape)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable euclideanDistance(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.euclideanDistance(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable manhattanDistance(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.manhattanDistance(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossBinaryXENT(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossBinaryXENT(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossCosineSimilarity(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossCosineSimilarity(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossHinge(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossHinge(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossKLD(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossKLD(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL1(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossL1(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossL2(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossL2(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMAE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMAE(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     public SDVariable lossMAPE(String name,SDVariable iX,SDVariable i_y,int...dimensions) {
     int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

     SDVariable ret = SDVariable.builder()
     .arr(null).shape(arrayReduceShape)
     .differentialFunction(functionFactory.lossMAPE(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
     .varName(name)
     .build();
     Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
     addVariable(ret);
     return ret;
     }

     /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMSE(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMCXENT(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMCXENT(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossMSLE(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMSLE(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossNegativeLogLikelihood(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.transpose(getFunctionInput(iX)))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossPoisson(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossPoisson(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param i_y
     * @param dimensions
     * @return
     */
    public SDVariable lossSquaredHinge(String name,SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossSquaredHinge(getFunctionInput(iX),getFunctionInput(i_y),dimensions))
                .varName(name)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param variable
     */
    public void addVariable(SDVariable variable) {
        if(variableMap == null)
            variableMap = new HashMap<>();



        /**
         * Of note here:
         * We don't validate base don vertex id
         * because more than one input can have the same
         * vertex id as a result.
         *
         * We validate based on variable name instead
         * which takes in to account function names as well
         * as input ids
         */
        if(variableMap.containsKey(variable.getVarName()) && !variableMap.get(variable.getVarName()).equals(variable)) {
            throw new IllegalArgumentException("Variable already found with variable name " + variable.getVarName());
        }

        vertexIdToVariable.put(getFunctionInput(variable).resultVertexId(),variable);
        variableMap.put(variable.getVarName(),variable);
        if( variable.getArr() != null) {
            reverseArrayLookup.put(variable.getArr(),variable);
        }

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
     * given the name
     * @param functionName the name of the function
     * @return the same diff function instance
     * defined for the given name
     */
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }


    private INDArray getX(OpExecAction opExecAction) {
        INDArray ret =  vertexToArray.get(opExecAction.getInputs()[0].getVarName());
        return ret;
    }

    private INDArray getY(OpExecAction opExecAction) {
        if(opExecAction.getInputsIds().length > 1) {
            SDVariable opId = opExecAction.getInputs()[1];
            INDArray ret = vertexToArray.get(opId.getVarName());
            return ret;
        }
        return null;
    }

    private INDArray getZ(OpExecAction opExecAction) {
        if(opExecAction.isInPlace())
            return getX(opExecAction);
        SDVariable opId = opExecAction.getOutput();
        INDArray ret =  vertexToArray.get(opId.getVarName());
        return ret;
    }


    /**
     *
     * @param opType
     * @param opExecAction
     * @return
     */
    public DifferentialFunction createOp(Op.Type opType,
                                         OpExecAction opExecAction) {
        DifferentialFunction differentialFunction = opExecAction.getOpState().getDifferentialFunction();

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
        allocate();
        if(graph().numVertices() == 0)
            throw new ND4JIllegalStateException("Unable to run exec pipeline. No vertices in graph");


        for(int i = 0; i < ops.size(); i++) {
            Op op = (Op) ops.get(i);
            Nd4j.getExecutioner().exec(op);
        }
        return ops;
    }

    private DifferentialFunction getFunctionInput(SDVariable iX) {
        DifferentialFunction ret =  iX.getDifferentialFunction() != null ?
                iX.getDifferentialFunction() : iX;
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
            context.allocate();
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
            sub.setWorkspace(workspace);
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
            sub.setWorkspace(workspace);
            //setup subgraph
            //re execute to populate subgraph
            functionDefinition.define(sub,inputs, null);

            sameDiffFunctionInstances.put(function,sub);
        }

    }

    /**
     * Exec a given function
     * @param functionName the name of the function
     *                     to invoke
     * @return
     */
    public INDArray execAndEndResult(String functionName) {
        return sameDiffFunctionInstances.get(functionName).execAndEndResult();
    }


    /**
     * Exec a given function
     * @param functionName the name of the function
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
     * @param functionName the name of the function to
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
                    sameDiff.gradientBackwardsMarker(sameDiff.getVertexIdToVariable().get(opOrder.get(0).getOutputId()));

                    //start with scalar backprop
                    DifferentialFunction initialGrad = sameDiff.setupFunction(sameDiff.functionFactory.one(new int[]{1,1}));
                    DifferentialFunction firstBackward = opOrder.get(0).getOpState().getDifferentialFunction();
                    firstBackward.setGradient(initialGrad);
                    SDVariable initialGradVar = SDVariable.builder()
                            .varName("initialgrad")
                            .vertexId(initialGrad.resultVertexId())
                            .sameDiff(sameDiff).arr(Nd4j.scalar(1.0))
                            .shape(initialGrad.getResultShape())
                            .differentialFunction(initialGrad)
                            .build();
                    sameDiff.addVariable(initialGradVar);


                    Set<DifferentialFunction> seen = new HashSet<>();
                    for(OpExecAction action : opOrder) {
                        if(action == null || action.getOpState() == null) {
                            log.warn("Action op state is null");
                            continue;
                        }

                        DifferentialFunction currFunction = action.getOpState().getDifferentialFunction();
                        currFunction.toString();
                        List<DifferentialFunction> backwardResult = currFunction.diff(Arrays.asList(currFunction.getGradient()));

                        //clear out all the variables
                        List<SDVariable> functionVars = debugMode ? new ArrayList<SDVariable>(2) : null;

                        for(int i = 0; i < backwardResult.size(); i++) {
                            DifferentialFunction differentialFunction = backwardResult.get(i);
                            DifferentialFunction x  = sameDiff.setupFunction(currFunction.args()[i]);
                            if(!seen.contains(x)) {
                                seen.add(x);


                                SDVariable forwardVar = sameDiff.getVertexIdToVariable().get(x.resultVertexId());
                                SDVariable add = SDVariable.builder()
                                        .arr(null).differentialFunction(differentialFunction)
                                        .vertexId(differentialFunction.resultVertexId())
                                        .shape(differentialFunction.getResultShape())
                                        .sameDiff(sameDiff)
                                        .varName(forwardVar.getVarName() + "-grad")
                                        .build();

                                sameDiff.addVariable(add);
                                forwardVar.setGradient(add);
                                add.setForwardVariable(forwardVar);


                                if (isDebugMode()) {
                                    if (add.gradient() != null)
                                        sameDiff.addVariable(add.gradient());
                                    functionVars.add(add);
                                }
                            }

                            else {
                                SDVariable forwardVar = sameDiff.getVertexIdToVariable().get(x.resultVertexId());
                                SDVariable grad = forwardVar.gradient();
                                grad.setVertexId(differentialFunction.resultVertexId());
                                grad.setDifferentialFunction(differentialFunction);
                                sameDiff.getVertexIdToVariable().put(differentialFunction.resultVertexId(),grad);
                                Op func = (Op) differentialFunction;
                                grad.setVarName(sameDiff.generateVariableName(func.name(),
                                        true,
                                        differentialFunction));
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


                    return new   SDVariable[] {SDVariable.builder()
                            .differentialFunction(opOrder.get(0).getOpState().getDifferentialFunction())
                            .sameDiff(sameDiff)
                            .varName("grad")
                            .build()};
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
        allocate();
        List<DifferentialFunction> ops = new ArrayList<>();
        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();

        Map<SDVariable,DifferentialFunction> opMap = new HashMap<>();

        boolean onBackward = false;
        for(int i = 0; i < opExecActions.size(); i++) {

            OpExecAction opExecAction = opExecActions.get(i);
            if(!onBackward && opExecAction.getOpState().getOpName().equals(new GradientBackwardsMarker().name())) {
                onBackward = true;
            }

            DifferentialFunction differentialFunction = createOp(
                    opExecAction.getOpState().getOpType(),
                    opExecAction);
            if(differentialFunction instanceof If) {
                If ifOp = (If) differentialFunction;
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

                ops.add(differentialFunction);

            }
            else if(differentialFunction instanceof While) {
                While whileOp = (While) differentialFunction;
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


                SDVariable currVariable = getVertexIdToVariable().get(opExecAction.getOutputId());
                if(currVariable ==  null) {
                    List<SDVariable> functions = new ArrayList<>(opExecAction.getInputsIds().length);
                    SDVariable add = SDVariable.builder()
                            .differentialFunction(opExecAction.getOpState().getDifferentialFunction())
                            .sameDiff(this)
                            .varName(!functions.isEmpty() ? generateVariableName(opExecAction.getOpState().getOpName(),true,
                                    functions.toArray(new SDVariable[functions.size()])) : opExecAction.getOpState().getOpName() + "-" + UUID.randomUUID().toString())
                            .arr(op.z())
                            .shape(op.z().shape())
                            .vertexId(opExecAction.getOutputId())
                            .build();
                    addVariable(add);
                    currVariable = add;

                }
                else

                    currVariable.setArr(op.z());
                opMap.put(currVariable,differentialFunction);
                getVertexToArray().put(opExecAction.getOutput().getVarName(),op.z());
                getFunctionInstances().put(opExecAction.getOutputId(),opExecAction.getOpState().getDifferentialFunction());
            }

        }






        return new Pair<>(opMap,ops);
    }

}
