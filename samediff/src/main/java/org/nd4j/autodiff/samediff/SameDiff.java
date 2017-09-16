package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.Constant;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

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
    private SDGraph graph = new SDGraph();
    private ArrayFactory arrayFactory = new ArrayFactory(this);
    private DifferentialFunctionFactory functionFactory;
    private Map<String,SDVariable> variableMap;
    private Map<Integer,SDVariable> vertexIdToVariable;
    private Map<String,INDArray> vertexToArray;
    private Map<Integer,NDArrayInformation> vertexIdxToInfo;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;
    private Map<Integer,DifferentialFunction> functionInstances;
    private Map<Integer,ArrayField> arrayFieldInstances;
    private static Cloner cloner = new Cloner();
    private static Map<String,Method> opMethods;


    //debug mode variables
    private boolean debugMode;
    private Map<Integer,Op> opsForResult;
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
     * Returns this samediff instances'
     * {@link ArrayFactory}
     * @return
     */
    public ArrayFactory a() {
        return arrayFactory;
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
            NDArrayInformation clone = cloner.deepClone(graph.getVertex(i + 1).getValue());
            if(clone.getOwner() != null && clone.getOwner().getArrayField() != null)
                clone.getOwner().getArrayField().setOps(sameDiff);
            if(clone.getOwner() != null && clone.getOwner().getDifferentialFunction() != null)
                clone.getOwner().getDifferentialFunction().setSameDiff(sameDiff);
            NDArrayVertex info = new NDArrayVertex(
                    sameDiff,
                    nextVertexId,
                    graph.getVertex(i + 1).depth(),
                    clone);
            thisVertexIdToNew.put(graph.getVertex(i + 1).vertexID(),nextVertexId);
            sameDiff.graph().addVertex(info);
        }

        for(Map.Entry<Integer,NDArrayInformation> informationEntry : vertexIdxToInfo.entrySet()) {
            sameDiff.vertexIdxToInfo.put(thisVertexIdToNew.get(informationEntry.getKey()),informationEntry.getValue());
        }


        for(int i = 0; i < graph().numVertices(); i++) {
            /**
             * In this loop also remap the
             * same diff variables
             * with setupFunction,
             */
            List<Edge<OpState>> edgesForVertex = graph.getEdges().get(i + 1);
            List<Edge<OpState>> incomingEdgesForVertex = graph.getIncomingEdges().get(i + 1);
            //map to new vertex
            int newVertexMap = thisVertexIdToNew.get(i + 1);
            if(edgesForVertex != null) {
                List<Edge<OpState>> edgesForNewVertex = new ArrayList<>();
                sameDiff.graph().getEdges().put(newVertexMap, edgesForNewVertex);
                for (Edge<OpState> edge : edgesForVertex) {
                    Preconditions.checkState(thisVertexIdToNew.containsKey(edge.getFrom()),"Edge missing from vertex id for copy " + edge.getFrom());
                    Preconditions.checkState(thisVertexIdToNew.containsKey(edge.getTo()),"Edge missing to vertex id for copy " + edge.getTo());

                    Edge<OpState> newEdge = new Edge<>(
                            thisVertexIdToNew.get(edge.getFrom()),
                            thisVertexIdToNew.get(edge.getTo()),
                            cloner.deepClone(edge.getValue()), true);
                    newEdge.getValue().setVertexIds(new String[]{String.valueOf(newEdge.getFrom()),String.valueOf(newEdge.getTo())});
                    edgesForNewVertex.add(newEdge);

                }
            }

            if(incomingEdgesForVertex != null) {
                List<Edge<OpState>> newIncomingEdges = new ArrayList<>();
                sameDiff.graph().getIncomingEdges().put(newVertexMap,newIncomingEdges);
                for(Edge<OpState> edge : incomingEdgesForVertex) {
                    Edge<OpState> newEdge = new Edge<>(
                            thisVertexIdToNew.get(edge.getFrom()),
                            thisVertexIdToNew.get(edge.getTo()),
                            cloner.deepCloneDontCloneInstances(edge.getValue()),true);
                    newEdge.getValue().setVertexIds(new String[]{String.valueOf(newEdge.getFrom()),String.valueOf(newEdge.getTo())});

                    newIncomingEdges.add(newEdge);

                    if(newEdge.getValue().getArrayField() != null) {
                        newEdge.getValue().getArrayField().setOps(sameDiff);

                    }
                    if(newEdge.getValue().getDifferentialFunction() != null) {
                        ensureSameDiffInstance(sameDiff,newEdge.getValue().getDifferentialFunction());
                        newEdge.getValue().setDifferentialFunction(sameDiff.setupFunction(newEdge.getValue().getDifferentialFunction()));
                        newEdge.getValue().getDifferentialFunction().setVertexId(edge.getValue().getDifferentialFunction().getVertexId());
                    }
                }
            }


            if(arrayFieldInstances.containsKey(i + 1)) {
                ArrayField clone = sameDiff.setupArrayField(cloner.deepClone(arrayFieldInstances.get(i + 1)));
                clone.getVertex().setIdx(newVertexMap);
                sameDiff.arrayFieldInstances.put(newVertexMap,clone);
                ensureSameDiffInstance(sameDiff,clone);
            }

            if(functionInstances.containsKey(i + 1)) {
                DifferentialFunction function = functionInstances.get(i + 1);
                DifferentialFunction clone = sameDiff.setupFunction(cloner.deepClone(function));
                clone.setVertexId(newVertexMap);
                sameDiff.functionInstances.put(newVertexMap,clone);
                ensureSameDiffInstance(sameDiff,clone);
            }



        }



        List<SDVariable> variables = variables();

        //copy over variables
        for(SDVariable variable : variables) {
            SDVariable deepClone = cloner.deepCloneDontCloneInstances(
                    variable,
                    variable.getDifferentialFunction(),
                    variable.getArrayField(),
                    variable.getArr(),
                    variable.getSameDiff(),
                    variable.getInfo(),
                    variable.getShape());
            Preconditions.checkState(thisVertexIdToNew.containsKey(variable.getVertexId()),variable.getVertexId() + " not found in mapped vertices!");
            int newVertexMap = thisVertexIdToNew.get(variable.getVertexId());

            //change the vertex id to the new value
            //for the graph transition
            if(variable.getArrayField() != null) {
                Variable variable1 = (Variable)  sameDiff.functionInstances.get(newVertexMap);
                deepClone.setArrayField(variable1);
            }
            else if(variable.getDifferentialFunction() != null) {
                DifferentialFunction val = sameDiff.functionInstances.get(newVertexMap);
                deepClone.setDifferentialFunction(val);

            }

            deepClone.setVertexId(newVertexMap);
            deepClone.setSameDiff(sameDiff);
            sameDiff.addVariable(deepClone);


        }

        sameDiff.vertexToArray.putAll(vertexToArray);
        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }

    private void ensureSameDiffInstance(SameDiff sameDiff,ArrayField val) {
        val.setOps(sameDiff);
    }

    private void ensureSameDiffInstance(SameDiff sameDiff,DifferentialFunction val) {
        val.setSameDiff(sameDiff);
        if(val instanceof Variable) {
            Variable variable1 = (Variable) val;
            variable1.setSameDiff(sameDiff);
            variable1.setVertexId(val.getVertexId());
            variable1.getM_x().setOps(sameDiff);
            sameDiff.setupFunction(variable1);


        }
        else if(val instanceof Constant) {
            Constant constant = (Constant) val;
            constant.setSameDiff(sameDiff);
            constant.getM_x().setOps(sameDiff);
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
        arrayFactory = new ArrayFactory(this);
        functionFactory = new DifferentialFunctionFactory(this);
        variableMap = new HashMap<>();
        vertexToArray = new HashMap<>();
        vertexIdxToInfo = new HashMap<>();
        sameDiffFunctionDefinitionMap = new HashMap<>();
        sameDiffFunctionInstances = new HashMap<>();
        arrayFieldInstances = new HashMap<>();
        functionInstances = new HashMap<>();
        vertexIdToVariable = new HashMap<>();
        forwardBackwardStates = new HashMap<>();
        opsForResult = new HashMap<>();
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
        int idx = function.getVertexId();
        if(functionInstances.containsKey(idx)) {
            DifferentialFunction get = functionInstances.get(idx);
            //note that we check if the graph is frozen
            //if the graph is frozen this reference is disposable
            if(!graph().isFrozen() && !function.equals(get)) {
                throw new IllegalStateException("Attempted to override Differential Function instance with idx " + idx + " with instance " + function);
            }
            //return the  checked instance
            return (X) get;
        }
        else {
            functionInstances.put(idx,function);
            return function;
        }
    }


    /**
     * Attempts to insert the {@link ArrayField}
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
     * @param arrayField the array field to attempt to create
     * @return
     */
    public ArrayField setupArrayField(ArrayField arrayField) {
        int idx = arrayField.getVertex().getIdx();
        if(arrayFieldInstances.containsKey(idx)) {
            ArrayField get = arrayFieldInstances.get(idx);
            //note that we check if the graph is frozen
            //if the graph is frozen this reference is disposable
            if(!graph().isFrozen() && !arrayField.equals(get)) {
                throw new IllegalStateException("Attempted to override array field instance with idx " + idx + " with instance " + arrayField);
            }
            //return the  checked instance
            return get;
        }
        else {
            arrayFieldInstances.put(idx,arrayField);
            return arrayField;
        }
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
                .vertexIdxToInfo(originalSameDiff.getVertexIdxToInfo())
                .sameDiffFunctionInstances(originalSameDiff.getSameDiffFunctionInstances())
                .vertexToArray(originalSameDiff.getVertexToArray())
                .graph(clone)
                .build();
        //ensuring proper sameDiff reference
        clone.setSameDiff(ret);
        ArrayFactory arrayFactory = new ArrayFactory(ret);
        DifferentialFunctionFactory differentialFunctionFactory =
                new
                        DifferentialFunctionFactory(ret);
        ret.setFunctionFactory(differentialFunctionFactory);
        ret.setArrayFactory(arrayFactory);
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
        if (vertexIdxToInfo != null ? !vertexIdxToInfo.equals(sameDiff.vertexIdxToInfo) : sameDiff.vertexIdxToInfo != null)
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

        List<Op> opExecAction = execPipeline.exec().getRight();
        if(opExecAction.isEmpty())
            throw new IllegalStateException("No ops found to execute.");
        INDArray[] ret = new INDArray[opExecAction.size()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = opExecAction.get(i).z();
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
     *
     */
    public void allocate() {
        if(workspace != null) {
            workspace.close();
        }
        else {
            initWorkspace();
        }


        for (Integer i : graph().getVertices().keySet()) {
            NDArrayInformation info = graph.getInformationFor(i);
            if(!variableMap.containsKey(info.getId())) {
                DifferentialFunction func = functionInstances.get(i);

                SDVariable.SDVariableBuilder variableBuilder = SDVariable.builder()
                        .sameDiff(this)
                        .varName(info.getId());
                //associate the proper differential function with the given
                //variable
                if(func != null && func instanceof Variable) {
                    variableBuilder.arrayField((Variable) func);
                }
                else if(func != null)
                    variableBuilder.differentialFunction(func);

                if(func != null)
                    variableBuilder.shape(func.getResultShape());

                variableBuilder.vertexId(i);

                SDVariable variable = variableBuilder.build();
                variable.setShape(info.getShape());
                variableMap.put(info.getId(),variable);
            }

            /**
             * Problem:
             * Vertexes are not a unique identifier of an actual array.
             * Duplicate vertices are put in to place
             * to avoid cycles by may point at the same array.
             * NDArrayInformation should somehow be unique
             * and point to an actual array.
             */
            if(!vertexToArray.containsKey(info.getArrId())) {
                //initialize value if it's actually a scalar constant (zero or 1 typically...)
                if(info.getScalarValue() != null && ArrayUtil.prod(info.getShape()) == 1) {
                    vertexToArray.put(info.getArrId(), Nd4j.valueArrayOf(info.getShape(),
                            info.getScalarValue().doubleValue()));
                }
                else
                    vertexToArray.put(info.getArrId(), Nd4j.zeros(info.getShape()));

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

        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name)
                .arrId(UUID.randomUUID().toString())
                .build();

        if(ArrayUtil.prod(arr.shape()) == 1)
            ndArrayInformation.setScalarValue(arr.getDouble(0));

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,graph.nextVertexId(), 0,ndArrayInformation);
        graph.addVertex(ndArrayVertex);
        ArrayField arrayField = setupArrayField(new ArrayField(ndArrayVertex,this));
        SDVariable ret = SDVariable.builder()
                .sameDiff(this).
                        arrayField(setupFunction(functionFactory.var(name,arrayField)))
                .shape(arr.shape())
                .varName(name)
                .arr(arr).build();
        addVariable(ret);
        //ensure there is a reference to the array in the integer index
        //this is used later for op creation
        vertexToArray.put(ndArrayInformation.getArrId(),arr);
        vertexIdxToInfo.put(ndArrayVertex.vertexID(),ndArrayInformation);
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
     * {@link NDArrayInformation}
     * @param info the informaton to get the array for
     * @return
     */
    public INDArray getNDArray(NDArrayInformation info) {
        return getVertexToArray().get(info.getArrId());
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
    public SDVariable neq(String name,SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neq(getFunctionInput(iX),iy.getArrayField()))
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
                .differentialFunction(functionFactory.eq(getFunctionInput(iX),iy.getArrayField()))
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
                .differentialFunction(functionFactory.or(getFunctionInput(iX),iy.getArrayField()))
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

    private DifferentialFunction getFunctionInput(String name,SDVariable iX) {
        DifferentialFunction ret =  iX.getDifferentialFunction() != null ?
                iX.getDifferentialFunction() : iX.getArrayField();
        Preconditions.checkState(ret.getSameDiff() == ret.getValue(true).getOps(),"Function input does not have same samediff instance as get value");
        Preconditions.checkState(ret.getSameDiff() == functionFactory.getSameDiff(),"Function input does not have same samediff instance as get value");

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
                .differentialFunction(functionFactory.rollAxis(x.getArrayField(),axis))
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
                .differentialFunction(functionFactory.mmul(x.getArrayField(), y.getArrayField()))
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
                .differentialFunction(functionFactory.tensorMmul(x.getArrayField(), y.getArrayField(), dimensions))
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
                i_y.getArrayField(),
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
                .differentialFunction(functionFactory.euclideanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.manhattanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossBinaryXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossCosineSimilarity(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossKLD(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossL1(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossL2(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossMAE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
     .differentialFunction(functionFactory.lossMAPE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossMSE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossMCXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossMSLE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossPoisson(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(functionFactory.lossSquaredHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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

        if(variable.getArrayField() != null) {
            variable.getArrayField().setSameDiff(this);
        }


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

        vertexIdToVariable.put(getFunctionInput(variable).getVertexId(),variable);
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
     * given the name
     * @param functionName the name of the function
     * @return the same diff function instance
     * defined for the given name
     */
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }


    private INDArray getX(OpExecAction opExecAction) {
        INDArray ret =  vertexToArray.get(opExecAction.getInputs()[0].getArrId());
        return ret;
    }

    private INDArray getY(OpExecAction opExecAction) {
        if(opExecAction.getInputsIds().length > 1) {
            NDArrayInformation opId = opExecAction.getInputs()[1];
            INDArray ret = vertexToArray.get(opId.getArrId());
            return ret;
        }
        return null;
    }

    private INDArray getZ(OpExecAction opExecAction) {
        if(opExecAction.isInPlace())
            return getX(opExecAction);
        NDArrayInformation opId = opExecAction.getOutput();
        INDArray ret =  vertexToArray.get(opId.getArrId());
        return ret;
    }


    /**
     *
     * @param opType
     * @param opExecAction
     * @return
     */
    public Op createOp(OpState.OpType opType,
                       OpExecAction opExecAction) {
        OpState opState = opExecAction.getOpState();
        switch (opType) {
            case GRADIENT:
                return Nd4j.getOpFactory().createGradientOp(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction));
            case SHAPE:
                return Nd4j.getOpFactory().createShape(
                        opState.getOpName(),
                        getX(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs());
            case SCALAR_TRANSFORM:
                return Nd4j.getOpFactory().createScalarTransform(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs(),
                        opState.getScalarValue().doubleValue());
            case ACCUMULATION:
                return Nd4j.getOpFactory().createAccum(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs());
            case TRANSFORM:
                return Nd4j.getOpFactory().createTransform(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs());
            case BROADCAST:
                return Nd4j.getOpFactory().createBroadcastOp(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs());

            case INDEX_ACCUMULATION:
                return Nd4j.getOpFactory().createIndexAccum(
                        opState.getOpName(),
                        getX(opExecAction),
                        getY(opExecAction),
                        getZ(opExecAction),
                        opState.getExtraArgs());
            case AGGREGATE: break;
        }

        throw new IllegalStateException("Illegal type specified " + opType);
    }

    /**
     *u
     * @return
     */
    public INDArray execAndEndResult(List<Op> ops) {
        List<Op> exec = exec(ops);
        return exec.get(exec.size() - 1).z();
    }

    /**
     *
     * @return
     */
    public INDArray execAndEndResult() {
        List<Op> exec = exec().getRight();
        return exec.get(exec.size() - 1).z();
    }


    /**
     * Executes the list of operations.
     * This exec method is for
     * only invoking operations
     * rather than creating them
     * @param ops the list of already created ops
     * @return the passes in list
     */
    public List<Op> exec(List<Op> ops) {
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
            Op op = ops.get(i);
            Nd4j.getExecutioner().exec(op);
        }
        return ops;
    }

    private DifferentialFunction getFunctionInput(SDVariable iX) {
        DifferentialFunction ret =  iX.getDifferentialFunction() != null ?
                iX.getDifferentialFunction() : iX.getArrayField();
        Preconditions.checkState(iX.getSameDiff() != null,"Samediff instance must not be null.");
        if(graph().getGraphApply() == null) {
            Preconditions.checkState(ret.getSameDiff() == ret.getValue(true).getOps(), "Function input does not have same samediff instance as get value");
            Preconditions.checkState(ret.getSameDiff() == functionFactory.getSameDiff(), "Function input does not have same samediff instance as get value");
        }
        return ret;
    }


    public interface SameDiffFunctionDefinition {

        /**
         *
         * @param inputs
         * @return
         */
        SDVariable define(SameDiff sameDiff, Map<String, INDArray> inputs);
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
    public void defineFunction(String function,SameDiffFunctionDefinition functionDefinition) {
        defineFunction(function,functionDefinition,null);
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
            functionDefinition.define(sub,inputs);

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
    public Pair<Map<SDVariable, Op>, List<Op>> exec(String functionName) {
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
    public List<Op> exec(String functionName,List<Op> cachedOps) {
        return sameDiffFunctionInstances.get(functionName).exec(cachedOps);
    }


    /**
     * Builds a backwards graph
     * and executes the operations
     * on that graph.
     * @return
     */
    public Pair<Map<SDVariable, Op>, List<Op>> execBackwards() {
        SameDiff outer = this;
        if(getFunction("grad") == null)
            defineFunction("grad", new SameDiffFunctionDefinition() {

                @Override
                public SDVariable define(SameDiff sameDiff, Map<String, INDArray> inputs) {
                    //propagate graph to this samediff instance
                    //which wil also contain the backward
                    if(SameDiff.this.debugMode) {
                        sameDiff.enableDebugMode();
                    }

                    outer.invokeGraphOn(sameDiff);
                    List<OpExecAction> opOrder = sameDiff.graph().getOpOrder(true).getActions();
                    List<OpExecAction> exec = new ArrayList<>();
                    sameDiff.gradientBackwardsMarker(sameDiff.getVertexIdToVariable().get(opOrder.get(opOrder.size() - 1).getOutputId()));

                    //start with scalar backprop
                    List<DifferentialFunction> currentDiff = Arrays.asList(sameDiff.functionFactory.one(new int[]{1,1}));
                    boolean firstGradientSet = false;

                    for(OpExecAction action : opOrder) {
                        if(action == null || action.getOpState() == null) {
                            log.warn("Action op state is null");
                            continue;
                        }

                        DifferentialFunction currFunction = action.getOpState().getDifferentialFunction();
                        if(!firstGradientSet) {
                            firstGradientSet = true;
                            currFunction.setGradient(currentDiff.get(0));
                            SDVariable initialGrad = SDVariable.builder()
                                    .varName("initialgrad")
                                    .vertexId(currentDiff.get(0).resultVertexId())
                                    .sameDiff(sameDiff).arr(Nd4j.scalar(1.0))
                                    .shape(currentDiff.get(0).getResultShape())
                                    .differentialFunction(currentDiff.get(0))
                                    .build();
                            sameDiff.addVariable(initialGrad);
                        }

                        currentDiff = currFunction.diff(currentDiff);

                        //clear out all the variables
                        List<SDVariable> functionVars = debugMode ? new ArrayList<>(2) : null;

                        for(DifferentialFunction differentialFunction : currentDiff) {
                            SDVariable add = SDVariable.builder()
                                    .arr(null).differentialFunction(differentialFunction)
                                    .vertexId(differentialFunction.resultVertexId())
                                    .shape(differentialFunction.getResultShape())
                                    .sameDiff(sameDiff)
                                    .varName(sameDiff.generateVariableName(differentialFunction.functionName(),
                                            true,
                                            differentialFunction))
                                    .build();

                            sameDiff.addVariable(add);
                            SDVariable forwardVar = sameDiff.getVertexIdToVariable().get(action.getOutputId());
                            add.setForwardVariable(forwardVar);


                            if (isDebugMode()) {
                                if(add.gradient() != null)
                                    sameDiff.addVariable(add.gradient());
                                functionVars.add(add);
                            }
                        }

                        if(isDebugMode()) {
                            exec.add(action);
                        }

                    }



                    /**
                     * Op order is wrong here.
                     * There is an edge case with logistic regression where
                     * it executes sigmoid in the wrong order
                     * when it encounters 1- activation.
                     *
                     * What it *should* do is when it sees sigmoid
                     * to activate that operation first
                     * before the 1- activation
                     * to ensure the proper gradient exists.
                     *
                     * The other frameworks dynamically find
                     * the right dependencies
                     * and add them as necessary.
                     * We need to ensure our sort mechanism does the same.
                     *
                     * The debugging situation to watch is
                     * the:
                     * cand_funcs
                     *
                     * which contains the function execution order.
                     * The function execution order for our case should be:
                     * 1 - output -> sigmoid -> 1- predictions
                     *
                     * Right now, it is 1 - predictions -> sigmoid.
                     *
                     * Another thing to look in to is whether output_grad
                     * propagates properly as well.
                     *
                     * The goal here should be to build a minm aal test
                     * that reproduces this wrong behavior
                     * for transitive dependencies.
                     */



                    if(sameDiff.isDebugMode()) {
                        //ensure all gradients are present for all variables
                        for(SDVariable sdVariable : variables()) {
                            sdVariable.gradient();
                        }
                    }


                    return SDVariable.builder()
                            .differentialFunction(currentDiff.get(0))
                            .sameDiff(sameDiff)
                            .varName("grad")
                            .build();
                }
            });


        Pair<Map<SDVariable, Op>, List<Op>> forward = exec("grad");
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
        List<Op> backwards = execBackwards().getRight();
        return backwards.get(backwards.size() - 1).z();
    }




    /**
     * Creates and executes a list of operations
     * @return
     */
    public Pair<Map<SDVariable,Op>,List<Op>> exec() {
        allocate();
        List<Op> ops = new ArrayList<>();
        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();

        Map<SDVariable,Op> opMap = new HashMap<>();

        boolean onBackward = false;
        for(int i = 0; i < opExecActions.size(); i++) {

            OpExecAction opExecAction = opExecActions.get(i);
            SDVariable variable = null;
            Op forwardOp = null;
            if(onBackward && getVertexIdToVariable().get(opExecAction.getOutputId()) != null) {
                variable = getVertexIdToVariable().get(opExecAction.getOutputId()).getForwardVariable();
                forwardOp = opMap.get(variable);
                System.out.println(forwardOp);

            }

            if(!onBackward && opExecAction.getOpState().getOpName().equals(new GradientBackwardsMarker().name())) {
                onBackward = true;
            }

            Op op = createOp(
                    opExecAction.getOpState().getOpType(),
                    opExecAction);

            if(debugMode) {
                opsForResult.put(opExecAction.getOutputId(),op);
            }

            ops.add(op);

            if(opExecAction.getOpState().getAxes() == null)
                Nd4j.getExecutioner().exec(op);

            else {
                int[] axes = opExecAction.getOpState().getAxes();
                if(op instanceof Accumulation) {
                    Accumulation accumulation = (Accumulation) op;
                    Nd4j.getExecutioner().exec(accumulation,axes);

                }

                else if(op instanceof BroadcastOp) {
                    BroadcastOp broadcastOp = (BroadcastOp) op;
                    Nd4j.getExecutioner().exec(broadcastOp,axes);
                }
                else if(op instanceof GradientOp) {
                    Nd4j.getExecutioner().exec(op);
                }
                else if(op instanceof IndexAccumulation) {
                    IndexAccumulation indexAccumulation = (IndexAccumulation) op;
                    Nd4j.getExecutioner().exec(indexAccumulation,axes);

                }
            }

            SDVariable currVariable = getVertexIdToVariable().get(opExecAction.getOutputId());
            if(currVariable ==  null) {
                List<SDVariable> functions = new ArrayList<>(opExecAction.getInputsIds().length);
                SDVariable add = SDVariable.builder()
                        .differentialFunction(opExecAction.getOpState().getDifferentialFunction())
                        .sameDiff(this)
                        .varName(generateVariableName(opExecAction.getOpState().getOpName(),true,
                                functions.toArray(new SDVariable[functions.size()])))
                        .arr(op.z())
                        .shape(op.z().shape())
                        .vertexId(opExecAction.getOutputId())
                        .build();
                addVariable(add);
                currVariable = add;

            }
            else

                currVariable.setArr(op.z());
            opMap.put(currVariable,op);
        }



        return new Pair<>(opMap,ops);
    }

}
