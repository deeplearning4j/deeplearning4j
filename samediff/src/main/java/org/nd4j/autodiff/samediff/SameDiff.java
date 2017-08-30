package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
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
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
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
public class SameDiff {
    private SDGraph graph = new SDGraph();
    private ArrayFactory arrayFactory = new ArrayFactory(this);
    private DifferentialFunctionFactory<ArrayField> functionFactory;
    private List<SDVariable> sameDiffVariables = new ArrayList<>();
    private Map<String,SDVariable> variableMap;
    private Map<String,INDArray> vertexToArray;
    private Map<Integer,NDArrayInformation> vertexIdxToInfo;
    private MemoryWorkspace workspace;
    private Map<String,SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String,SameDiff> sameDiffFunctionInstances;
    private Map<Integer,DifferentialFunction<ArrayField>> functionInstances;
    private Map<Integer,ArrayField> arrayFieldInstances;
    private static Cloner cloner = new Cloner();

    private static Map<String,Method> opMethods;

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

                    if(newEdge.getValue().getArrayField() != null)
                        newEdge.getValue().getArrayField().setOps(sameDiff);

                    if(newEdge.getValue().getDifferentialFunction() != null) {
                        ensureSameDiffInstance(sameDiff,newEdge.getValue().getDifferentialFunction());
                        newEdge.getValue().setDifferentialFunction(sameDiff.setupFunction(newEdge.getValue().getDifferentialFunction()));
                    }
                }
            }


            if(arrayFieldInstances.containsKey(i + 1)) {
                ArrayField clone = sameDiff.setupArrayField(cloner.deepClone(arrayFieldInstances.get(i + 1)));
                clone.getVertex().setIdx(newVertexMap);
                sameDiff.arrayFieldInstances.put(newVertexMap,clone);
            }

            if(!functionInstances.containsKey(i + 1)) {
                DifferentialFunction<ArrayField> function = functionInstances.get(i + 1);
                DifferentialFunction<ArrayField> clone = sameDiff.setupFunction(cloner.deepClone(function));
                clone.setVertexId(newVertexMap);
                sameDiff.functionInstances.put(newVertexMap,clone);
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
            int newVertexMap = thisVertexIdToNew.get(variable.getVertexId());

            //change the vertex id to the new value
            //for the graph transition
            if(variable.getArrayField() != null) {
                Variable<ArrayField> variable1 = deepClone.getArrayField();
                deepClone.setArrayField((Variable<ArrayField>) sameDiff.functionInstances.get(newVertexMap));
                ensureSameDiffInstance(sameDiff,variable1);
            }
            else if(variable.getDifferentialFunction() != null) {
                DifferentialFunction<ArrayField> val = deepClone.getDifferentialFunction();
                deepClone.setDifferentialFunction(sameDiff.functionInstances.get(newVertexMap));
                ensureSameDiffInstance(sameDiff,val);

            }

            deepClone.setVertexId(newVertexMap);
            deepClone.setSameDiff(sameDiff);
            sameDiff.addVariable(deepClone);


        }

        sameDiff.vertexToArray.putAll(vertexToArray);
        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }


    private void ensureSameDiffInstance(SameDiff sameDiff,DifferentialFunction<ArrayField> val) {
        val.setSameDiff(sameDiff);
        if(val instanceof Variable) {
            Variable<ArrayField> variable1 = (Variable<ArrayField>) val;
            variable1.setSameDiff(sameDiff);
            variable1.setVertexId(val.getVertexId());
            variable1.getM_x().setOps(sameDiff);
            sameDiff.setupFunction(variable1);


        }
        else if(val instanceof Constant) {
            Constant<ArrayField> constant = (Constant<ArrayField>) val;
            constant.setSameDiff(sameDiff);
            constant.getM_x().setOps(sameDiff);
            sameDiff.setupFunction(constant);
        }

        //recursive case
        else if(val.args() != null) {
            for(DifferentialFunction<ArrayField> equation  : val.args()) {
                sameDiff.setupFunction(equation);
                ensureSameDiffInstance(sameDiff,equation);

            }
        }

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
        functionFactory = new DifferentialFunctionFactory<>(this);
        sameDiffVariables = new ArrayList<>();
        variableMap = new HashMap<>();
        vertexToArray = new HashMap<>();
        vertexIdxToInfo = new HashMap<>();
        sameDiffFunctionDefinitionMap = new HashMap<>();
        sameDiffFunctionInstances = new HashMap<>();
        arrayFieldInstances = new HashMap<>();
        functionInstances = new HashMap<>();
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
    public <X extends DifferentialFunction<ArrayField>> X setupFunction(X  function) {
        int idx = function.getVertexId();
        if(functionInstances.containsKey(idx)) {
            DifferentialFunction<ArrayField> get = functionInstances.get(idx);
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
        return graph;
    }



    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (graph != null ? graph.hashCode() : 0);
        result = 31 * result + (sameDiffVariables != null ? sameDiffVariables.hashCode() : 0);
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
                .sameDiffVariables(originalSameDiff.getSameDiffVariables())
                .vertexIdxToInfo(originalSameDiff.getVertexIdxToInfo())
                .sameDiffFunctionInstances(originalSameDiff.getSameDiffFunctionInstances())
                .vertexToArray(originalSameDiff.getVertexToArray())
                .graph(clone)
                .build();
        //ensuring proper sameDiff reference
        clone.setSameDiff(ret);
        ArrayFactory arrayFactory = new ArrayFactory(ret);
        DifferentialFunctionFactory<ArrayField> differentialFunctionFactory =
                new
                        DifferentialFunctionFactory<>(ret);
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
        if (sameDiffVariables != null ? !sameDiffVariables.equals(sameDiff.sameDiffVariables) : sameDiff.sameDiffVariables != null)
            return false;
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
     * Set the ndarray for the given value
     * @param value
     * @param arr
     */
    public void updateNDArray(String value,INDArray arr) {
        if(!variableMap.containsKey(value))
            throw new IllegalArgumentException("Illegal key specified vor variable " + value);
        if(!Arrays.equals(arr.shape(),variableMap.get(value).getShape()))
            throw new IllegalArgumentException("Illegal array specified must be of shape " + Arrays.toString(variableMap.get(value).getShape()));
        getVariableMap().get(value).setArr(arr);
    }


    /**
     * Evaluate the given inputs
     * based on the current graph
     * @param inputs the inputs to evaluate
     * @return
     */
    public INDArray[] eval(Map<String,INDArray> inputs) {

        SameDiff execPipeline = dup();

        List<Op> opExecAction = execPipeline.exec();
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


        if(sameDiffVariables == null)
            sameDiffVariables = new ArrayList<>();

        for (Integer i : graph().getVertices().keySet()) {
            NDArrayInformation info = graph.getInformationFor(i);
            if(!variableMap.containsKey(info.getId())) {
                SDVariable variable = SDVariable.builder()
                        .sameDiff(this)
                        .varName(info.getId())
                        .build();
                variable.setShape(info.getShape());
                sameDiffVariables.add(variable);
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
        return sameDiffVariables;
    }

    /**
     *
     *
     * @param name
     * @param arr
     * @return
     */
    public SDVariable var(String name, INDArray arr) {
        if(variableMap.containsKey(name))
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

        NDArrayVertex ndArrayVertex = new NDArrayVertex(this,graph.nextVertexId(), ndArrayInformation);
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
    public SDVariable grad(SDVariable iX, SDVariable wrt) {
        Preconditions.checkState(iX.getSameDiff() == wrt.getSameDiff(),"Same diff instances must be the same.");
        Preconditions.checkArgument(getFunctionInput(iX).getSameDiff() == this);
        Preconditions.checkArgument(getFunctionInput(wrt).getSameDiff() == this);

        List<DifferentialFunction<ArrayField>> arrField = getFunctionInput(iX).diff(Arrays.asList(getFunctionInput(wrt)));
        Preconditions.checkArgument(arrField.get(0).getSameDiff() == this);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrField.get(0).getResultShape())
                .differentialFunction(arrField.get(0))
                .varName("grad(" + iX.getVarName() + ")").sameDiff(this)
                .build();
        //Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neq(getFunctionInput(iX),iy.getArrayField()))
                .varName("neq(" + iX.getVarName() + "," + iy.getVarName() + ")").sameDiff(this)
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
    public SDVariable eq(SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.eq(getFunctionInput(iX),iy.getArrayField()))
                .varName("eq(" + iX.getVarName() + "," + iy.getVarName() + ")").sameDiff(this)
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
    public SDVariable or(SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.or(getFunctionInput(iX),iy.getArrayField()))
                .varName("or(" + iX.getVarName() + "," + iy.getVarName() + ")").sameDiff(this)
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
    public SDVariable neg(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.neg(getFunctionInput(iX)))
                .varName("neg(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable cos(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.cos(getFunctionInput(iX)))
                .varName("cos(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sin(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.sin(getFunctionInput(iX)))
                .varName("sin(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable tan(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.tan(getFunctionInput(iX)))
                .varName("tan(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable acos(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.acos(getFunctionInput(iX)))
                .varName("acos(" + iX.getVarName() + ")").sameDiff(this)
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

    public SDVariable asin(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.asin(getFunctionInput(iX)))
                .varName("asin(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable atan(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.atan(getFunctionInput(iX)))
                .varName("atan(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable cosh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.cosh(getFunctionInput(iX)))
                .varName("cosh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sinh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.sinh(getFunctionInput(iX)))
                .varName("sinh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable tanh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.tanh(getFunctionInput(iX)))
                .varName("tanh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable acosh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.acosh(getFunctionInput(iX)))
                .varName("acosh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable asinh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.asinh(getFunctionInput(iX)))
                .varName("asinh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable atanh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.atanh(getFunctionInput(iX)))
                .varName("atanh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable exp(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.exp(getFunctionInput(iX)))
                .varName("exp(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable log(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.log(getFunctionInput(iX)))
                .varName("log(" + iX.getVarName() + ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @return
     */
    public SDVariable pow(SDVariable iX, SDVariable i_y) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.pow(getFunctionInput(iX),null))
                .varName("pow(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sqrt(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.sqrt(getFunctionInput(iX)))
                .varName("sqrt(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable square(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.square(getFunctionInput(iX)))
                .varName("square(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable floor(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.floor(getFunctionInput(iX)))
                .varName("floor(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable relu(SDVariable iX,double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.relu(getFunctionInput(iX),cutoff))
                .varName("relu(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable softmax(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softmax(getFunctionInput(iX)))
                .varName("softmax(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable softmaxDerivative(SDVariable iX,SDVariable wrt) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softmaxDerivative(getFunctionInput(iX),getFunctionInput(wrt)))
                .varName("softmaxderivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable hardTanh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.hardTanh(getFunctionInput(iX)))
                .varName("hardTanh(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable hardTanhDerivative(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.hardTanhDerivative(getFunctionInput(iX)))
                .varName("hardTanhDerivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sigmoid(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.sigmoid(getFunctionInput(iX)))
                .varName("sigmoid(" + iX.getVarName() + ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    private DifferentialFunction<ArrayField> getFunctionInput(SDVariable iX) {
        DifferentialFunction<ArrayField> ret =  iX.getDifferentialFunction() != null ?
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
    public SDVariable sigmoidDerivative(SDVariable iX,SDVariable wrt) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory
                        .sigmoidDerivative(getFunctionInput(iX), getFunctionInput(wrt)))
                .varName("sigmoidDerivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sign(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory
                        .sign(getFunctionInput(iX))).differentialFunction(functionFactory
                        .sign(iX.getDifferentialFunction()))
                .varName("sign(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable softsign(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softsign(getFunctionInput(iX)))
                .varName("softsign(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable softsignDerivative(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softsignDerivative(getFunctionInput(iX)))
                .varName("softsignDerivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable softplus(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.softplus(getFunctionInput(iX)))
                .varName("softplus(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable elu(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.elu(getFunctionInput(iX)))
                .varName("elu(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable eluDerivative(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.eluDerivative(getFunctionInput(iX)))
                .varName("eluDerivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(functionFactory.leakyRelu(getFunctionInput(iX),cutoff))
                .varName("leakyRelu(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable leakyReluDerivative(SDVariable iX, SDVariable wrt,double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.leakyReluDerivative(getFunctionInput(iX),
                        getFunctionInput(wrt),
                        cutoff))
                .varName("leakyReluDerivative(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable mean(SDVariable iX) {

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.mean(getFunctionInput(iX)))
                .varName("mean(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable standardDeviation(SDVariable iX,
                                        boolean biasCorrected,
                                        int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.std(
                        getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable variance(SDVariable iX,
                               boolean biasCorrected,
                               int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.variance(getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable sum(SDVariable iX,
                          int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.sum(getFunctionInput(iX),dimensions))
                .varName("sum(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable prod(SDVariable iX,
                           int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.prod(getFunctionInput(iX),dimensions))
                .varName("prod(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable max(SDVariable iX, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.max(getFunctionInput(iX),dimensions))
                .varName("max(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable min(SDVariable iX,
                          int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.min(getFunctionInput(iX),dimensions))
                .varName("min(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable reshape(SDVariable iX,
                              int...shape) {
        shape = Shape.resolveNegativeShapeIfNeccessary(shape,iX.getShape());

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory
                        .reshape(getFunctionInput(iX),shape))
                .varName("reshape(" + iX.getVarName() + ")").sameDiff(this)
                .shape(shape)
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
    public SDVariable transpose(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.transpose(getFunctionInput(iX)))
                .varName("transpose(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable rollAxis(SDVariable x, int axis) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.rollAxis(x.getArrayField(),axis))
                .varName("rollAxis(" + x.getVarName() + ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param argNum
     * @param x
     * @param y
     * @return
     */
    public SDVariable mmul(int argNum, SDVariable x, SDVariable y) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.mmul(x.getArrayField(), y.getArrayField()))
                .varName("mmul(" + x.getVarName() + "," + y.getVarName()  + ")").sameDiff(this)
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
     * @param argNum
     * @return
     */
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions,
                                 int argNum) {

        int[] shape = ArrayUtil.getTensorMmulShape(x.getShape(), y.getShape(), dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(functionFactory.tensorMmul(x.getArrayField(), y.getArrayField(), dimensions))
                .varName("tensorMmul(" + x.getVarName() + "," + y.getVarName() +  ")")
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
    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int...dimensions) {
        DifferentialFunction<ArrayField> cosim = functionFactory.cosineSimilarity(
                getFunctionInput(iX),
                i_y.getArrayField(),
                dimensions);

        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(cosim)
                .varName("cosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")")
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
    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.euclideanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("euclideanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.manhattanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("manhattanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossBinaryXENT(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossBinaryXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossBinaryXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossCosineSimilarity(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossCosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossKLD(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossKLD(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossL1(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossL1(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossL2(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossL2(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMAE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMAE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    /**
     public SDVariable lossMAPE(SDVariable iX,SDVariable i_y,int...dimensions) {
     int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

     SDVariable ret = SDVariable.builder()
     .arr(null).shape(arrayReduceShape)
     .differentialFunction(functionFactory.lossMAPE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
     .varName("lossMAPE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMSE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMSE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMCXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMCXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossMSLE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMSLE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.transpose(getFunctionInput(iX)))
                .varName("lossNegativeLogLikelihood(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossPoisson(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossPoisson(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(functionFactory.lossSquaredHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossSquaredHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
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
        if(sameDiffVariables == null)
            sameDiffVariables = new ArrayList<>();
        if(variableMap == null)
            variableMap = new HashMap<>();

        if(variable.getArrayField() != null) {
            variable.getArrayField().setSameDiff(this);
        }


        sameDiffVariables.add(variable);
        variableMap.put(variable.getVarName(),variable);
    }

    private boolean isInPlace(Object[] extraArgs) {
        if(extraArgs == null)
            return false;
        for(int i = 0; i < extraArgs.length; i++) {
            if(extraArgs[i] instanceof Boolean) {
                return (Boolean) extraArgs[i];
            }
        }

        return false;
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
        List<Op> exec = exec();
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
    public List<Op> exec(String functionName) {
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
    public List<Op> execBackwards() {
        SameDiff outer = this;
        if(getFunction("grad") == null)
            defineFunction("grad", new SameDiffFunctionDefinition() {

                @Override
                public SDVariable define(SameDiff sameDiff, Map<String, INDArray> inputs) {
                    //propagate graph to this samediff instance
                    //which wil also contain the backward
                    outer.invokeGraphOn(sameDiff);
                    List<OpExecAction> opOrder = sameDiff.graph().getOpOrder().getActions();
                    Collections.reverse(opOrder);
                    //start with scalar backprop
                    List<DifferentialFunction<ArrayField>> currentDiff = Arrays.asList(sameDiff.functionFactory.one(new int[]{1,1}));

                    for(OpExecAction action : opOrder) {
                        if(action.getOpState() != null) {
                            DifferentialFunction<ArrayField> func = action.getOpState().getDifferentialFunction();
                            if(func != null) {
                                currentDiff = func.diff(currentDiff);
                            }
                            else if(action.getOpState().getArrayField() != null) {

                            }
                        }
                    }

                    return SDVariable.builder()
                            .differentialFunction(currentDiff.get(0))
                            .sameDiff(sameDiff)
                            .varName("grad")
                            .build();
                }
            });


        List<Op> forward = exec("grad");
        return forward;
    }


    /**
     * Exec a backwards operation
     * and return the end result
     * @return
     */
    public INDArray execBackwardAndEndResult() {
        List<Op> backwards = execBackwards();
        return backwards.get(backwards.size() - 1).z();
    }

    /**
     * Creates and executes a list of operations
     * @return
     */
    public List<Op> exec() {
        allocate();
        List<Op> ops = new ArrayList<>();
        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();
        for(int i = 0; i < opExecActions.size(); i++) {
            OpExecAction opExecAction = opExecActions.get(i);
            Op op = createOp(
                    opExecAction.getOpState().getOpType(),
                    opExecAction);

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

        }



        return ops;
    }

}
