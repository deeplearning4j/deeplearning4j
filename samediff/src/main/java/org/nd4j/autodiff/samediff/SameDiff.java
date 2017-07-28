package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

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
    private ArrayFactory arrayFactory = new ArrayFactory(graph);
    private DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory;
    private List<SDVariable> sameDiffVariables = new ArrayList<>();
    private Map<String,SDVariable> variableMap;
    private Map<String,INDArray> vertexToArray;
    private Map<Integer,NDArrayInformation> vertexIdxToInfo;

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
        arrayFactory = new ArrayFactory(graph);
        arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        sameDiffVariables = new ArrayList<>();
        variableMap = new HashMap<>();
        vertexToArray = new HashMap<>();
        vertexIdxToInfo = new HashMap<>();
    }




    /**
     * The same diff graph
     * @return
     */
    public SDGraph graph() {
        return graph;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SameDiff that = (SameDiff) o;

        if (graph != null ? !graph.equals(that.graph) : that.graph != null) return false;
        if (sameDiffVariables != null ? !sameDiffVariables.equals(that.sameDiffVariables) : that.sameDiffVariables != null)
            return false;
        return variableMap != null ? variableMap.equals(that.variableMap) : that.variableMap == null;
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
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        SDGraph clone = new SDGraph(graph);
        SameDiff ret = SameDiff.builder()
                .variableMap(originalSameDiff.getVariableMap())
                .arrayFactory(arrayFactory)
                .sameDiffVariables(originalSameDiff.getSameDiffVariables())
                .arrayFieldDifferentialFunctionFactory(new DifferentialFunctionFactory<>(graph,arrayFactory))
                .graph(clone)
                .build();
        //ensuring proper sameDiff reference
        clone.setSameDiff(ret);

        return ret;
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
        for(String s : inputs.keySet()) {
            if(!variableMap.containsKey(s))
                throw new IllegalArgumentException("Illegal key for variables " + s);
        }

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
                vertexToArray.put(info.getArrId(), Nd4j.zeros(info.getShape()));

            }
        }

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
        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name).arrId(UUID.randomUUID().toString())
                .build();
        if(ArrayUtil.prod(arr.shape()) == 1)
            ndArrayInformation.setScalarValue(arr.getDouble(0));
        NDArrayVertex ndArrayVertex = new NDArrayVertex(graph.nextVertexId(), ndArrayInformation);
        ArrayField arrayField = new ArrayField(ndArrayVertex,graph);
        SDVariable ret = SDVariable.builder()
                .sameDiff(this).
                        arrayField(arrayFieldDifferentialFunctionFactory.var(name,arrayField))
                .shape(arr.shape())
                .varName(name)
                .arr(arr).build();
        addVariable(ret);
        //ensure there is a reference to the array in the integer index
        //this is used later for op creation
        vertexToArray.put(ndArrayInformation.getArrId(),arr);
        vertexIdxToInfo.put(ndArrayVertex.vertexID(),ndArrayInformation);
        return ret;

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
        DifferentialFunction<ArrayField> arrField = getFunctionInput(iX).diff(wrt.getArrayField());
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(wrt.getShape())
                .differentialFunction(arrField)
                .varName("grad(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.neq(getFunctionInput(iX),iy.getArrayField()))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eq(getFunctionInput(iX),iy.getArrayField()))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.or(getFunctionInput(iX),iy.getArrayField()))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.neg(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cos(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sin(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tan(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acos(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asin(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atan(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cosh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sinh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tanh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acosh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asinh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atanh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.exp(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.log(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.pow(getFunctionInput(iX),null))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sqrt(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.square(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.floor(getFunctionInput(iX)))
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
    public SDVariable relu(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.relu(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softmax(getFunctionInput(iX)))
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
    public SDVariable hardTanh(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanh(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanhDerivative(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoid(getFunctionInput(iX)))
                .varName("sigmoid(" + iX.getVarName() + ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    private DifferentialFunction<ArrayField> getFunctionInput(SDVariable iX) {
        return iX.getDifferentialFunction() != null ?
                iX.getDifferentialFunction() : iX.getArrayField();
    }

    /**
     *
     * @param iX
     * @return
     */
    public SDVariable sigmoidDerivative(SDVariable iX) {
        SDVariable ret = SDVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory
                        .sigmoidDerivative(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory
                        .sign(getFunctionInput(iX))).differentialFunction(arrayFieldDifferentialFunctionFactory
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsign(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsignDerivative(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softplus(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.elu(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eluDerivative(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyRelu(getFunctionInput(iX),cutoff))
                .varName("leakyRelu(" + iX.getVarName() + ")").sameDiff(this)
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
    public SDVariable leakyReluDerivative(SDVariable iX, double cutoff) {
        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyReluDerivative(getFunctionInput(iX),cutoff))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mean(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.std(
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.variance(getFunctionInput(iX),
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sum(getFunctionInput(iX),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.prod(getFunctionInput(iX),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.max(getFunctionInput(iX),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.min(getFunctionInput(iX),dimensions))
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
        shape = Shape.resolveNegativeShapeIfNeccessary(shape);

        SDVariable ret = SDVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.rollAxis(x.getArrayField(),axis))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mmul(argNum ,x.getArrayField(), y.getArrayField()))
                .varName("mmul(" + x.getVarName() + "," + y.getVarName()  + ")").sameDiff(this)
                .build();
        ret.setShape(ret.getDifferentialFunction().getOpState().getResult().getShape());
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tensorMmul(x.getArrayField(), y.getArrayField(), dimensions, argNum))
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
        DifferentialFunction<ArrayField> cosim = arrayFieldDifferentialFunctionFactory.cosineSimilarity(
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
    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SDVariable ret = SDVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.euclideanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.manhattanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossBinaryXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossCosineSimilarity(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossKLD(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL1(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL2(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
     .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAPE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMCXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSLE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(getFunctionInput(iX)))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossPoisson(getFunctionInput(iX),i_y.getArrayField(),dimensions))
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
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossSquaredHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossSquaredHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").sameDiff(this)
                .build();
        Preconditions.checkState(Arrays.equals(ret.getShape(),ret.getDifferentialFunction().getResultShape()));
        addVariable(ret);
        return ret;
    }

    private void addVariable(SDVariable variable) {
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
            case SHAPE:
                return Nd4j.getOpFactory().createShape(
                        opState.getOpName(),
                        getX(opExecAction),
                        getZ(opExecAction)
                );
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
        }

        return exec(ops);
    }

}
