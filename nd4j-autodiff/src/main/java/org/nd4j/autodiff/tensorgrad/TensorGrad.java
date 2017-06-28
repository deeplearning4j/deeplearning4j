package org.nd4j.autodiff.tensorgrad;

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
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Created by agibsonccc on 4/9/17.
 */
@AllArgsConstructor
@Data
@Builder
public class TensorGrad {
    private TensorGradGraph graph = new TensorGradGraph();
    private ArrayFactory arrayFactory = new ArrayFactory(graph);
    private DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory;
    private List<TensorGradVariable> tensorGradVariables = new ArrayList<>();
    private Map<String,TensorGradVariable> variableMap;

    private TensorGrad() {
        graph = new TensorGradGraph();
        graph.setTensorGrad(this);
        arrayFactory = new ArrayFactory(graph);
        arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        tensorGradVariables = new ArrayList<>();
        variableMap = new HashMap<>();
    }

    public TensorGradGraph graph() {
        return graph;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TensorGrad that = (TensorGrad) o;

        if (graph != null ? !graph.equals(that.graph) : that.graph != null) return false;
        if (tensorGradVariables != null ? !tensorGradVariables.equals(that.tensorGradVariables) : that.tensorGradVariables != null)
            return false;
        return variableMap != null ? variableMap.equals(that.variableMap) : that.variableMap == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (graph != null ? graph.hashCode() : 0);
        result = 31 * result + (tensorGradVariables != null ? tensorGradVariables.hashCode() : 0);
        result = 31 * result + (variableMap != null ? variableMap.hashCode() : 0);
        return result;
    }

    /**
     *
     * @param originalTensorGrad
     * @param graph
     * @return
     */
    public static TensorGrad create(TensorGrad originalTensorGrad, TensorGradGraph graph) {
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        TensorGradGraph clone = new TensorGradGraph(graph);
        TensorGrad ret = TensorGrad.builder()
                .variableMap(originalTensorGrad.getVariableMap())
                .arrayFactory(arrayFactory)
                .tensorGradVariables(originalTensorGrad.getTensorGradVariables())
                .arrayFieldDifferentialFunctionFactory(new DifferentialFunctionFactory<>(graph,arrayFactory))
                .graph(clone)
                .build();
        //ensuring proper tensorgrad reference
        clone.setTensorGrad(ret);

        return ret;
    }

    /**
     *
     * @return
     */
    public static TensorGrad create() {
        return new TensorGrad();
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

        TensorGrad execPipeline = dup();

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
    public TensorGrad dup() {
        Cloner cloner = new Cloner();
        return cloner.deepClone(this);
    }


    /**
     *
     * @return
     */
    public long numElements() {
        long ret = 0;
        for(TensorGradVariable variable : variables()) {
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
                TensorGradVariable variable = TensorGradVariable.builder()
                        .tensorGrad(this)
                        .varName(info.getId())
                        .build();
                variable.setShape(info.getShape());
                tensorGradVariables.add(variable);
                variableMap.put(info.getId(),variable);

            }
        }

        for(TensorGradVariable variable : variables()) {
            variable.allocate();
        }
    }

    public List<TensorGradVariable> variables() {
        return tensorGradVariables;
    }

    /**
     *
     *
     * @param name
     * @param arr
     * @return
     */
    public  TensorGradVariable var(String name, INDArray arr) {
        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name).build();
        NDArrayVertex ndArrayVertex = new NDArrayVertex(graph.nextVertexId(), ndArrayInformation);
        ArrayField arrayField = new ArrayField(ndArrayVertex,graph);
        TensorGradVariable ret = TensorGradVariable.builder()
                .tensorGrad(this).
                        arrayField(arrayFieldDifferentialFunctionFactory.var(name,arrayField))
                .varName(name)
                .arr(arr).build();
        addVariable(ret);
        return ret;

    }

    /**
     *
     * @param name
     * @param value
     * @return
     */
    public TensorGradVariable scalar(String name,double value) {
        return var(name,Nd4j.scalar(value));
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable grad(TensorGradVariable iX,TensorGradVariable wrt) {
        DifferentialFunction<ArrayField> arrField = iX .getArrayField() != null ?
               getFunctionInput(iX).diff(wrt.getArrayField()) :
                iX.getDifferentialFunction().diff(wrt.getArrayField());
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrField)
                .varName("grad(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable neq(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.neq(getFunctionInput(iX),iy.getArrayField()))
                .varName("neq(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable eq(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eq(getFunctionInput(iX),iy.getArrayField()))
                .varName("eq(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable or(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.or(getFunctionInput(iX),iy.getArrayField()))
                .varName("or(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable cos(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cos(getFunctionInput(iX)))
                .varName("cos(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sin(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sin(getFunctionInput(iX)))
                .varName("sin(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable tan(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tan(getFunctionInput(iX)))
                .varName("tan(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable acos(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acos(getFunctionInput(iX)))
                .varName("acos(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */

    public TensorGradVariable asin(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asin(getFunctionInput(iX)))
                .varName("asin(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable atan(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atan(getFunctionInput(iX)))
                .varName("atan(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable cosh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cosh(getFunctionInput(iX)))
                .varName("cosh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sinh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sinh(getFunctionInput(iX)))
                .varName("sinh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable tanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tanh(getFunctionInput(iX)))
                .varName("tanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable acosh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acosh(getFunctionInput(iX)))
                .varName("acosh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable asinh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asinh(getFunctionInput(iX)))
                .varName("asinh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable atanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atanh(getFunctionInput(iX)))
                .varName("atanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable exp(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.exp(getFunctionInput(iX)))
                .varName("exp(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable log(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.log(getFunctionInput(iX)))
                .varName("log(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param i_y
     * @return
     */
    public TensorGradVariable pow(TensorGradVariable iX,  TensorGradVariable i_y) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.pow(getFunctionInput(iX),null))
                .varName("pow(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sqrt(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sqrt(getFunctionInput(iX)))
                .varName("sqrt(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable square(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.square(getFunctionInput(iX)))
                .varName("square(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable floor(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.floor(getFunctionInput(iX)))
                .varName("floor(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable relu(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.relu(getFunctionInput(iX)))
                .varName("relu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable softmax(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softmax(getFunctionInput(iX)))
                .varName("softmax(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable hardTanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanh(getFunctionInput(iX)))
                .varName("hardTanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable hardTanhDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanhDerivative(getFunctionInput(iX)))
                .varName("hardTanhDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sigmoid(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoid(getFunctionInput(iX)))
                .varName("sigmoid(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    private DifferentialFunction<ArrayField> getFunctionInput(TensorGradVariable iX) {
        return iX.getArrayField() == null ?
                iX.getDifferentialFunction() : iX.getDifferentialFunction();
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sigmoidDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory
                        .sigmoidDerivative(getFunctionInput(iX)))
                .varName("sigmoidDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable sign(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory
                        .sign(getFunctionInput(iX))).differentialFunction(arrayFieldDifferentialFunctionFactory
                        .sign(iX.getDifferentialFunction()))
                .varName("sign(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable softsign(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsign(getFunctionInput(iX)))
                .varName("softsign(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable softsignDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsignDerivative(getFunctionInput(iX)))
                .varName("softsignDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable softplus(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softplus(getFunctionInput(iX)))
                .varName("softplus(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable elu(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.elu(getFunctionInput(iX)))
                .varName("elu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable eluDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eluDerivative(getFunctionInput(iX)))
                .varName("eluDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public TensorGradVariable leakyRelu(TensorGradVariable iX, double cutoff) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyRelu(getFunctionInput(iX),cutoff))
                .varName("leakyRelu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param cutoff
     * @return
     */
    public TensorGradVariable leakyReluDerivative(TensorGradVariable iX, double cutoff) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyReluDerivative(getFunctionInput(iX),cutoff))
                .varName("leakyReluDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable mean(TensorGradVariable iX) {

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mean(getFunctionInput(iX)))
                .varName("mean(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
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
    public TensorGradVariable standardDeviation(TensorGradVariable iX,
                                                boolean biasCorrected,
                                                int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.std(
                       getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
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
    public TensorGradVariable variance(TensorGradVariable iX,
                                       boolean biasCorrected,
                                       int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.variance(getFunctionInput(iX),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public TensorGradVariable sum(TensorGradVariable iX,
                                  int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sum(getFunctionInput(iX),dimensions))
                .varName("sum(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public TensorGradVariable prod(TensorGradVariable iX,
                                   int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.prod(getFunctionInput(iX),dimensions))
                .varName("prod(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public TensorGradVariable max(TensorGradVariable iX,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.max(getFunctionInput(iX),dimensions))
                .varName("max(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param dimensions
     * @return
     */
    public TensorGradVariable min(TensorGradVariable iX,
                                  int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.min(getFunctionInput(iX),dimensions))
                .varName("min(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @param shape
     * @return
     */
    public TensorGradVariable reshape(TensorGradVariable iX,
                                      int...shape) {
        shape = Shape.resolveNegativeShapeIfNeccessary(shape);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory
                        .reshape(getFunctionInput(iX),shape))
                .varName("reshape(" + iX.getVarName() + ")").tensorGrad(this)
                .shape(shape)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
     *
     * @param iX
     * @return
     */
    public TensorGradVariable transpose(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(getFunctionInput(iX)))
                .varName("transpose(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param x
     * @param axis
     * @return
     */
    public TensorGradVariable rollAxis(TensorGradVariable x, int axis) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.rollAxis(x.getArrayField(),axis))
                .varName("rollAxis(" + x.getVarName() + ")").tensorGrad(this)
                .build();
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
    public TensorGradVariable mmul(int argNum,TensorGradVariable x,TensorGradVariable y) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mmul(argNum ,x.getArrayField(), y.getArrayField()))
                .varName("mmul(" + x.getVarName() + "," + y.getVarName()  + ")").tensorGrad(this)
                .build();
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
    public TensorGradVariable tensorMmul(TensorGradVariable x,
                                         TensorGradVariable y,
                                         int[][] dimensions,
                                         int argNum) {

        int[] shape = ArrayUtil.getTensorMmulShape(x.getShape(), y.getShape(), dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tensorMmul(x.getArrayField(), y.getArrayField(), dimensions, argNum))
                .varName("tensorMmul(" + x.getVarName() + "," + y.getVarName() +  ")")
                .tensorGrad(this)
                .shape(shape)
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
    public TensorGradVariable cosineSimilarity(TensorGradVariable iX,TensorGradVariable i_y, int...dimensions) {
        DifferentialFunction<ArrayField> cosim = arrayFieldDifferentialFunctionFactory.cosineSimilarity(
               getFunctionInput(iX),
                i_y.getArrayField(),
                dimensions);

        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(cosim)
                .varName("cosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")")
                .tensorGrad(this).shape(arrayReduceShape)
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
    public TensorGradVariable euclideanDistance(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.euclideanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("euclideanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable manhattanDistance(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.manhattanDistance(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("manhattanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossBinaryXENT(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossBinaryXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossBinaryXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossCosineSimilarity(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossCosineSimilarity(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossCosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossHinge(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossKLD(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossKLD(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossKLD(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossL1(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL1(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossL1(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossL2(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL2(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossL2(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossMAE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMAE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
    public TensorGradVariable lossMAPE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAPE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMAPE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossMSE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMSE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossMCXENT(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMCXENT(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMCXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossMSLE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSLE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMSLE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossNegativeLogLikelihood(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(getFunctionInput(iX)))
                .varName("lossNegativeLogLikelihood(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public  TensorGradVariable lossPoisson(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossPoisson(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossPoisson(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
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
    public TensorGradVariable lossSquaredHinge(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossSquaredHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossSquaredHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    private void addVariable(TensorGradVariable variable) {
        tensorGradVariables.add(variable);
        variableMap.put(variable.getVarName(),variable);
    }


    private INDArray getX(OpExecAction opExecAction) {
        //   return variables().get(opExecAction.getInputsIds()[0]).getArr();
        return variableMap.get(opExecAction.getInputs()[0].getId()).getArr();
    }

    private INDArray getY(OpExecAction opExecAction) {
        if(opExecAction.getInputsIds().length > 1)
            return variableMap.get(opExecAction.getInputs()[1].getId()).getArr();
        return null;
    }

    private INDArray getZ(OpExecAction opExecAction) {
        return variableMap.get(opExecAction.getOutput().getId()).getArr();
    }


    /**
     *
     * @param opType
     * @param opExecAction
     * @return
     */
    public Op createOp(OpState.OpType opType,OpExecAction opExecAction) {
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
     *
     * @return
     */
    public INDArray execAndEndResult() {
        List<Op> exec = exec();
        return exec.get(exec.size() - 1).z();
    }

    /**
     *
     * @return
     */
    public List<Op> exec() {
        allocate();
        List<Op> ops = new ArrayList<>();
        if(graph().numVertices() == 0)
            throw new ND4JIllegalStateException("Unable to run exec pipeline. No vertices in graph");

        for(OpExecAction opExecAction : graph().getOpOrder().getActions()) {
            Op op = createOp(
                    opExecAction.getOpState().getOpType(),
                    opExecAction);
            Nd4j.getExecutioner().exec(op);
            ops.add(op);
        }

        return ops;
    }

}
