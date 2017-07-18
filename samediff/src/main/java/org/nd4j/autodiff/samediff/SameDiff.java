package org.nd4j.autodiff.samediff;

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
import org.nd4j.autodiff.samediff.impl.SameDiffVariable;
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
public class SameDiff {
    private SameDiffGraph graph = new SameDiffGraph();
    private ArrayFactory arrayFactory = new ArrayFactory(graph);
    private DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory;
    private List<SameDiffVariable> tensorGradVariables = new ArrayList<>();
    private Map<String,SameDiffVariable> variableMap;

    private SameDiff() {
        graph = new SameDiffGraph();
        graph.setTensorGrad(this);
        arrayFactory = new ArrayFactory(graph);
        arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        tensorGradVariables = new ArrayList<>();
        variableMap = new HashMap<>();
    }

    public SameDiffGraph graph() {
        return graph;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SameDiff that = (SameDiff) o;

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
     * @param originalSameDiff
     * @param graph
     * @return
     */
    public static SameDiff create(SameDiff originalSameDiff, SameDiffGraph graph) {
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        SameDiffGraph clone = new SameDiffGraph(graph);
        SameDiff ret = SameDiff.builder()
                .variableMap(originalSameDiff.getVariableMap())
                .arrayFactory(arrayFactory)
                .tensorGradVariables(originalSameDiff.getTensorGradVariables())
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
        for(SameDiffVariable variable : variables()) {
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
                SameDiffVariable variable = SameDiffVariable.builder()
                        .tensorGrad(this)
                        .varName(info.getId())
                        .build();
                variable.setShape(info.getShape());
                tensorGradVariables.add(variable);
                variableMap.put(info.getId(),variable);

            }
        }

        for(SameDiffVariable variable : variables()) {
            variable.allocate();
        }
    }

    public List<SameDiffVariable> variables() {
        return tensorGradVariables;
    }

    /**
     *
     *
     * @param name
     * @param arr
     * @return
     */
    public SameDiffVariable var(String name, INDArray arr) {
        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name).build();
        if(ArrayUtil.prod(arr.shape()) == 1)
            ndArrayInformation.setScalarValue(arr.getDouble(0));
        NDArrayVertex ndArrayVertex = new NDArrayVertex(graph.nextVertexId(), ndArrayInformation);
        ArrayField arrayField = new ArrayField(ndArrayVertex,graph);
        SameDiffVariable ret = SameDiffVariable.builder()
                .tensorGrad(this).
                        arrayField(arrayFieldDifferentialFunctionFactory.var(name,arrayField))
                .shape(arr.shape())
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
    public SameDiffVariable scalar(String name, double value) {
        return var(name,Nd4j.scalar(value));
    }

    /**
     *
     * @param iX
     * @return
     */
    public SameDiffVariable grad(SameDiffVariable iX, SameDiffVariable wrt) {
        DifferentialFunction<ArrayField> arrField = getFunctionInput(iX).diff(wrt.getArrayField());
        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null).shape(wrt.getShape())
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
    public SameDiffVariable neq(SameDiffVariable iX, SameDiffVariable iy) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable eq(SameDiffVariable iX, SameDiffVariable iy) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable or(SameDiffVariable iX, SameDiffVariable iy) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable neg(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.neg(getFunctionInput(iX)))
                .varName("neg(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    /**
     *
     * @param iX
     * @return
     */
    public SameDiffVariable cos(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sin(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable tan(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable acos(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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

    public SameDiffVariable asin(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable atan(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable cosh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sinh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable tanh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable acosh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable asinh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable atanh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable exp(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable log(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable pow(SameDiffVariable iX, SameDiffVariable i_y) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sqrt(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable square(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable floor(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable relu(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable softmax(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable hardTanh(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable hardTanhDerivative(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sigmoid(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null).shape(iX.getShape())
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoid(getFunctionInput(iX)))
                .varName("sigmoid(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    private DifferentialFunction<ArrayField> getFunctionInput(SameDiffVariable iX) {
        return iX.getDifferentialFunction() != null ?
                iX.getDifferentialFunction() : iX.getArrayField();
    }

    /**
     *
     * @param iX
     * @return
     */
    public SameDiffVariable sigmoidDerivative(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sign(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable softsign(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable softsignDerivative(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable softplus(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable elu(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable eluDerivative(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable leakyRelu(SameDiffVariable iX, double cutoff) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable leakyReluDerivative(SameDiffVariable iX, double cutoff) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable mean(SameDiffVariable iX) {

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable standardDeviation(SameDiffVariable iX,
                                              boolean biasCorrected,
                                              int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable variance(SameDiffVariable iX,
                                     boolean biasCorrected,
                                     int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable sum(SameDiffVariable iX,
                                int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable prod(SameDiffVariable iX,
                                 int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable max(SameDiffVariable iX, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable min(SameDiffVariable iX,
                                int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable reshape(SameDiffVariable iX,
                                    int...shape) {
        shape = Shape.resolveNegativeShapeIfNeccessary(shape);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable transpose(SameDiffVariable iX) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable rollAxis(SameDiffVariable x, int axis) {
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable mmul(int argNum, SameDiffVariable x, SameDiffVariable y) {
        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mmul(argNum ,x.getArrayField(), y.getArrayField()))
                .varName("mmul(" + x.getVarName() + "," + y.getVarName()  + ")").tensorGrad(this)
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
    public SameDiffVariable tensorMmul(SameDiffVariable x,
                                       SameDiffVariable y,
                                       int[][] dimensions,
                                       int argNum) {

        int[] shape = ArrayUtil.getTensorMmulShape(x.getShape(), y.getShape(), dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable cosineSimilarity(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        DifferentialFunction<ArrayField> cosim = arrayFieldDifferentialFunctionFactory.cosineSimilarity(
               getFunctionInput(iX),
                i_y.getArrayField(),
                dimensions);

        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);
        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable euclideanDistance(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable manhattanDistance(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossBinaryXENT(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossCosineSimilarity(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossHinge(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossKLD(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossL1(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossL2(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossMAE(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAE(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossMAE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    /**
    public SameDiffVariable lossMAPE(SameDiffVariable iX,SameDiffVariable i_y,int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossMSE(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossMCXENT(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossMSLE(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossNegativeLogLikelihood(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossPoisson(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
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
    public SameDiffVariable lossSquaredHinge(SameDiffVariable iX, SameDiffVariable i_y, int...dimensions) {
        int[] arrayReduceShape = Shape.getReducedShape(iX.getShape(),dimensions);

        SameDiffVariable ret = SameDiffVariable.builder()
                .arr(null).shape(arrayReduceShape)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossSquaredHinge(getFunctionInput(iX),i_y.getArrayField(),dimensions))
                .varName("lossSquaredHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    private void addVariable(SameDiffVariable variable) {
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
        List<Op> ops = new ArrayList<>();
        if(graph().numVertices() == 0)
            throw new ND4JIllegalStateException("Unable to run exec pipeline. No vertices in graph");


        List<OpExecAction> opExecActions = graph().getOpOrder().getActions();
        for(int i = 0; i < opExecActions.size(); i++) {
            OpExecAction opExecAction = opExecActions.get(i);
            Op op = createOp(
                    opExecAction.getOpState().getOpType(),
                    opExecAction);
            Nd4j.getExecutioner().exec(op);
            ops.add(op);
        }

        return ops;
    }

}
