package org.nd4j.autodiff.tensorgrad;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
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
        arrayFactory = new ArrayFactory(graph);
        arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
        tensorGradVariables = new ArrayList<>();
        variableMap = new HashMap<>();
    }

    public TensorGradGraph graph() {
        return graph;
    }


    public static TensorGrad create(TensorGradGraph graph) {
        ArrayFactory arrayFactory = new ArrayFactory(graph);
        TensorGrad ret = TensorGrad.builder()
                .variableMap(new HashMap<>())
                .arrayFactory(arrayFactory)
                .tensorGradVariables(new ArrayList<>())
                .arrayFieldDifferentialFunctionFactory(new DifferentialFunctionFactory<>(graph,arrayFactory))
                .graph(graph)
                .build();

        return ret;
    }

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
        for(Map.Entry<String,INDArray> variableEntry : inputs.entrySet()) {
            execPipeline.updateNDArray(variableEntry.getKey(),variableEntry.getValue());
        }

        List<Op> opExecAction = execPipeline.exec();
        if(opExecAction.isEmpty())
            throw new IllegalStateException("No ops found to execute.");
        INDArray[] ret = new INDArray[opExecAction.size()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = opExecAction.get(i).z();
        }
        return ret;
    }

    public TensorGrad dup() {
        TensorGradGraph tensorGradGraph = TensorGradGraph.builder()
                .allowMultipleEdges(graph.isAllowMultipleEdges())
                .frozen(graph.isFrozen())
                .edges(new HashMap<>(graph.getEdges()))
                .incomingEdges(new HashMap<>(graph.getIncomingEdges()))
                .vertices(new HashMap<>(graph.getVertices()))
                .build();
        TensorGrad ret = TensorGrad.builder()
                .arrayFactory(new ArrayFactory(tensorGradGraph))
                .graph(tensorGradGraph)
                .tensorGradVariables(new ArrayList<>(tensorGradVariables))
                .variableMap(new HashMap<>(variableMap))
                .arrayFieldDifferentialFunctionFactory(arrayFieldDifferentialFunctionFactory)
                .build();
        return ret;
    }


    public long numElements() {
        long ret = 0;
        for(TensorGradVariable variable : variables()) {
            ret += ArrayUtil.prod(variable.getShape());
        }
        return ret;
    }

    public void allocate() {
        for(int i = 0; i < graph().numVertices(); i++) {
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


    public  TensorGradVariable var(String name, INDArray arr) {
        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name).build();
        NDArrayVertex ndArrayVertex = new NDArrayVertex(graph.numVertices(),ndArrayInformation);
        ArrayField arrayField = new ArrayField(ndArrayVertex,graph);
        TensorGradVariable ret = TensorGradVariable.builder()
                .tensorGrad(this).
                        arrayField(arrayFieldDifferentialFunctionFactory.var(name,arrayField))
                .varName(name)
                .arr(arr).build();
        addVariable(ret);
        return ret;

    }


    public TensorGradVariable grad(TensorGradVariable iX,TensorGradVariable wrt) {
        DifferentialFunction<ArrayField> arrField = iX .getArrayField() != null ?
                iX.getArrayField().diff(wrt.getArrayField()) :
                iX.getDifferentialFunction().diff(wrt.getArrayField());
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrField)
                .varName("grad(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable neq(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.neq(iX.getArrayField(),iy.getArrayField()))
                .varName("neq(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable eq(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eq(iX.getArrayField(),iy.getArrayField()))
                .varName("eq(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable or(TensorGradVariable iX,TensorGradVariable iy) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.or(iX.getArrayField(),iy.getArrayField()))
                .varName("or(" + iX.getVarName() + "," + iy.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable cos(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cos(iX.getArrayField()))
                .varName("cos(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sin(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sin(iX.getArrayField()))
                .varName("sin(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable tan(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tan(iX.getArrayField()))
                .varName("tan(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable acos(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acos(iX.getArrayField()))
                .varName("acos(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable asin(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asin(iX.getArrayField()))
                .varName("asin(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable atan(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atan(iX.getArrayField()))
                .varName("atan(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable cosh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cosh(iX.getArrayField()))
                .varName("cosh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sinh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sinh(iX.getArrayField()))
                .varName("sinh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable tanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tanh(iX.getArrayField()))
                .varName("tanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable acosh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acosh(iX.getArrayField()))
                .varName("acosh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable asinh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asinh(iX.getArrayField()))
                .varName("asinh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable atanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atanh(iX.getArrayField()))
                .varName("atanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable exp(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.exp(iX.getArrayField()))
                .varName("exp(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable log(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.log(iX.getArrayField()))
                .varName("log(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable pow(TensorGradVariable iX,  TensorGradVariable i_y) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.pow(iX.getArrayField(),null))
                .varName("pow(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sqrt(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sqrt(iX.getArrayField()))
                .varName("sqrt(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable square(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.square(iX.getArrayField()))
                .varName("square(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable floor(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.floor(iX.getArrayField()))
                .varName("floor(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable relu(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.relu(iX.getArrayField()))
                .varName("relu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable softmax(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softmax(iX.getArrayField()))
                .varName("softmax(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable hardTanh(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanh(iX.getArrayField()))
                .varName("hardTanh(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable hardTanhDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanhDerivative(iX.getArrayField()))
                .varName("hardTanhDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sigmoid(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoid(iX.getArrayField()))
                .varName("sigmoid(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sigmoidDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoidDerivative(iX.getArrayField()))
                .varName("sigmoidDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sign(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sign(iX.getArrayField()))
                .varName("sign(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable softsign(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsign(iX.getArrayField()))
                .varName("softsign(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable softsignDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsignDerivative(iX.getArrayField()))
                .varName("softsignDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable softplus(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softplus(iX.getArrayField()))
                .varName("softplus(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable elu(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.elu(iX.getArrayField()))
                .varName("elu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable eluDerivative(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eluDerivative(iX.getArrayField()))
                .varName("eluDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable leakyRelu(TensorGradVariable iX, double cutoff) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyRelu(iX.getArrayField(),cutoff))
                .varName("leakyRelu(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable leakyReluDerivative(TensorGradVariable iX, double cutoff) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyReluDerivative(iX.getArrayField(),cutoff))
                .varName("leakyReluDerivative(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }



    public TensorGradVariable mean(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mean(iX.getArrayField()))
                .varName("mean(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable standardDeviation(TensorGradVariable iX,
                                                boolean biasCorrected,
                                                int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.std(
                        iX.getArrayField(),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable variance(TensorGradVariable iX,
                                       boolean biasCorrected,
                                       int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.variance(iX.getArrayField(),
                        biasCorrected ,
                        dimensions))
                .varName("variance(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable sum(TensorGradVariable iX,
                                  int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sum(iX.getArrayField(),dimensions))
                .varName("sum(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable prod(TensorGradVariable iX,
                                   int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.prod(iX.getArrayField(),dimensions))
                .varName("prod(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable max(TensorGradVariable iX,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.max(iX.getArrayField(),dimensions))
                .varName("max(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable min(TensorGradVariable iX,
                                  int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.min(iX.getArrayField(),dimensions))
                .varName("min(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable reshape(TensorGradVariable iX,
                                      int...shape) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.reshape(iX.getArrayField(),shape))
                .varName("reshape(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable transpose(TensorGradVariable iX) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(iX.getArrayField()))
                .varName("transpose(" + iX.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }



    public TensorGradVariable rollAxis(TensorGradVariable x, int axis) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.rollAxis(x.getArrayField(),axis))
                .varName("rollAxis(" + x.getVarName() + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable mmul(int argNum,TensorGradVariable x,TensorGradVariable y) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.mmul(argNum ,x.getArrayField(), y.getArrayField()))
                .varName("mmul(" + x.getVarName() + "," + y.getVarName()  + ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable tensorMmul(TensorGradVariable x,
                                         TensorGradVariable y,
                                         int[][]dimensions,
                                         int argNum) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tensorMmul(x.getArrayField(),y.getArrayField(),dimensions,argNum))
                .varName("tensorMmul(" + x.getVarName() + "," + y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable cosineSimilarity(TensorGradVariable iX,TensorGradVariable i_y, int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cosineSimilarity(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("cosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable euclideanDistance(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.euclideanDistance(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("euclideanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable manhattanDistance(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.manhattanDistance(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("manhattanDistance(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossBinaryXENT(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossBinaryXENT(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossBinaryXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossCosineSimilarity(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossCosineSimilarity(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossCosineSimilarity(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossHinge(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossHinge(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossHinge(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossKLD(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossKLD(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossKLD(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossL1(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL1(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossL1(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossL2(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossL2(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossL2(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossMAE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAE(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossMAE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossMAPE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMAPE(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossMAPE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossMSE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSE(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossMSE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossMCXENT(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMCXENT(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossMCXENT(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossMSLE(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossMSLE(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossMSLE(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public TensorGradVariable lossNegativeLogLikelihood(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.transpose(iX.getArrayField()))
                .varName("lossNegativeLogLikelihood(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }

    public  TensorGradVariable lossPoisson(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossPoisson(iX.getArrayField(),i_y.getArrayField(),dimensions))
                .varName("lossPoisson(" + iX.getVarName() + "," + i_y.getVarName() +  ")").tensorGrad(this)
                .build();
        addVariable(ret);
        return ret;
    }


    public TensorGradVariable lossSquaredHinge(TensorGradVariable iX,TensorGradVariable i_y,int...dimensions) {
        TensorGradVariable ret = TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.lossSquaredHinge(iX.getArrayField(),i_y.getArrayField(),dimensions))
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


    public Op createOp(OpState.OpType opType,OpExecAction opExecAction) {
        OpState opState = opExecAction.getOpState();
        switch (opType) {
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
                return Nd4j.getOpFactory().createTransform(opState.getOpName(),
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

        throw new IllegalStateException("");
    }



    public INDArray execAndEndResult() {
        List<Op> exec = exec();
        return exec.get(exec.size() - 1).z();
    }

    public List<Op> exec() {
        allocate();
        List<Op> ops = new ArrayList<>();
        for(OpExecAction opExecAction : graph().getOpOrder().getActions()) {
            Op op = createOp(opExecAction.getOpState().getOpType(),
                    opExecAction);
            Nd4j.getExecutioner().exec(op);
            ops.add(op);
        }

        return ops;
    }

}
