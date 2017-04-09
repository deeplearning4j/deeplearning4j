package org.nd4j.autodiff.tensorgrad;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.autodiff.Constant;
import org.nd4j.autodiff.autodiff.DifferentialFunction;
import org.nd4j.autodiff.autodiff.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 4/9/17.
 */
@AllArgsConstructor
public class TensorGrad {
    private Graph<NDArrayInformation,OpState> graph = new Graph<>();
    private ArrayFactory arrayFactory = new ArrayFactory(graph);
    private DifferentialFunctionFactory<ArrayField> arrayFieldDifferentialFunctionFactory;

    private TensorGrad() {
        graph = new Graph<>();
        arrayFactory = new ArrayFactory(graph);
        arrayFieldDifferentialFunctionFactory = new DifferentialFunctionFactory<>(graph,arrayFactory);
    }


    public static TensorGrad create() {
        return new TensorGrad();
    }



    public  TensorGradVariable var(String name, INDArray arr) {
        NDArrayInformation ndArrayInformation = NDArrayInformation.builder()
                .shape(arr.shape()).id(name).build();
        NDArrayVertex ndArrayVertex = new NDArrayVertex(graph.numVertices(),ndArrayInformation);
        ArrayField arrayField = new ArrayField(ndArrayVertex,graph);
        return  TensorGradVariable.builder()
                .tensorGrad(this).
                        arrayField(arrayFieldDifferentialFunctionFactory.var(name,arrayField))
                .varName(name)
                .arr(arr).build();

    }


    public TensorGradVariable grad(TensorGradVariable iX,TensorGradVariable wrt) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(iX.getArrayField().diff(wrt.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable cos(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cos(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sin(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sin(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable tan(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tan(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable acos(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acos(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable asin(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asin(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable atan(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atan(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable cosh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.cosh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sinh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sinh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable tanh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.tanh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable acosh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.acosh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable asinh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.asinh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable atanh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.atanh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable exp(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.exp(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable log(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.log(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable pow(TensorGradVariable iX,  TensorGradVariable i_y) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.pow(iX.getArrayField(),null))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sqrt(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sqrt(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable square(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.square(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable floor(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.floor(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable relu(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.relu(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable softmax(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softmax(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable hardTanh(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanh(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable hardTanhDerivative(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.hardTanhDerivative(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sigmoid(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoid(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sigmoidDerivative(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sigmoidDerivative(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable sign(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.sign(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable softsign(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsign(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable softsignDerivative(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softsignDerivative(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable softplus(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.softplus(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable elu(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.elu(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable eluDerivative(TensorGradVariable iX) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.eluDerivative(iX.getArrayField()))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable leakyRelu(TensorGradVariable iX, double cutoff) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyRelu(iX.getArrayField(),cutoff))
                .varName("").tensorGrad(this)
                .build();
    }

    public TensorGradVariable leakyReluDerivative(TensorGradVariable iX, double cutoff) {
        return TensorGradVariable.builder()
                .arr(null)
                .differentialFunction(arrayFieldDifferentialFunctionFactory.leakyReluDerivative(iX.getArrayField(),cutoff))
                .varName("").tensorGrad(this)
                .build();
    }




}
