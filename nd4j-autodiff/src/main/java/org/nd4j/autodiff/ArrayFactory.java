package org.nd4j.autodiff;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;

@AllArgsConstructor
@Data
public class ArrayFactory implements AbstractFactory<ArrayField> {

    private Graph<NDArrayInformation,OpState> graph;

    public ArrayFactory() {
        this(new Graph<>());
    }


    @Override
    public Graph<NDArrayInformation, OpState> graph() {
        return graph;
    }

    @Override
    public ArrayField eq(ArrayField i_x, ArrayField i_y) {
        return i_x.eq(i_y);
    }

    @Override
    public ArrayField neq(ArrayField i_x, ArrayField i_y) {
        return i_x.neq(i_y);
    }

    @Override
    public ArrayField or(ArrayField i_x, ArrayField i_y) {
        return i_x.or(i_y);
    }

    @Override
    public ArrayField add(ArrayField i_x, Number value) {
        return i_x.plus(value.doubleValue());
    }

    @Override
    public ArrayField sub(ArrayField i_x, Number value) {
        return i_x.minus(value.doubleValue());
    }

    @Override
    public ArrayField mul(ArrayField i_x, Number value) {
        return i_x.mul((long) value.doubleValue());
    }

    @Override
    public ArrayField div(ArrayField i_x, Number value) {
        return i_x.div(value.doubleValue());
    }

    @Override
    public ArrayField broadcast(ArrayField i_x, int... shape) {
        return i_x.broadcast(shape);
    }

    @Override
    public ArrayField repeat(ArrayField i_x, int axis) {
        return i_x.repeat(axis);
    }

    @Override
    public ArrayField tile(ArrayField i_x, int... repeat) {
        return i_x.tile(repeat);
    }

    @Override
    public ArrayField sum(ArrayField i_x, int... dimensions) {
        return i_x.sum(dimensions);
    }

    @Override
    public ArrayField prod(ArrayField i_x, int... dimensions) {
        return i_x.prod(dimensions);
    }

    @Override
    public ArrayField mean(ArrayField i_x, int... dimensions) {
        return i_x.mean(dimensions);
    }

    @Override
    public ArrayField std(ArrayField i_x, boolean biasCorrected, int... dimensions) {
        return i_x.std(dimensions);
    }

    @Override
    public ArrayField variance(ArrayField i_x, boolean biasCorrected, int... dimensions) {
        return i_x.variance(dimensions);
    }

    @Override
    public ArrayField max(ArrayField i_x, int... dimensions) {
        return i_x.max(dimensions);
    }

    @Override
    public ArrayField min(ArrayField i_x, int... dimensions) {
        return i_x.min(dimensions);
    }

    @Override
    public ArrayField norm1(ArrayField i_x, int... dimensions) {
        return i_x.norm1(dimensions);
    }

    @Override
    public ArrayField norm2(ArrayField i_x, int... dimensions) {
        return i_x.norm2(dimensions);
    }

    @Override
    public ArrayField normmax(ArrayField i_x, int... dimensions) {
        return i_x.normmax(dimensions);
    }

    @Override
    public ArrayField transpose(ArrayField i_x) {
        return i_x.transpose();
    }

    @Override
    public ArrayField reshape(ArrayField i_x, int[] shape) {
        return i_x.reshape(shape);
    }

    @Override
    public ArrayField valueArrayOf(ArrayField i_x, int[] shape) {
        return i_x.valueArrayOf(shape);
    }

    @Override
    public ArrayField val(double v) {
        // return Nd4j.valueArrayOf(v,i);
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField abs(ArrayField x) {
        return x.abs();
    }

    @Override
    public ArrayField min(ArrayField x, ArrayField y) {
       /* return x.doubleValue() < y.doubleValue() ? new ArrayField(
                x.doubleValue()) : new ArrayField(y.doubleValue());*/
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField max(ArrayField x, ArrayField y) {
     /*   return x.doubleValue() > y.doubleValue() ? new ArrayField(
                x.doubleValue()) : new ArrayField(y.doubleValue());*/
        throw new UnsupportedOperationException();
    }



    @Override
    public ArrayField zero() {
        return new ArrayField(new NDArrayVertex(graph.numVertices(),new int[]{1,1}),graph);
    }

    @Override
    public ArrayField one() {
        return new ArrayField(new NDArrayVertex(graph.numVertices(),new int[]{1,1}),graph);
    }

    @Override
    public ArrayField cos(ArrayField x) {
        return x.cos();
    }

    @Override
    public ArrayField acos(ArrayField x) {
        return x.acos();
    }

    @Override
    public ArrayField cosh(ArrayField x) {
        return x.cosh();
    }

    @Override
    public ArrayField acosh(ArrayField x) {
        return x.acosh();
    }

    @Override
    public ArrayField sin(ArrayField x) {
        return x.sin();
    }

    @Override
    public ArrayField asin(ArrayField x) {
        return x.asin();
    }

    @Override
    public ArrayField sinh(ArrayField x) {
        return x.sinh();
    }

    @Override
    public ArrayField asinh(ArrayField x) {
        return x.asinh();
    }

    @Override
    public ArrayField tan(ArrayField x) {
        return x.tan();
    }

    @Override
    public ArrayField atan(ArrayField x) {
        return x.atan();
    }

    @Override
    public ArrayField atan2(ArrayField x, ArrayField y) {
        //   return new ArrayField(Math.atan2(x.doubleValue(), y.doubleValue()));
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField tanh(ArrayField x) {
        return x.tanh();
    }

    @Override
    public ArrayField atanh(ArrayField x) {
        return x.atanh();
    }

    @Override
    public ArrayField exp(ArrayField x) {
        return x.exp();
    }

    @Override
    public ArrayField log(ArrayField x) {
        return x.log();
    }

    @Override
    public ArrayField log10(ArrayField x) {
        return x.log10();
    }

    @Override
    public ArrayField flat(ArrayField x) {
      /*  double xValue = x.doubleValue();
        return new ArrayField(-xValue + (xValue + xValue) * randomGenerator.nextDouble());*/
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField mc(ArrayField x, ArrayField y) {
      /*  double max = Math.max(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue()));
        double min = Math.min(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue()));
        return new ArrayField(min + (max - min) * randomGenerator.nextDouble());*/
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField rand(ArrayField x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField random(ArrayField x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField gauss(ArrayField x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField sgn(ArrayField x) {
        return x.sgn();
    }

    @Override
    public ArrayField ifx(ArrayField x, ArrayField y, ArrayField z) {
        //return x.doubleValue() > .5 ? y : z;
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField buf(ArrayField x) {
        //return x.doubleValue() > .5 ? new ArrayField(1) : new ArrayField(0);
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField inv(ArrayField x) {
        //  return x.doubleValue() > .5 ? new ArrayField(0) : new ArrayField(1);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField u(ArrayField x) {
        //return x.doubleValue() > 0 ? new ArrayField(1) : new ArrayField(0);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField uramp(ArrayField x) {
        // return x.doubleValue() > 0 ? new ArrayField(x.doubleValue()) : new ArrayField(0);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField pow(ArrayField x, ArrayField y) {
        return x.pow(y);
    }

    @Override
    public ArrayField pwr(ArrayField x, ArrayField y) {
        return x.pwr(y);
    }

    @Override
    public ArrayField pwrs(ArrayField x, ArrayField y) {
        return x.pwrs(y);
    }

    @Override
    public ArrayField sqrt(ArrayField x) {
        return x.sqrt();
    }

    @Override
    public ArrayField square(ArrayField x) {
        return x.square();
    }

    @Override
    public ArrayField hypot(ArrayField x, ArrayField y) {
        return x.pow(2).plus(y.pow(2)).sqrt();
    }

    @Override
    public ArrayField floor(ArrayField value) {
        return value.floor();
    }

    @Override
    public ArrayField ceil(ArrayField value) {
        return value.ceil();
    }

    @Override
    public ArrayField round(ArrayField value) {
        return value.round();
    }

    @Override
    public ArrayField relu(ArrayField value) {
        return value.relu();
    }

    @Override
    public ArrayField leakyRelu(ArrayField value, double alpha) {
        return value.leakyRelu();
    }

    /**
     * Leaky relu with an alpha of
     * 0.01
     *
     * @param value the value to transform
     * @return
     */
    @Override
    public ArrayField leakyRelu(ArrayField value) {
        return value.leakyRelu();
    }

    @Override
    public ArrayField leakyReluDerivative(ArrayField value, double alpha) {
        return value.leakyReluDerivative(alpha);
    }

    /**
     * Leaky relu with an alpha of
     * 0.01
     *
     * @param value the value to transform
     * @return
     */
    @Override
    public ArrayField leakyReluDerivative(ArrayField value) {
        return value.leakyReluDerivative(0.001);
    }

    @Override
    public ArrayField hardTanh(ArrayField value) {
        return value.hardTanh();
    }

    @Override
    public ArrayField hardTanhDerivative(ArrayField value) {
        return value.hardTanh();
    }

    @Override
    public ArrayField sigmoid(ArrayField value) {
        return value.sigmoid();
    }

    @Override
    public ArrayField sigmoidDerivative(ArrayField value) {
        return value.sigmoidDerivative();
    }

    @Override
    public ArrayField softmax(ArrayField value) {
        return value.softmax();
    }

    @Override
    public ArrayField elu(ArrayField value) {
        return value.elu();
    }

    @Override
    public ArrayField eluDerivative(ArrayField value) {
        return value.eluDerivative();
    }

    @Override
    public ArrayField step(ArrayField value) {
        return value.step();
    }

    @Override
    public ArrayField sign(ArrayField value) {
        return value.sgn();
    }

    @Override
    public ArrayField softsign(ArrayField value) {
        return value.softsign();
    }

    @Override
    public ArrayField softsignDeriviative(ArrayField value) {
        return value.softsignDerivative();
    }

    @Override
    public ArrayField softplus(ArrayField value) {
        return value.softplus();
    }

    @Override
    public ArrayField rollAxis(ArrayField value, int axis) {
        return value.rollAxis(axis);
    }

    @Override
    public ArrayField lossSquaredHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossSquaredHinge(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossPoisson(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossPoisson(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossNegativeLogLikelihood(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossNegativeLogLikelihood(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossMSLE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossMSLE(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossMCXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossMCXENT(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossMSE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossMSE(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossMAPE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossMAPE(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossMAE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossMAE(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossL2(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossL2(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossL1(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossL1(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossKLD(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossKLD(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossHinge(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossCosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossCosineSimilarity(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField lossBinaryXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().lossBinaryXENT(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField manhattanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().manhattanDistance(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField euclideanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().euclideanDistance(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField cosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return iX.getValue().cosineSimilarity(i_y.getValue(),dimensions);
    }

    @Override
    public ArrayField expandDims(ArrayField input, int dim) {
        return input.expandDims(dim);
    }

}
