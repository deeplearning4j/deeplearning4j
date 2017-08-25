package org.nd4j.autodiff;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

/**
 *
 * @param <X>
 */
public interface AbstractFactory<X extends Field<X>>
        extends AbstractIdentityFactory<X> {


    /**
     *
     * @return
     */
    SameDiff sameDiff();

    List<String> methodNames();

    X invoke(String name,Object[] args);


    ArrayField selu(ArrayField value);

    X eq(X i_x, X i_y);

    X neq(X i_x, X i_y);

    X or(X i_x, X i_y);


    X add(X i_x,Number value);
    X sub(X i_x,Number value);
    X mul(X i_x,Number value);
    X div(X i_x,Number value);

    X broadcast(X i_x,int...shape);

    X repeat(X i_x,int axis);

    X tile(X i_x,int...repeat);

    X sum(X i_x,int...dimensions);
    X prod(X i_x,int...dimensions);
    X mean(X i_x,int...dimensions);
    X std(X i_x, boolean biasCorrected, int... dimensions);
    X variance(X i_x, boolean biasCorrected, int... dimensions);
    X max(X i_x,int...dimensions);
    X min(X i_x,int...dimensions);
    X norm1(X i_x,int...dimensions);
    X norm2(X i_x,int...dimensions);
    X normmax(X i_x,int...dimensions);


    X neg(X i_x);

    X transpose(X i_x);

    X reshape(X i_x, int[] shape);


    X valueArrayOf(X i_x,int[] shape);

    X val(double i_v);

    X abs(X i_x);

    X min(X i_x, X i_y);
    X max(X i_x, X i_y);

    X cos(X i_x);
    X acos(X i_x);
    X cosh(X i_x);
    X acosh(X i_x);

    X sin(X i_x);
    X asin(X i_x);
    X sinh(X i_x);
    X asinh(X i_x);

    X tan(X i_x);
    X atan(X i_x);
    X atan2(X i_x, X i_y);
    X tanh(X i_x);
    X atanh(X i_x);

    X exp(X i_x);
    X log(X i_x);

    X log10(X i_x);

    X flat(X i_x);
    X mc(X i_x, X i_y);
    X rand(X i_x);
    X random(X i_x);
    X gauss(X i_x);

    X sgn(X i_x);

    X ifx(X i_x, X i_y, X i_z);

    X buf(X i_x);
    X inv(X i_x);
    X u(X i_x);
    X uramp(X i_x);

    X pow(X i_x, X i_y);
    X pwr(X i_x, X i_y);
    X pwrs(X i_x, X i_y);
    X sqrt(X i_x);
    X square(X i_x);
    X hypot(X i_x, X i_y);

    X floor(X value);
    X ceil(X value);
    X round(X value);

    X relu(X value);

    X leakyRelu(X value,double alpha);

    /**
     * Leaky relu with an alpha of
     * 0.01
     * @param value the value to transform
     * @return
     */
    X leakyRelu(X value);

    X leakyReluDerivative(X value, X wrt, double alpha);

    /**
     * Leaky relu with an alpha of
     * 0.01
     * @param value the value to transform
     * @param wrt
     * @return
     */
    X leakyReluDerivative(X value, X wrt);


    X hardTanh(X value);

    X hardTanhDerivative(X value, X wrt);

    X sigmoid(X value);

    X sigmoidDerivative(X value, X wrt);


    X softmax(X value);

    X elu(X value);

    X eluDerivative(X value, X wrt);

    X step(X value);

    X sign(X value);

    X softsign(X value);

    X softsignDeriviative(X value, X wrt);

    X softplus(X value);

    X rollAxis(X value, int axis);

    X lossSquaredHinge(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossPoisson(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossNegativeLogLikelihood(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossMSLE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossMCXENT(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossMSE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossMAPE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossMAE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossL2(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossL1(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossKLD(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossHinge(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossCosineSimilarity(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X lossBinaryXENT(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X manhattanDistance(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X euclideanDistance(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X cosineSimilarity(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int[] dimensions);

    X expandDims(X input,int dim);

    X mmul(DifferentialFunction<X> arrayField, DifferentialFunction<X> y);

    X tensorMmul(DifferentialFunction<X> arrayField, DifferentialFunction<X> y, int[][] dimensions);

    /*

     */
    X permute(X value, int[] dimensions);

    X set(X value, X value1);

    X softmaxDerivative(X value, X wrt);

    X seluDerivative(X value, X wrt);

    X tanhDerivative(X value, X wrt);
}
