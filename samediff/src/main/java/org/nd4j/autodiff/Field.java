package org.nd4j.autodiff;


import org.nd4j.autodiff.functions.DifferentialFunction;

/**
 *
 * @param <X>
 */
public interface Field<X> extends CommutativeRing<X> {


    X inverse();


    X rsubi(X i_v);

    X rdivi(X i_v);

    X subi(X i_v);

    X divi(X i_v);

    X inversei();

    X subi(double i_v);

    X rsubi(double v);

    X rdivi(double v);

    X divi(double v);

    X rdiv(X i_v);

    X div(X i_v);

    double getReal();

    X[] args();


    X rsub(double v);

    X rdiv(double v);

    X pow(X a);

    X floor();

    X ceil();

    X round();

    X abs();

    X sqrt();

    X minus(double v);

    X prod(double v);

    X div(double v);

    X pow(double v);

    X cos();

    X acos();

    X cosh();

    X acosh();

    X sin();

    X asin();

    X sinh();

    X asinh();

    X tan();

    X atan();

    X tanh();

    X atanh();

    X exp();

    X log();

    X log10();

    X sgn();

    X pwr(X y);

    X pwrs(X y);

    X square();

    X relu();

    X hardTanh();

    X hardTanhDerivative(X wrt);

    X leakyRelu();

    X elu();

    X eluDerivative(X wrt);

    X leakyRelu(double cutoff);

    X leakyReluDerivative();

    X leakyReluDerivative(X wrt, double cutoff);

    X sigmoid();

    X sigmoidDerivative(X wrt);

    X step();

    X softsign();

    X softsignDerivative(X wrt);

    X softmax();

    X softmaxDerivative(ArrayField wrt);

    X softplus();

    X reshape(int[] shape);

    X transpose();

    X permute(int[] dimensions);

    X expandDims(int dim);

    X sum(int[] dimensions);

    X prod(int[] dimensions);

    X mean(int[] dimensions);

    X std(int[] dimensions, boolean biasCorrected);

    X variance(int[] dimensions, boolean biasCorrected);

    X std(int[] dimensions);

    X variance(int[] dimensions);

    X max(int[] dimensions);

    X min(int[] dimensions);

    X norm1(int[] dimensions);

    X norm2(int[] dimensions);

    X normmax(int[] dimensions);

    X valueArrayOf(int[] shape);

    X tile(int[] repeat);

    X repeat(int axis);

    X set(X value1);

    X broadcast(int[] shape);

    X eq(X i_y);

    X neq(X i_y);

    X or(X i_y);

    X rollAxis(int axis);

    X cosineSimilarity(X i_y, int... dimensions);

    X euclideanDistance(X i_y, int... dimensions);

    X manhattanDistance(X i_y, int... dimensions);

    X lossBinaryXENT(X i_y, int... dimensions);

    X lossCosineSimilarity(X i_y, int... dimensions);

    X lossHinge(X i_y, int... dimensions);

    X lossKLD(X i_y, int... dimensions);

    X lossL1(X i_y, int... dimensions);

    X lossL2(X i_y, int... dimensions);

    X lossMAE(X i_y, int... dimensions);

    X lossMAPE(X i_y, int... dimensions);

    X lossMSE(X i_y, int... dimensions);

    X lossMCXENT(X i_y, int... dimensions);

    X lossMSLE(X i_y, int... dimensions);

    X lossNegativeLogLikelihood(X i_y, int... dimensions);

    X lossPoisson(X i_y, int... dimensions);

    X lossSquaredHinge(X i_y, int... dimensions);

    DifferentialFunction arg();

    X selu();

    X tanhDerivative(X wrt);

    X seluDerivative(X wrt);
}
