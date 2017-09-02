package org.nd4j.autodiff;


/**
 *
 */
public interface Field extends CommutativeRing {


    ArrayField inverse();


    ArrayField rsubi(ArrayField i_v);

    ArrayField rdivi(ArrayField i_v);

    ArrayField subi(ArrayField i_v);

    ArrayField divi(ArrayField i_v);

    ArrayField inversei();


    ArrayField rsubi(double v);

    ArrayField rdivi(double v);

    ArrayField divi(double v);

    ArrayField rdiv(ArrayField i_v);

    ArrayField div(ArrayField i_v);

    double getReal();

    ArrayField[] args();


    ArrayField rsub(double v);

    ArrayField rdiv(double v);

    ArrayField pow(ArrayField a);

    ArrayField floor();

    ArrayField ceil();

    ArrayField round();

    ArrayField abs();

    ArrayField sqrt();

    ArrayField minus(double v);

    ArrayField prod(double v);

    ArrayField div(double v);

    ArrayField pow(double v);

    ArrayField cos();

    ArrayField acos();

    ArrayField cosh();

    ArrayField acosh();

    ArrayField sin();

    ArrayField asin();

    ArrayField sinh();

    ArrayField asinh();

    ArrayField tan();

    ArrayField atan();

    ArrayField tanh();

    ArrayField atanh();

    ArrayField exp();

    ArrayField log();

    ArrayField log10();

    ArrayField sgn();

    ArrayField pwr(ArrayField y);

    ArrayField pwrs(ArrayField y);

    ArrayField square();

    ArrayField relu();

    ArrayField hardTanh();

    ArrayField hardTanhDerivative(ArrayField wrt);

    ArrayField leakyRelu();

    ArrayField elu();

    ArrayField eluDerivative(ArrayField wrt);

    ArrayField leakyRelu(double cutoff);

    ArrayField leakyReluDerivative();

    ArrayField leakyReluDerivative(ArrayField wrt, double cutoff);

    ArrayField sigmoid();

    ArrayField sigmoidDerivative(ArrayField wrt);

    ArrayField step();

    ArrayField softsign();

    ArrayField softsignDerivative(ArrayField wrt);

    ArrayField softmax();

    ArrayField logSoftmax();

    ArrayField softmaxDerivative(ArrayField wrt);

    ArrayField softplus();

    ArrayField reshape(int[] shape);

    ArrayField transpose();

    ArrayField permute(int[] dimensions);

    ArrayField expandDims(int dim);

    ArrayField sum(int[] dimensions);

    ArrayField prod(int[] dimensions);

    ArrayField mean(int[] dimensions);

    ArrayField std(int[] dimensions, boolean biasCorrected);

    ArrayField variance(int[] dimensions, boolean biasCorrected);

    ArrayField std(int[] dimensions);

    ArrayField variance(int[] dimensions);

    ArrayField max(int[] dimensions);

    ArrayField min(int[] dimensions);

    ArrayField norm1(int[] dimensions);

    ArrayField norm2(int[] dimensions);

    ArrayField normmax(int[] dimensions);

    ArrayField valueArrayOf(int[] shape);

    ArrayField tile(int[] repeat);

    ArrayField repeat(int axis);

    ArrayField set(ArrayField value1);

    ArrayField broadcast(int[] shape);

    ArrayField eq(ArrayField i_y);

    ArrayField neq(ArrayField i_y);

    ArrayField or(ArrayField i_y);

    ArrayField rollAxis(int axis);

    ArrayField cosineSimilarity(ArrayField i_y, int... dimensions);

    ArrayField euclideanDistance(ArrayField i_y, int... dimensions);

    ArrayField manhattanDistance(ArrayField i_y, int... dimensions);

    ArrayField lossBinaryXENT(ArrayField i_y, int... dimensions);

    ArrayField lossCosineSimilarity(ArrayField i_y, int... dimensions);

    ArrayField lossHinge(ArrayField i_y, int... dimensions);

    ArrayField lossKLD(ArrayField i_y, int... dimensions);

    ArrayField lossL1(ArrayField i_y, int... dimensions);

    ArrayField lossL2(ArrayField i_y, int... dimensions);

    ArrayField lossMAE(ArrayField i_y, int... dimensions);

    ArrayField lossMAPE(ArrayField i_y, int... dimensions);

    ArrayField lossMSE(ArrayField i_y, int... dimensions);

    ArrayField lossMCXENT(ArrayField i_y, int... dimensions);

    ArrayField lossMSLE(ArrayField i_y, int... dimensions);

    ArrayField lossNegativeLogLikelihood(ArrayField i_y, int... dimensions);

    ArrayField lossPoisson(ArrayField i_y, int... dimensions);

    ArrayField lossSquaredHinge(ArrayField i_y, int... dimensions);

    ArrayField arg();

    ArrayField selu();

    ArrayField tanhDerivative(ArrayField wrt);

    ArrayField seluDerivative(ArrayField wrt);

    ArrayField max(double v);

    ArrayField min(double v);

    ArrayField fmod(double v);

    ArrayField set(double v);



}
