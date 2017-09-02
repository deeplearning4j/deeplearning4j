package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.Field;
import org.nd4j.linalg.api.blas.params.MMulTranspose;

/**
 * Created by agibsonccc on 4/9/17.
 */
public interface FunctionFactory<ArrayField extends Field<ArrayField>> {



    DifferentialFunction<ArrayField> invoke(String name, Object[] args);

    Constant val(ArrayField iX);


    Variable var(String iName, ArrayField iX, PreEvaluator preEvaluator);

    Variable var(String iName, ArrayField iX);



    Zero zero(int[] shape);

    One one(int[] shape);


    DifferentialFunction<ArrayField> tile(DifferentialFunction<ArrayField> iX, int[] repeat);

    DifferentialFunction<ArrayField> valueArrayOf(DifferentialFunction<ArrayField> iX, int[] shape);

    DifferentialFunction<ArrayField> sum(DifferentialFunction<ArrayField> i_x, int...dimensions);

    DifferentialFunction<ArrayField> prod(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> mean(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> std(DifferentialFunction<ArrayField> i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction<ArrayField> variance(DifferentialFunction<ArrayField> i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction<ArrayField> max(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> min(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> norm1(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> norm2(DifferentialFunction<ArrayField> i_x,int...dimensions);

    DifferentialFunction<ArrayField> normmax(DifferentialFunction<ArrayField> i_x,int...dimensions);


    DifferentialFunction<ArrayField> expandDims(DifferentialFunction<ArrayField> iX, int axis);

    DifferentialFunction<ArrayField> abs(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> neg(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> cos(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> sin(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> tan(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> permute(DifferentialFunction<ArrayField> iX, int... dimensions);

    DifferentialFunction<ArrayField> transpose(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> acos(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> asin(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> atan(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> cosh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> sinh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> tanh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> tanhDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> wrt);

    DifferentialFunction<ArrayField> acosh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> asinh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> atanh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> exp(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> log(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> or(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y);

    DifferentialFunction<ArrayField> eq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y);

    DifferentialFunction<ArrayField> neq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y);

    DifferentialFunction<ArrayField> pow(DifferentialFunction<ArrayField> iX, double i_y);

    DifferentialFunction<ArrayField> sqrt(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> square(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> floor(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> relu(DifferentialFunction<ArrayField> iX, double cutoff);

    DifferentialFunction<ArrayField> softmax(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> hardTanh(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> hardTanhDerivative(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> sigmoid(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> sigmoidDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> wrt);

    DifferentialFunction<ArrayField> sign(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> broadcast(DifferentialFunction<ArrayField> iX, int... shape);

    DifferentialFunction<ArrayField> repeat(DifferentialFunction<ArrayField> iX, int axis);

    DifferentialFunction<ArrayField> softsign(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> softsignDerivative(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> softplus(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> elu(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> eluDerivative(DifferentialFunction<ArrayField> iX);

    DifferentialFunction<ArrayField> leakyRelu(DifferentialFunction<ArrayField> iX, double cutoff);

    DifferentialFunction<ArrayField> leakyReluDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> iY, double cutoff);

    DifferentialFunction<ArrayField> reshape(DifferentialFunction<ArrayField> arrayField, int[] shape);


    DifferentialFunction<ArrayField> rollAxis(Variable iX, int axis);

    DifferentialFunction<ArrayField> cosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int...dimensions);
    DifferentialFunction<ArrayField> euclideanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> manhattanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossBinaryXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int...dimensions);
    DifferentialFunction<ArrayField> lossCosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossKLD(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossL1(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossL2(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossMAE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossMAPE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossMSE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossMCXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossMSLE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossNegativeLogLikelihood(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossPoisson(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);
    DifferentialFunction<ArrayField> lossSquaredHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y,int...dimensions);


    DifferentialFunction<ArrayField> mmul(DifferentialFunction<ArrayField> x,
                                          DifferentialFunction<ArrayField> y,
                                          MMulTranspose mMulTranspose);

    DifferentialFunction<ArrayField> mmul(DifferentialFunction<ArrayField> x, DifferentialFunction<ArrayField> y);
    DifferentialFunction<ArrayField> tensorMmul(DifferentialFunction<ArrayField> x, DifferentialFunction<ArrayField> y, int[][] dimensions);

    DifferentialFunction<ArrayField> softmaxDerivative(DifferentialFunction<ArrayField> functionInput, DifferentialFunction<ArrayField> wrt);

    DifferentialFunction<ArrayField> logSoftmax(DifferentialFunction<ArrayField> i_v);

    DifferentialFunction<ArrayField> selu(DifferentialFunction<ArrayField> arg);

    DifferentialFunction<ArrayField> seluDerivative(DifferentialFunction<ArrayField> arg);

    DifferentialFunction<ArrayField> rsub(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> rdiv(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> rdivi(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> rsubi(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> addi(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> sub(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> subi(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> muli(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> div(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);
    DifferentialFunction<ArrayField> divi(DifferentialFunction<ArrayField> xDifferentialFunction, DifferentialFunction<ArrayField> i_v);





    DifferentialFunction<ArrayField> rsub(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> rdiv(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> rdivi(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> rsubi(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> addi(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> sub(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> subi(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> muli(DifferentialFunction<ArrayField> xDifferentialFunction,double i_v);
    DifferentialFunction<ArrayField> div(DifferentialFunction<ArrayField> xDifferentialFunction, double i_v);
    DifferentialFunction<ArrayField> divi(DifferentialFunction<ArrayField> xDifferentialFunction,double i_v);

}
