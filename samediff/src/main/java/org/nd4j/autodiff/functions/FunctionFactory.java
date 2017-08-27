package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;

/**
 * Created by agibsonccc on 4/9/17.
 */
public interface FunctionFactory<X extends Field<X>> {



    DifferentialFunction<X> invoke(String name, Object[] args);

    Constant<X> val(X iX);


    Variable<X> var(String iName, X iX, PreEvaluator<X> preEvaluator);

    Variable<X> var(String iName, X iX);



    Zero<X> zero(int[] shape);

    One<X> one(int[] shape);


    DifferentialFunction<X> tile(DifferentialFunction<X> iX, int[] repeat);

    DifferentialFunction<X> valueArrayOf(DifferentialFunction<X> iX, int[] shape);

    DifferentialFunction<X> sum(DifferentialFunction<X> i_x, int...dimensions);

    DifferentialFunction<X> prod(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> mean(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> std(DifferentialFunction<X> i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction<X> variance(DifferentialFunction<X> i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction<X> max(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> min(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> norm1(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> norm2(DifferentialFunction<X> i_x,int...dimensions);

    DifferentialFunction<X> normmax(DifferentialFunction<X> i_x,int...dimensions);


    DifferentialFunction<X> expandDims(DifferentialFunction<X> iX, int axis);

    DifferentialFunction<X> abs(DifferentialFunction<X> iX);

    DifferentialFunction<X> neg(DifferentialFunction<X> iX);

    DifferentialFunction<X> cos(DifferentialFunction<X> iX);

    DifferentialFunction<X> sin(DifferentialFunction<X> iX);

    DifferentialFunction<X> tan(DifferentialFunction<X> iX);

    DifferentialFunction<X> permute(DifferentialFunction<X> iX, int... dimensions);

    DifferentialFunction<X> transpose(DifferentialFunction<X> iX);

    DifferentialFunction<X> acos(DifferentialFunction<X> iX);

    DifferentialFunction<X> asin(DifferentialFunction<X> iX);

    DifferentialFunction<X> atan(DifferentialFunction<X> iX);

    DifferentialFunction<X> cosh(DifferentialFunction<X> iX);

    DifferentialFunction<X> sinh(DifferentialFunction<X> iX);

    DifferentialFunction<X> tanh(DifferentialFunction<X> iX);

    DifferentialFunction<ArrayField> tanhDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> wrt);

    DifferentialFunction<X> acosh(DifferentialFunction<X> iX);

    DifferentialFunction<X> asinh(DifferentialFunction<X> iX);

    DifferentialFunction<X> atanh(DifferentialFunction<X> iX);

    DifferentialFunction<X> exp(DifferentialFunction<X> iX);

    DifferentialFunction<X> log(DifferentialFunction<X> iX);

    DifferentialFunction<X> or(DifferentialFunction<X> iX, DifferentialFunction<X> i_y);

    DifferentialFunction<X> eq(DifferentialFunction<X> iX, DifferentialFunction<X> i_y);

    DifferentialFunction<X> neq(DifferentialFunction<X> iX, DifferentialFunction<X> i_y);

    DifferentialFunction<X> pow(DifferentialFunction<X> iX, Constant<X> i_y);

    DifferentialFunction<X> sqrt(DifferentialFunction<X> iX);

    DifferentialFunction<X> square(DifferentialFunction<X> iX);

    DifferentialFunction<X> floor(DifferentialFunction<X> iX);

    DifferentialFunction<X> relu(DifferentialFunction<X> iX, double cutoff);

    DifferentialFunction<X> softmax(DifferentialFunction<X> iX);

    DifferentialFunction<X> hardTanh(DifferentialFunction<X> iX);

    DifferentialFunction<X> hardTanhDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> sigmoid(DifferentialFunction<X> iX);

    DifferentialFunction<X> sigmoidDerivative(DifferentialFunction<X> iX, DifferentialFunction<X> wrt);

    DifferentialFunction<X> sign(DifferentialFunction<X> iX);

    DifferentialFunction<X> broadcast(DifferentialFunction<X> iX, int... shape);

    DifferentialFunction<X> repeat(DifferentialFunction<X> iX, int axis);

    DifferentialFunction<X> softsign(DifferentialFunction<X> iX);

    DifferentialFunction<X> softsignDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> softplus(DifferentialFunction<X> iX);

    DifferentialFunction<X> elu(DifferentialFunction<X> iX);

    DifferentialFunction<X> eluDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> leakyRelu(DifferentialFunction<X> iX, double cutoff);

    DifferentialFunction<X> leakyReluDerivative(DifferentialFunction<X> iX, DifferentialFunction<X> iY, double cutoff);

    DifferentialFunction<X> reshape(DifferentialFunction<X> arrayField, int[] shape);


    DifferentialFunction<X> rollAxis(Variable<X> iX, int axis);

    DifferentialFunction<X> cosineSimilarity(DifferentialFunction<X> iX, DifferentialFunction<X> i_y, int...dimensions);
    DifferentialFunction<X> euclideanDistance(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> manhattanDistance(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossBinaryXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int...dimensions);
    DifferentialFunction<X> lossCosineSimilarity(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossHinge(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossKLD(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossL1(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossL2(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossMAE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossMAPE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossMSE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossMCXENT(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossMSLE(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossNegativeLogLikelihood(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossPoisson(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);
    DifferentialFunction<X> lossSquaredHinge(DifferentialFunction<X> iX, DifferentialFunction<X> i_y,int...dimensions);


    DifferentialFunction<X> mmul(DifferentialFunction<X> x, DifferentialFunction<X> y);
    DifferentialFunction<X> tensorMmul(DifferentialFunction<X> x, DifferentialFunction<X> y, int[][] dimensions);

    DifferentialFunction<X> softmaxDerivative(DifferentialFunction<X> functionInput, DifferentialFunction<X> wrt);

    DifferentialFunction<X> logSoftmax(DifferentialFunction<X> i_v);

    DifferentialFunction<X> selu(DifferentialFunction<X> arg);

    DifferentialFunction<X> seluDerivative(DifferentialFunction<X> arg);
}
