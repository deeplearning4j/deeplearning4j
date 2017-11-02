package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.transforms.Constant;

/**
 * Created by agibsonccc on 4/9/17.
 */
public interface FunctionFactory {



    DifferentialFunction invoke(String name, Object[] args);

    Constant val(SDVariable iX);



    SDVariable var(String iName, SDVariable iX);



    SDVariable zero(int[] shape);

    SDVariable one(int[] shape);


    DifferentialFunction tile(DifferentialFunction iX, int[] repeat);


    DifferentialFunction sum(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction prod(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction mean(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction std(DifferentialFunction i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction variance(DifferentialFunction i_x, boolean biasCorrected, int... dimensions);

    DifferentialFunction max(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction min(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction norm1(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction norm2(DifferentialFunction i_x, int... dimensions);

    DifferentialFunction normmax(DifferentialFunction i_x, int... dimensions);


    DifferentialFunction expandDims(DifferentialFunction iX, int axis);

    DifferentialFunction abs(DifferentialFunction iX);

    DifferentialFunction neg(DifferentialFunction iX);

    DifferentialFunction cos(DifferentialFunction iX);

    DifferentialFunction sin(DifferentialFunction iX);

    DifferentialFunction tan(DifferentialFunction iX);

    DifferentialFunction permute(DifferentialFunction iX, int... dimensions);

    DifferentialFunction transpose(DifferentialFunction iX);

    DifferentialFunction acos(DifferentialFunction iX);

    DifferentialFunction asin(DifferentialFunction iX);

    DifferentialFunction atan(DifferentialFunction iX);

    DifferentialFunction cosh(DifferentialFunction iX);

    DifferentialFunction sinh(DifferentialFunction iX);

    DifferentialFunction tanh(DifferentialFunction iX);

    DifferentialFunction tanhDerivative(DifferentialFunction iX, DifferentialFunction wrt);

    DifferentialFunction acosh(DifferentialFunction iX);

    DifferentialFunction asinh(DifferentialFunction iX);

    DifferentialFunction atanh(DifferentialFunction iX);

    DifferentialFunction exp(DifferentialFunction iX);

    DifferentialFunction log(DifferentialFunction iX);

    DifferentialFunction or(DifferentialFunction iX, DifferentialFunction i_y);

    DifferentialFunction eq(DifferentialFunction iX, DifferentialFunction i_y);

    DifferentialFunction neq(DifferentialFunction iX, double i_y);

    DifferentialFunction neqi(DifferentialFunction iX, double i_y);

    DifferentialFunction neqi(DifferentialFunction iX, DifferentialFunction i_y);

    DifferentialFunction neq(DifferentialFunction iX, DifferentialFunction i_y);

    DifferentialFunction pow(DifferentialFunction iX, double i_y);

    DifferentialFunction sqrt(DifferentialFunction iX);

    DifferentialFunction square(DifferentialFunction iX);

    DifferentialFunction floor(DifferentialFunction iX);

    DifferentialFunction relu(DifferentialFunction iX, double cutoff);

    DifferentialFunction softmax(DifferentialFunction iX);

    DifferentialFunction hardTanh(DifferentialFunction iX);

    DifferentialFunction hardTanhDerivative(DifferentialFunction iX);

    DifferentialFunction sigmoid(DifferentialFunction iX);

    DifferentialFunction sigmoidDerivative(DifferentialFunction iX, DifferentialFunction wrt);

    DifferentialFunction swish(DifferentialFunction iX);

    DifferentialFunction swishDerivative(DifferentialFunction iX, DifferentialFunction wrt);

    DifferentialFunction sign(DifferentialFunction iX);

    DifferentialFunction broadcast(DifferentialFunction iX, int... shape);

    DifferentialFunction repeat(DifferentialFunction iX, int axis);

    DifferentialFunction softsign(DifferentialFunction iX);

    DifferentialFunction softsignDerivative(DifferentialFunction iX);

    DifferentialFunction softplus(DifferentialFunction iX);

    DifferentialFunction elu(DifferentialFunction iX);

    DifferentialFunction eluDerivative(DifferentialFunction iX);

    DifferentialFunction leakyRelu(DifferentialFunction iX, double cutoff);

    DifferentialFunction leakyReluDerivative(DifferentialFunction iX, DifferentialFunction iY, double cutoff);

    DifferentialFunction reshape(DifferentialFunction NDArrayInformation, int[] shape);


    DifferentialFunction gradientBackwardsMarker(DifferentialFunction iX);


    DifferentialFunction rollAxis(SDVariable iX, int axis);

    DifferentialFunction cosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction euclideanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction manhattanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossBinaryXENT(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossCosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossKLD(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossL1(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossL2(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossMAE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossMAPE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossMSE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossMCXENT(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossMSLE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossNegativeLogLikelihood(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossPoisson(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);
    DifferentialFunction lossSquaredHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions);


    DifferentialFunction mmul(DifferentialFunction x,
                              DifferentialFunction y,
                              MMulTranspose mMulTranspose);

    DifferentialFunction mmul(DifferentialFunction x, DifferentialFunction y);
    DifferentialFunction tensorMmul(DifferentialFunction x, DifferentialFunction y, int[][] dimensions);

    DifferentialFunction softmaxDerivative(DifferentialFunction functionInput, DifferentialFunction wrt);

    DifferentialFunction logSoftmax(DifferentialFunction i_v);

    DifferentialFunction logSoftmaxDerivative(DifferentialFunction arg, DifferentialFunction wrt);


    DifferentialFunction selu(DifferentialFunction arg);

    DifferentialFunction seluDerivative(DifferentialFunction arg);

    DifferentialFunction rsub(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction rdiv(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction rdivi(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction rsubi(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction add(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction addi(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction sub(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction subi(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction mul(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction muli(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction div(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);
    DifferentialFunction divi(DifferentialFunction xDifferentialFunction, DifferentialFunction i_v);





    DifferentialFunction rsub(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction rdiv(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction rdivi(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction rsubi(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction add(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction addi(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction sub(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction subi(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction mul(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction muli(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction div(DifferentialFunction xDifferentialFunction, double i_v);
    DifferentialFunction divi(DifferentialFunction xDifferentialFunction, double i_v);

    DifferentialFunction gt(DifferentialFunction functionInput, DifferentialFunction functionInput1);
    DifferentialFunction lt(DifferentialFunction functionInput, DifferentialFunction functionInput1);

    DifferentialFunction gti(DifferentialFunction functionInput, DifferentialFunction functionInput1);
    DifferentialFunction lti(DifferentialFunction functionInput, DifferentialFunction functionInput1);

    DifferentialFunction gte(DifferentialFunction functionInput, DifferentialFunction functionInput1);
    DifferentialFunction lte(DifferentialFunction functionInput, DifferentialFunction functionInput1);

    DifferentialFunction gtei(DifferentialFunction functionInput, DifferentialFunction functionInput1);
    DifferentialFunction ltOrEqi(DifferentialFunction functionInput, DifferentialFunction functionInput1);

    DifferentialFunction gt(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction lt(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction gti(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction lti(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction gte(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction lte(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction gtei(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction ltei(DifferentialFunction functionInput, double functionInput1);

    DifferentialFunction eq(DifferentialFunction iX, double i_y);

    DifferentialFunction eqi(DifferentialFunction iX, double i_y);
}
