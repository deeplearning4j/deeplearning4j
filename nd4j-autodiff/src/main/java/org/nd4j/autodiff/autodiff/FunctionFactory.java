package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.Field;

/**
 * Created by agibsonccc on 4/9/17.
 */
public interface FunctionFactory<X extends Field<X>> {
    Constant<X> val(X iX);

    ConstantVector<X> val(X... iX);

    // ZeroVector
    ConstantVector<X> zero(int iSize);

    Variable<X> var(String iName, X iX, PreEvaluator<X> preEvaluator);

    Variable<X> var(String iName, X iX);

    VariableVector<X> var(String iName, X... iX);

    VariableVector<X> var(String iName, int iSize);

    DifferentialVectorFunction<X> function(DifferentialFunction<X>... iX);

    Zero<X> zero();

    One<X> one();

    DifferentialFunction<X> cos(DifferentialFunction<X> iX);

    DifferentialFunction<X> sin(DifferentialFunction<X> iX);

    DifferentialFunction<X> tan(DifferentialFunction<X> iX);

    DifferentialFunction<X> acos(DifferentialFunction<X> iX);

    DifferentialFunction<X> asin(DifferentialFunction<X> iX);

    DifferentialFunction<X> atan(DifferentialFunction<X> iX);

    DifferentialFunction<X> cosh(DifferentialFunction<X> iX);

    DifferentialFunction<X> sinh(DifferentialFunction<X> iX);

    DifferentialFunction<X> tanh(DifferentialFunction<X> iX);

    DifferentialFunction<X> acosh(DifferentialFunction<X> iX);

    DifferentialFunction<X> asinh(DifferentialFunction<X> iX);

    DifferentialFunction<X> atanh(DifferentialFunction<X> iX);

    DifferentialFunction<X> exp(DifferentialFunction<X> iX);

    DifferentialFunction<X> log(DifferentialFunction<X> iX);

    DifferentialFunction<X> pow(DifferentialFunction<X> iX, Constant<X> i_y);

    DifferentialFunction<X> sqrt(DifferentialFunction<X> iX);

    DifferentialFunction<X> square(DifferentialFunction<X> iX);

    DifferentialFunction<X> floor(DifferentialFunction<X> iX);

    DifferentialFunction<X> relu(DifferentialFunction<X> iX);

    DifferentialFunction<X> softmax(DifferentialFunction<X> iX);

    DifferentialFunction<X> hardTanh(DifferentialFunction<X> iX);

    DifferentialFunction<X> hardTanhDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> sigmoid(DifferentialFunction<X> iX);

    DifferentialFunction<X> sigmoidDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> sign(DifferentialFunction<X> iX);

    DifferentialFunction<X> softsign(DifferentialFunction<X> iX);

    DifferentialFunction<X> softsignDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> softplus(DifferentialFunction<X> iX);

    DifferentialFunction<X> elu(DifferentialFunction<X> iX);

    DifferentialFunction<X> eluDerivative(DifferentialFunction<X> iX);

    DifferentialFunction<X> leakyRelu(DifferentialFunction<X> iX, double cutoff);

    DifferentialFunction<X> leakyReluDerivative(DifferentialFunction<X> iX, double cutoff);
}
