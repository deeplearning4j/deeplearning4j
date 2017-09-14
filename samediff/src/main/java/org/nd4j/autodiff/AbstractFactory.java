package org.nd4j.autodiff;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;

import java.util.List;

/**
 *
 */
public interface AbstractFactory
        extends AbstractIdentityFactory {


    /**
     *
     * @return
     */
    SameDiff sameDiff();

    List<String> methodNames();

    ArrayField invoke(String name,Object[] args);


    ArrayField selu(ArrayField value);

    ArrayField eq(ArrayField i_x, ArrayField i_y);

    ArrayField neq(ArrayField i_x, ArrayField i_y);

    ArrayField or(ArrayField i_x, ArrayField i_y);


    ArrayField add(ArrayField i_x,Number value);
    ArrayField sub(ArrayField i_x,Number value);
    ArrayField mul(ArrayField i_x,Number value);
    ArrayField div(ArrayField i_x,Number value);

    ArrayField broadcast(ArrayField i_x,int...shape);

    ArrayField repeat(ArrayField i_x,int axis);

    ArrayField tile(ArrayField i_x,int...repeat);

    ArrayField sum(ArrayField i_x,int...dimensions);
    ArrayField prod(ArrayField i_x,int...dimensions);
    ArrayField mean(ArrayField i_x,int...dimensions);
    ArrayField std(ArrayField i_x, boolean biasCorrected, int... dimensions);
    ArrayField variance(ArrayField i_x, boolean biasCorrected, int... dimensions);
    ArrayField max(ArrayField i_x,int...dimensions);
    ArrayField min(ArrayField i_x,int...dimensions);
    ArrayField norm1(ArrayField i_x,int...dimensions);
    ArrayField norm2(ArrayField i_x,int...dimensions);
    ArrayField normmax(ArrayField i_x,int...dimensions);


    ArrayField neg(ArrayField i_x);

    ArrayField transpose(ArrayField i_x);

    ArrayField reshape(ArrayField i_x, int[] shape);


    ArrayField valueArrayOf(ArrayField i_x,int[] shape);

    ArrayField val(double i_v);

    ArrayField abs(ArrayField i_x);

    ArrayField min(ArrayField i_x, ArrayField i_y);
    ArrayField max(ArrayField i_x, ArrayField i_y);

    ArrayField cos(ArrayField i_x);
    ArrayField acos(ArrayField i_x);
    ArrayField cosh(ArrayField i_x);
    ArrayField acosh(ArrayField i_x);

    ArrayField sin(ArrayField i_x);
    ArrayField asin(ArrayField i_x);
    ArrayField sinh(ArrayField i_x);
    ArrayField asinh(ArrayField i_x);

    ArrayField tan(ArrayField i_x);
    ArrayField atan(ArrayField i_x);
    ArrayField atan2(ArrayField i_x, ArrayField i_y);
    ArrayField tanh(ArrayField i_x);
    ArrayField atanh(ArrayField i_x);

    ArrayField exp(ArrayField i_x);
    ArrayField log(ArrayField i_x);

    ArrayField log10(ArrayField i_x);

    ArrayField flat(ArrayField i_x);
    ArrayField mc(ArrayField i_x, ArrayField i_y);
    ArrayField rand(ArrayField i_x);
    ArrayField random(ArrayField i_x);
    ArrayField gauss(ArrayField i_x);

    ArrayField sgn(ArrayField i_x);

    ArrayField ifx(ArrayField i_x, ArrayField i_y, ArrayField i_z);

    ArrayField buf(ArrayField i_x);
    ArrayField inv(ArrayField i_x);
    ArrayField u(ArrayField i_x);
    ArrayField uramp(ArrayField i_x);

    ArrayField pow(ArrayField i_x, ArrayField i_y);
    ArrayField pwr(ArrayField i_x, ArrayField i_y);
    ArrayField pwrs(ArrayField i_x, ArrayField i_y);
    ArrayField sqrt(ArrayField i_x);
    ArrayField square(ArrayField i_x);
    ArrayField hypot(ArrayField i_x, ArrayField i_y);

    ArrayField floor(ArrayField value);
    ArrayField ceil(ArrayField value);
    ArrayField round(ArrayField value);

    ArrayField relu(ArrayField value);

    ArrayField leakyRelu(ArrayField value,double alpha);

    /**
     * Leaky relu with an alpha of
     * 0.01
     * @param value the value to transform
     * @return
     */
    ArrayField leakyRelu(ArrayField value);

    ArrayField leakyReluDerivative(ArrayField value, ArrayField wrt, double alpha);

    /**
     * Leaky relu with an alpha of
     * 0.01
     * @param value the value to transform
     * @param wrt
     * @return
     */
    ArrayField leakyReluDerivative(ArrayField value, ArrayField wrt);


    ArrayField hardTanh(ArrayField value);

    ArrayField hardTanhDerivative(ArrayField value, ArrayField wrt);

    ArrayField sigmoid(ArrayField value);

    ArrayField sigmoidDerivative(ArrayField value, ArrayField wrt);


    ArrayField softmax(ArrayField value);

    ArrayField elu(ArrayField value);

    ArrayField eluDerivative(ArrayField value, ArrayField wrt);

    ArrayField step(ArrayField value);

    ArrayField sign(ArrayField value);

    ArrayField softsign(ArrayField value);

    ArrayField softsignDeriviative(ArrayField value, ArrayField wrt);

    ArrayField softplus(ArrayField value);

    ArrayField rollAxis(ArrayField value, int axis);

    ArrayField lossSquaredHinge(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossPoisson(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossNegativeLogLikelihood(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossMSLE(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossMCXENT(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossMSE(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossMAPE(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossMAE(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossL2(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossL1(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossKLD(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossHinge(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossCosineSimilarity(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField lossBinaryXENT(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField manhattanDistance(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField euclideanDistance(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField cosineSimilarity(DifferentialFunction iArrayField, DifferentialFunction i_y, int[] dimensions);

    ArrayField expandDims(ArrayField input,int dim);

    ArrayField mmul(DifferentialFunction arrayField, DifferentialFunction y);

    ArrayField mmul(DifferentialFunction input, DifferentialFunction y, MMulTranspose mMulTranspose);

    ArrayField tensorMmul(DifferentialFunction arrayField, DifferentialFunction y, int[][] dimensions);

    /*

     */
    ArrayField permute(ArrayField value, int[] dimensions);

    ArrayField set(ArrayField value, ArrayField value1);

    ArrayField softmaxDerivative(ArrayField value, ArrayField wrt);

    ArrayField seluDerivative(ArrayField value, ArrayField wrt);

    ArrayField tanhDerivative(ArrayField value, ArrayField wrt);



    ArrayField logSoftmax(ArrayField value);

    ArrayField gradientBackwardsMarker(ArrayField value, ArrayField value1);
}
