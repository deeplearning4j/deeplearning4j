package org.nd4j.autodiff;

import com.google.common.base.Preconditions;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SameDiff;

import java.lang.reflect.Method;
import java.util.*;

@AllArgsConstructor
@Data
public class ArrayFactory implements AbstractFactory<ArrayField> {

    private SameDiff sameDiff;
    private Map<String,Method> methodNames;

    public ArrayFactory(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
        methodNames = new HashMap<>();
        Method[] methods = getClass().getDeclaredMethods();
        for(Method method : methods)
            methodNames.put(method.getName(),method);
    }



    @Override
    public SameDiff sameDiff() {
        return sameDiff;
    }

    @Override
    public List<String> methodNames() {
        return new ArrayList<>(methodNames.keySet());
    }

    @Override
    public ArrayField invoke(String name, Object[] args) {
        try {
            return (ArrayField) methodNames.get(name).invoke(this,args);
        } catch (Exception e) {
           throw new RuntimeException(e);
        }
    }

   

    @Override
    public ArrayField selu(ArrayField value) {
        return sameDiff.setupArrayField(value.selu());
    }

    @Override
    public ArrayField seluDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.seluDerivative(wrt));
    }

    @Override
    public ArrayField eq(ArrayField i_x, ArrayField i_y) {
        return sameDiff.setupArrayField(i_x.eq(i_y));
    }

    @Override
    public ArrayField neq(ArrayField i_x, ArrayField i_y) {
        return sameDiff.setupArrayField(i_x.neq(i_y));
    }

    @Override
    public ArrayField or(ArrayField i_x, ArrayField i_y) {
        return sameDiff.setupArrayField(i_x.or(i_y));
    }

    @Override
    public ArrayField add(ArrayField i_x, Number value) {
        return sameDiff.setupArrayField(i_x.add(value.doubleValue()));
    }

    @Override
    public ArrayField sub(ArrayField i_x, Number value) {
        return sameDiff.setupArrayField(i_x.minus(value.doubleValue()));
    }



    @Override
    public ArrayField mul(ArrayField i_x, Number value) {
        return sameDiff.setupArrayField(i_x.mul((long) value.doubleValue()));
    }

    @Override
    public ArrayField div(ArrayField i_x, Number value) {
        return sameDiff.setupArrayField(i_x.div(value.doubleValue()));
    }

    @Override
    public ArrayField broadcast(ArrayField i_x, int... shape) {
        return sameDiff.setupArrayField(i_x.broadcast(shape));
    }

    @Override
    public ArrayField repeat(ArrayField i_x, int axis) {
        return sameDiff.setupArrayField(i_x.repeat(axis));
    }

    @Override
    public ArrayField tile(ArrayField i_x, int... repeat) {
        return sameDiff.setupArrayField(i_x.tile(repeat));
    }

    @Override
    public ArrayField sum(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.sum(dimensions));
    }

    @Override
    public ArrayField prod(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.prod(dimensions));
    }

    @Override
    public ArrayField mean(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.mean(dimensions));
    }

    @Override
    public ArrayField std(ArrayField i_x, boolean biasCorrected, int... dimensions) {
        return sameDiff.setupArrayField(i_x.std(dimensions));
    }

    @Override
    public ArrayField variance(ArrayField i_x, boolean biasCorrected, int... dimensions) {
        return sameDiff.setupArrayField(i_x.variance(dimensions));
    }

    @Override
    public ArrayField max(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.max(dimensions));
    }

    @Override
    public ArrayField min(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.min(dimensions));
    }

    @Override
    public ArrayField norm1(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.norm1(dimensions));
    }

    @Override
    public ArrayField norm2(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.norm2(dimensions));
    }

    @Override
    public ArrayField normmax(ArrayField i_x, int... dimensions) {
        return sameDiff.setupArrayField(i_x.normmax(dimensions));
    }

    @Override
    public ArrayField neg(ArrayField i_x) {
        return sameDiff.setupArrayField(i_x.negate());
    }

    @Override
    public ArrayField transpose(ArrayField i_x) {
        return sameDiff.setupArrayField(i_x.transpose());
    }

    @Override
    public ArrayField reshape(ArrayField i_x, int[] shape) {
        return sameDiff.setupArrayField(i_x.reshape(shape));
    }

    @Override
    public ArrayField valueArrayOf(ArrayField i_x, int[] shape) {
        return sameDiff.setupArrayField(i_x.valueArrayOf(shape));
    }

    @Override
    public ArrayField val(double v) {
        // return sameDiff.setupArrayField(Nd4j.valueArrayOf(v,i));
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField abs(ArrayField x) {
        return sameDiff.setupArrayField(x.abs());
    }

    @Override
    public ArrayField min(ArrayField x, ArrayField y) {
       /* return sameDiff.setupArrayField(x.doubleValue() < y.doubleValue() ? new ArrayField(
                x.doubleValue()) : new ArrayField(y.doubleValue()));*/
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField max(ArrayField x, ArrayField y) {
     /*   return sameDiff.setupArrayField(x.doubleValue() > y.doubleValue() ? new ArrayField(
                x.doubleValue()) : new ArrayField(y.doubleValue()));*/
        throw new UnsupportedOperationException();

    }



    @Override
    public ArrayField zero(int[] shape) {
        NDArrayInformation information = NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString()).scalarValue(0.0)
                .id("zero-" + UUID.randomUUID().toString()).owner(null).shape(shape).build();
        return sameDiff.setupArrayField(new ArrayField(new NDArrayVertex(sameDiff(),sameDiff.getGraph().nextVertexId(), information), sameDiff));
    }

    @Override
    public ArrayField one(int[] shape) {
        NDArrayInformation information = NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString()).scalarValue(1.0)
                .id("one-"  + UUID.randomUUID().toString()).owner(null).shape(shape).build();
        return sameDiff.setupArrayField(new ArrayField(new NDArrayVertex(sameDiff(),sameDiff.getGraph().nextVertexId(), information), sameDiff));
    }

    /**
     * Scalar value
     *
     * @param value
     * @return
     */
    @Override
    public ArrayField scalar(double value) {
        NDArrayInformation information = NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString()).scalarValue(value)
                .id(String.valueOf(value)).owner(null).shape(new int[]{1,1}).build();
        return sameDiff.setupArrayField(new ArrayField(new NDArrayVertex(sameDiff,sameDiff.getGraph().nextVertexId(), information), sameDiff));
    }

    @Override
    public ArrayField cos(ArrayField x) {
        return sameDiff.setupArrayField(x.cos());
    }

    @Override
    public ArrayField acos(ArrayField x) {
        return sameDiff.setupArrayField(x.acos());
    }

    @Override
    public ArrayField cosh(ArrayField x) {
        return sameDiff.setupArrayField(x.cosh());
    }

    @Override
    public ArrayField acosh(ArrayField x) {
        return sameDiff.setupArrayField(x.acosh());
    }

    @Override
    public ArrayField sin(ArrayField x) {
        return sameDiff.setupArrayField(x.sin());
    }

    @Override
    public ArrayField asin(ArrayField x) {
        return sameDiff.setupArrayField(x.asin());
    }

    @Override
    public ArrayField sinh(ArrayField x) {
        return sameDiff.setupArrayField(x.sinh());
    }

    @Override
    public ArrayField asinh(ArrayField x) {
        return sameDiff.setupArrayField(x.asinh());
    }

    @Override
    public ArrayField tan(ArrayField x) {
        return sameDiff.setupArrayField(x.tan());
    }

    @Override
    public ArrayField atan(ArrayField x) {
        return sameDiff.setupArrayField(x.atan());
    }

    @Override
    public ArrayField atan2(ArrayField x, ArrayField y) {
        //   return sameDiff.setupArrayField(new ArrayField(Math.atan2(x.doubleValue(), y.doubleValue())));
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField tanh(ArrayField x) {
        return sameDiff.setupArrayField(x.tanh());
    }

    @Override
    public ArrayField tanhDerivative(ArrayField x, ArrayField wrt) {
        return sameDiff.setupArrayField(x.tanhDerivative(wrt));
    }

    @Override
    public ArrayField logSoftmax(ArrayField value) {
        return sameDiff.setupArrayField(value.logSoftmax());
    }

    @Override
    public ArrayField atanh(ArrayField x) {
        return sameDiff.setupArrayField(x.atanh());
    }

    @Override
    public ArrayField exp(ArrayField x) {
        return sameDiff.setupArrayField(x.exp());
    }

    @Override
    public ArrayField log(ArrayField x) {
        return sameDiff.setupArrayField(x.log());
    }

    @Override
    public ArrayField log10(ArrayField x) {
        return sameDiff.setupArrayField(x.log10());
    }

    @Override
    public ArrayField flat(ArrayField x) {
      /*  double xValue = x.doubleValue());
        return sameDiff.setupArrayField(new ArrayField(-xValue + (xValue + xValue) * randomGenerator.nextDouble()));*/
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField mc(ArrayField x, ArrayField y) {
      /*  double max = Math.max(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue())));
        double min = Math.min(x.doubleValue() * (1 + y.doubleValue()),
                x.doubleValue() * (1 - y.doubleValue())));
        return sameDiff.setupArrayField(new ArrayField(min + (max - min) * randomGenerator.nextDouble()));*/
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
        return sameDiff.setupArrayField(x.sgn());
    }

    @Override
    public ArrayField ifx(ArrayField x, ArrayField y, ArrayField z) {
        //return sameDiff.setupArrayField(x.doubleValue() > .5 ? y : z;
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField buf(ArrayField x) {
        //return sameDiff.setupArrayField(x.doubleValue() > .5 ? new ArrayField(1) : new ArrayField(0));
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField inv(ArrayField x) {
        //  return sameDiff.setupArrayField(x.doubleValue() > .5 ? new ArrayField(0) : new ArrayField(1));
        throw new UnsupportedOperationException();


    }

    @Override
    public ArrayField u(ArrayField x) {
        //return sameDiff.setupArrayField(x.doubleValue() > 0 ? new ArrayField(1) : new ArrayField(0));
        throw new UnsupportedOperationException();


    }

    @Override
    public ArrayField uramp(ArrayField x) {
        // return sameDiff.setupArrayField(x.doubleValue() > 0 ? new ArrayField(x.doubleValue()) : new ArrayField(0));
        throw new UnsupportedOperationException();


    }

    @Override
    public ArrayField pow(ArrayField x, ArrayField y) {
        return sameDiff.setupArrayField(x.pow(y));
    }

    @Override
    public ArrayField pwr(ArrayField x, ArrayField y) {
        return sameDiff.setupArrayField(x.pwr(y));
    }

    @Override
    public ArrayField pwrs(ArrayField x, ArrayField y) {
        return sameDiff.setupArrayField(x.pwrs(y));
    }

    @Override
    public ArrayField sqrt(ArrayField x) {
        return sameDiff.setupArrayField(x.sqrt());
    }

    @Override
    public ArrayField square(ArrayField x) {
        return sameDiff.setupArrayField(x.square());
    }

    @Override
    public ArrayField hypot(ArrayField x, ArrayField y) {
        return sameDiff.setupArrayField(x.pow(2).add(y.pow(2)).sqrt());
    }

    @Override
    public ArrayField floor(ArrayField value) {
        return sameDiff.setupArrayField(value.floor());
    }

    @Override
    public ArrayField ceil(ArrayField value) {
        return sameDiff.setupArrayField(value.ceil());
    }

    @Override
    public ArrayField round(ArrayField value) {
        return sameDiff.setupArrayField(value.round());
    }

    @Override
    public ArrayField relu(ArrayField value) {
        return sameDiff.setupArrayField(value.relu());
    }

    @Override
    public ArrayField leakyRelu(ArrayField value, double alpha) {
        return sameDiff.setupArrayField(value.leakyRelu());
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
        return sameDiff.setupArrayField(value.leakyRelu());
    }

    @Override
    public ArrayField leakyReluDerivative(ArrayField value, ArrayField wrt, double alpha) {
        return sameDiff.setupArrayField(value.leakyReluDerivative(wrt,alpha));
    }

    /**
     * Leaky relu with an alpha of
     * 0.01
     *
     * @param value the value to transform
     * @param wrt
     * @return
     */
    @Override
    public ArrayField leakyReluDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.leakyReluDerivative(wrt, 0.001));
    }

    @Override
    public ArrayField hardTanh(ArrayField value) {
        return sameDiff.setupArrayField(value.hardTanh());
    }

    @Override
    public ArrayField hardTanhDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.hardTanhDerivative(wrt));
    }

    @Override
    public ArrayField sigmoid(ArrayField value) {
        return sameDiff.setupArrayField(value.sigmoid());
    }

    @Override
    public ArrayField sigmoidDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.sigmoidDerivative(wrt));
    }

    @Override
    public ArrayField softmax(ArrayField value) {
        return sameDiff.setupArrayField(value.softmax());
    }

    @Override
    public ArrayField elu(ArrayField value) {
        return sameDiff.setupArrayField(value.elu());
    }

    @Override
    public ArrayField eluDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.eluDerivative(wrt));
    }

    @Override
    public ArrayField step(ArrayField value) {
        return sameDiff.setupArrayField(value.step());
    }

    @Override
    public ArrayField sign(ArrayField value) {
        return sameDiff.setupArrayField(value.sgn());
    }

    @Override
    public ArrayField softsign(ArrayField value) {
        return sameDiff.setupArrayField(value.softsign());
    }

    @Override
    public ArrayField softsignDeriviative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.softsignDerivative(wrt));
    }

    @Override
    public ArrayField softplus(ArrayField value) {
        return sameDiff.setupArrayField(value.softplus());
    }

    @Override
    public ArrayField rollAxis(ArrayField value, int axis) {
        return sameDiff.setupArrayField(value.rollAxis(axis));
    }

    @Override
    public ArrayField lossSquaredHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossSquaredHinge(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossPoisson(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossPoisson(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossNegativeLogLikelihood(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossNegativeLogLikelihood(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossMSLE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossMSLE(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossMCXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossMCXENT(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossMSE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossMSE(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossMAPE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossMAPE(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossMAE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossMAE(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossL2(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossL2(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossL1(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossL1(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossKLD(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossKLD(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossHinge(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossCosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossCosineSimilarity(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField lossBinaryXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).lossBinaryXENT(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField manhattanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).manhattanDistance(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField euclideanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).euclideanDistance(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField cosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int[] dimensions) {
        return sameDiff.setupArrayField(iX.getValue(true).cosineSimilarity(i_y.getValue(true),dimensions));
    }

    @Override
    public ArrayField expandDims(ArrayField input, int dim) {
        return sameDiff.setupArrayField(input.expandDims(dim));
    }

    @Override
    public ArrayField mmul(DifferentialFunction<ArrayField> input, DifferentialFunction<ArrayField> y) {
        return sameDiff.setupArrayField(input.getValue(true).mmul(y.getValue(true)));

    }

    @Override
    public ArrayField tensorMmul(DifferentialFunction<ArrayField> arrayField, DifferentialFunction<ArrayField> y, int[][] dimensions) {
        Preconditions.checkState(dimensions != null,"Dimensions must not be null.");
        return sameDiff.setupArrayField(arrayField.getValue(true).tensorMmul(y,dimensions));
    }

    @Override
    public ArrayField permute(ArrayField value, int[] dimensions) {
        Preconditions.checkState(dimensions != null,"Dimensions must not be null.");
        return sameDiff.setupArrayField(value.permute(dimensions));
    }

    @Override
    public String toString() {
        return "ArrayFactory{" +
                "methodNames=" + methodNames +
                '}';
    }

    @Override
    public ArrayField set(ArrayField value, ArrayField value1) {
        return sameDiff.setupArrayField(value.set(value1));
    }

    @Override
    public ArrayField softmaxDerivative(ArrayField value, ArrayField wrt) {
        return sameDiff.setupArrayField(value.softmaxDerivative(wrt));
    }


}
