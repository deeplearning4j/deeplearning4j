package org.nd4j.autodiff;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by agibsonccc on 4/4/17.
 */
@AllArgsConstructor
@Getter
public class ArrayField implements RealNumber<ArrayField> {
    private INDArray input;
    
    @Override
    public ArrayField negate() {
        return new ArrayField(input.neg());
    }

    @Override
    public ArrayField plus(ArrayField i_v) {
        return new ArrayField(input.add(i_v.getInput()));
    }

    @Override
    public ArrayField minus(ArrayField i_v) {
        return new ArrayField(input.sub(i_v.getInput()));
    }

    @Override
    public ArrayField mul(long i_n) {
        return new ArrayField(input.mul(i_n));
    }

    @Override
    public ArrayField mul(ArrayField i_v) {
        return new ArrayField(input.mul(i_v.getInput()));
    }

    @Override
    public ArrayField pow(int i_n) {
        return new ArrayField(Transforms.pow(input,i_n));
    }

    @Override
    public ArrayField inverse() {
        return new ArrayField(InvertMatrix.invert(input,false));
    }

    @Override
    public ArrayField div(ArrayField i_v) {
        return new ArrayField(input.div(i_v.getInput()));
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    public ArrayField pow(ArrayField a) {
        return new ArrayField(Transforms.pow(input,a.getInput()));
    }

    public ArrayField floor() {
        return new ArrayField(Transforms.floor(input));
    }

    public ArrayField ceil() {
        return new ArrayField(Transforms.ceil(input));
    }

    public ArrayField round() {
        return new ArrayField(Transforms.round(input));
    }

    public ArrayField abs() {
        return new ArrayField(Transforms.abs(input));
    }

    public ArrayField sqrt() {
        return new ArrayField(Transforms.sqrt(input));
    }
    // Operators for double

    public ArrayField plus(double v) {
        return new ArrayField(input.add(v));
    }

    public ArrayField minus(double v) {
        return new ArrayField(input.sub(v));
    }

    public ArrayField prod(double v) {
        return new ArrayField(input.mul(v));
    }

    public ArrayField div(double v) {
        return new ArrayField(input.div(v));
    }

    public ArrayField pow(double v) {
        return new ArrayField(Transforms.pow(input,v));
    }

    public ArrayField cos() {
        return new ArrayField(Transforms.cos(input));
    }

    public ArrayField acos() {
        return new ArrayField(Transforms.acos(input));
    }

    public ArrayField cosh() {
        return new ArrayField(Transforms.cosh(input));
    }

    public ArrayField acosh() {
      //  return new ArrayField(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) - 1) + x));
        throw new UnsupportedOperationException();

    }

    public ArrayField sin() {
        return new ArrayField(Transforms.sin(input));
    }

    public ArrayField asin() {
        return new ArrayField(Transforms.asin(input));
    }

    public ArrayField sinh() {
        return new ArrayField(Transforms.sinh(input));
    }

    public ArrayField asinh() {
      //  return new ArrayField(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x));
        throw new UnsupportedOperationException();

    }

    public ArrayField tan() {
        return new ArrayField(Transforms.tanh(input));
    }

    public ArrayField atan() {
        return new ArrayField(Transforms.atan(input));
    }

    public ArrayField tanh() {
        return new ArrayField(Transforms.tanh(input));
    }

    public ArrayField atanh() {
        return new ArrayField(Transforms.atanh(input));
    }

    public ArrayField exp() {
        return new ArrayField(Transforms.exp(input));
    }

    public ArrayField log() {
        return new ArrayField(Transforms.log(input));
    }

    public ArrayField log10() {
        //return new ArrayField(new INDArray(Math.log10(x));
        throw new UnsupportedOperationException();

    }

    public ArrayField sgn() {
        return new ArrayField(Transforms.sign(input));
    }

    public ArrayField pwr(ArrayField y) {
        //return new ArrayField(new INDArray(Math.pow(Math.abs(x), y.doubleValue()));
        throw new UnsupportedOperationException();
    }

    public ArrayField pwrs(ArrayField y) {
       // return new ArrayField(new INDArray(Math.pow(Math.abs(x), y.doubleValue()) * Math.signum(x));
        throw new UnsupportedOperationException();

    }

    public ArrayField square() {
        return new ArrayField(input.mul(input));
    }



}
