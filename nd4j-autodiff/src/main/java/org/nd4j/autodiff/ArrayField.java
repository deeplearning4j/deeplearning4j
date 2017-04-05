package org.nd4j.autodiff;

import com.google.common.base.Function;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.SubOp;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by agibsonccc on 4/4/17.
 */
@AllArgsConstructor
@Getter
public class ArrayField implements Field<ArrayField> {
    private INDArray input;
    private Function<INDArray,INDArray> result;
    private Op op;

    public ArrayField(Op op) {
        this.op = op;
    }

    public ArrayField(INDArray arr) {
        this.input = arr;
        this.op = new Identity(input);
    }

    @Override
    public ArrayField negate() {
        return new ArrayField(new Negative(input));
    }

    @Override
    public ArrayField plus(ArrayField i_v) {
        return new ArrayField(new AddOp(input,i_v.getInput()));
    }

    @Override
    public ArrayField minus(ArrayField i_v) {
        return new ArrayField(new SubOp(input,i_v.getInput()));
    }

    @Override
    public ArrayField mul(long i_n) {
        return new ArrayField(new ScalarMultiplication(input,i_n));
    }

    @Override
    public ArrayField mul(ArrayField i_v) {
        return new ArrayField(new MulOp(input,i_v.getInput()));
    }

    @Override
    public ArrayField pow(int i_n) {
        return new ArrayField(new Pow(input,i_n));
    }

    @Override
    public ArrayField inverse() {
     //   return new ArrayField(InvertMatrix.invert(input,false));
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField div(ArrayField i_v) {
        return new ArrayField(new DivOp(input,i_v.getInput()));
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    public ArrayField pow(ArrayField a) {
        return new ArrayField(new Pow(input,a.getInput(),input,input.length(),1));
    }

    public ArrayField floor() {
        return new ArrayField(new Floor(input));
    }

    public ArrayField ceil() {
        return new ArrayField(new Ceil(input));
    }

    public ArrayField round() {
        return new ArrayField(new Round(input));
    }

    public ArrayField abs() {
        return new ArrayField(new Abs(input));
    }

    public ArrayField sqrt() {
        return new ArrayField(new Sqrt(input));
    }
    // Operators for double

    public ArrayField plus(double v) {
        return new ArrayField(new ScalarAdd(input,v));
    }

    public ArrayField minus(double v) {
        return new ArrayField(new ScalarSubtraction(input,v));
    }

    public ArrayField prod(double v) {
        return new ArrayField(new ScalarMultiplication(input,v));
    }

    public ArrayField div(double v) {
        return new ArrayField(new ScalarDivision(input,v));
    }

    public ArrayField pow(double v) {
        return new ArrayField(new Pow(input,v));
    }

    public ArrayField cos() {
        return new ArrayField(new Cos(input));
    }

    public ArrayField acos() {
        return new ArrayField(new ACos(input));
    }

    public ArrayField cosh() {
        return new ArrayField(new Cosh(input));
    }

    public ArrayField acosh() {
      //  return new ArrayField(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) - 1) + x));
        throw new UnsupportedOperationException();

    }

    public ArrayField sin() {
        return new ArrayField(new Sin(input));
    }

    public ArrayField asin() {
        return new ArrayField(new ASin(input));
    }

    public ArrayField sinh() {
        return new ArrayField(new Sinh(input));
    }

    public ArrayField asinh() {
      //  return new ArrayField(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x));
        throw new UnsupportedOperationException();

    }

    public ArrayField tan() {
        return new ArrayField(new Tan(input));
    }

    public ArrayField atan() {
        return new ArrayField(new ATan(input));
    }

    public ArrayField tanh() {
        return new ArrayField(new Tanh(input));
    }

    public ArrayField atanh() {
        return new ArrayField(new ATanh(input));
    }

    public ArrayField exp() {
        return new ArrayField(new Exp(input));
    }

    public ArrayField log() {
        return new ArrayField(new Log(input));
    }

    public ArrayField log10() {
        //return new ArrayField(new INDArray(Math.log10(x));
        throw new UnsupportedOperationException();

    }

    public ArrayField sgn() {
        return new ArrayField(new Sign(input));
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
        return new ArrayField(new MulOp(input,input));
    }

    public ArrayField relu() {
        return new ArrayField(new RectifedLinear(input));
    }

    public ArrayField hardTanh() {
        return new ArrayField(new HardTanh(input));
    }

    public ArrayField hardTanhDerivative() {
        return new ArrayField(new HardTanhDerivative(input));
    }

    public ArrayField leakyRelu() {
        return new ArrayField(new LeakyReLU(input));
    }

    public ArrayField elu() {
        return new ArrayField(new ELU(input));
    }
    public ArrayField eluDerivative() {
        return new ArrayField(new ELUDerivative(input));
    }



    public ArrayField leakyRelu(double cutoff)  {
        return new ArrayField(new LeakyReLU(input,cutoff));
    }

    public ArrayField leakyReluDerivative() {
        return new ArrayField(new LeakyReLU(input));
    }

    public ArrayField leakyReluDerivative(double cutoff)  {
        return new ArrayField(new LeakyReLUDerivative(input,cutoff));
    }


    public ArrayField sigmoid() {
        return new ArrayField(new Sigmoid(input));
    }

    public ArrayField sigmoidDerivative() {
        return new ArrayField(new SigmoidDerivative(input));
    }

    public ArrayField step() {
        return new ArrayField(new Step(input));
    }


    public ArrayField softsign() {
        return new ArrayField(new SoftSign(input));
    }

    public ArrayField softsignDerivative() {
        return new ArrayField(new SoftSignDerivative(input));
    }


    public ArrayField softmax() {
        return new ArrayField(new SoftMax(input));
    }


    public ArrayField softplus() {
        return new ArrayField(new SoftPlus(input));
    }


    @Override
    public String toString() {
        return "ArrayField{" +
                "input=" + input +
                '}';
    }
}
