package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.EqualTo;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.NotEqualTo;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 *
 */
@Data
public class DifferentialFunctionFactory implements FunctionFactory  {

    protected SameDiff sameDiff;
    private Map<String,Method> methodNames;

    /**
     *
     * @param sameDiff
     */
    public DifferentialFunctionFactory(SameDiff sameDiff) {
        if (sameDiff != null) {
            this.sameDiff = sameDiff;
            methodNames = new HashMap<>();
            Method[] methods = getClass().getDeclaredMethods();
            for(Method method : methods)
                methodNames.put(method.getName().toLowerCase(),method);
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }


    }

    public SameDiff sameDiff() {
        if(sameDiff.graph().getGraphApply() != null) {
            return sameDiff.graph().getGraphApply().getSameDiff();
        }

        return sameDiff;
    }


    @Override
    public DifferentialFunction invoke(String name, Object[] args) {
        try {
            return (DifferentialFunction ) methodNames.get(name).invoke(this,args);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }




    @Override
    public Constant val(NDArrayInformation iX) {
        return sameDiff().setupFunction(new Constant(sameDiff(), iX,
                iX.getShape(),sameDiff().graph().nextVertexId()));
    }


    @Override
    public Variable  var(String iName, NDArrayInformation iX) {
        return sameDiff().setupFunction(new Variable(sameDiff(),iName, iX,sameDiff.graph().nextVertexId()));
    }

    @Override
    public Zero zero(int[] shape) {
        return sameDiff().setupFunction(new Zero(sameDiff(),shape,sameDiff().graph().nextVertexId()));
    }

    @Override
    public Ones one(int[] shape) {
        return sameDiff().setupFunction(new Ones(sameDiff(),shape,sameDiff.graph().nextVertexId()));
    }

    @Override
    public DifferentialFunction tile(DifferentialFunction iX, int[] repeat) {
        return sameDiff().setupFunction(new Tile(sameDiff(),iX,repeat));
    }



    @Override
    public DifferentialFunction sum(DifferentialFunction i_x,
                                    int... dimensions) {
        return sameDiff().setupFunction(new Sum(sameDiff(),i_x,dimensions));
    }

    @Override
    public DifferentialFunction prod(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new Prod(sameDiff(),i_x,dimensions));
    }

    @Override
    public DifferentialFunction mean(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new Mean(sameDiff(),i_x,dimensions));
    }

    @Override
    public DifferentialFunction std(DifferentialFunction i_x,
                                    boolean biasCorrected,
                                    int... dimensions) {
        return sameDiff().setupFunction(new StandardDeviation(sameDiff(),i_x,dimensions,biasCorrected));
    }

    @Override
    public DifferentialFunction variance(DifferentialFunction i_x,
                                         boolean biasCorrected,
                                         int... dimensions) {
        return sameDiff().setupFunction(new  Variance(sameDiff(),i_x,dimensions,biasCorrected));

    }

    @Override
    public DifferentialFunction max(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new Max(sameDiff(),i_x,dimensions));

    }

    @Override
    public DifferentialFunction min(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new Min(sameDiff(),i_x,dimensions));

    }

    @Override
    public DifferentialFunction norm1(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new  Norm1(sameDiff(),i_x,dimensions));

    }

    @Override
    public DifferentialFunction norm2(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new  Norm2(sameDiff(),i_x,dimensions));

    }

    @Override
    public DifferentialFunction normmax(DifferentialFunction i_x, int... dimensions) {
        return sameDiff().setupFunction(new NormMax(sameDiff(),i_x,dimensions));

    }

    /**
     * Handls gradient calculation
     * for all nor types
     * @param func
     * @param input
     * @param type
     * @param axes
     * @return
     */
    public DifferentialFunction doNormGrad(DifferentialFunction func,
                                           DifferentialFunction input,
                                           String type,
                                           int... axes) {

        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);
        DifferentialFunction result;
        if(Shape.isWholeArray(axes)) {
            result = input;
        }
        else if(axes.length > 1) {
            if(axes[0] > axes[1]) {
                axes[0]--;
            }

            result = expandDims(expandDims(mul(div(func,input),func.args()[0]),axes[0]),axes[1]);
        }
        else {
            result = expandDims(mul(div(func,input),func.args()[0]),axes[0]);
        }

        return result;
    }

    @Override
    public DifferentialFunction gradientBackwardsMarker(DifferentialFunction iX) {
        return sameDiff().setupFunction(new GradientBackwardsMarker(sameDiff(),iX,iX));
    }

    @Override
    public DifferentialFunction expandDims(DifferentialFunction iX,int axis) {
        return sameDiff().setupFunction(new ExpandDims(sameDiff(),iX,axis));
    }



    @Override
    public DifferentialFunction abs(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Abs(sameDiff(),iX,null));
    }


    @Override
    public DifferentialFunction neg(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Negative(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction cos(DifferentialFunction iX) {
        return sameDiff().setupFunction(new  Cos(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction sin(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Sin(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction tan(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Tan(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction permute(DifferentialFunction iX, int... dimensions) {
        return sameDiff().setupFunction(new Permute(sameDiff(),iX,dimensions));
    }


    @Override
    public DifferentialFunction transpose(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Transpose(sameDiff(),iX));
    }

    @Override
    public DifferentialFunction acos(DifferentialFunction iX) {
        return sameDiff().setupFunction(new  ACos(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction asin(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ASin(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction atan(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ATan(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction cosh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Cosh(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction sinh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Sinh(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction tanh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Tanh(sameDiff(),iX,null));
    }


    @Override
    public DifferentialFunction tanhDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff().setupFunction(new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(sameDiff(),iX,wrt));
    }

    @Override
    public DifferentialFunction acosh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ACosh(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction asinh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new  ASinh(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction atanh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ATanh(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction exp(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Exp(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction log(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Log(sameDiff(),iX,null));
    }



    @Override
    public DifferentialFunction or(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff().setupFunction(new Or(sameDiff(),iX,i_y));
    }


    @Override
    public DifferentialFunction eq(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff().setupFunction(new EqualTo(sameDiff(),iX,i_y));
    }

    @Override
    public DifferentialFunction neq(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff().setupFunction(new NotEqualTo(sameDiff(),iX,i_y));

    }

    @Override
    public DifferentialFunction pow(DifferentialFunction iX, double i_y) {
        return sameDiff().setupFunction(new ScalarMultiplication(  sameDiff(),iX,i_y));

    }

    @Override
    public DifferentialFunction sqrt(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Sqrt(sameDiff(),iX,null));
    }

    @Override
    public DifferentialFunction square(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Pow(sameDiff(),iX,false,2.0));
    }

    @Override
    public DifferentialFunction floor(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Floor(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction relu(DifferentialFunction iX, double cutoff) {
        return sameDiff().setupFunction(new RectifedLinear(sameDiff(),iX,false,cutoff));

    }



    @Override
    public DifferentialFunction softmax(DifferentialFunction iX) {
        return sameDiff().setupFunction(new SoftMax(sameDiff(),iX,new Object[]{}));

    }

    @Override
    public DifferentialFunction hardTanh(DifferentialFunction iX) {
        return sameDiff().setupFunction(new HardTanh(sameDiff(),iX,null));

    }



    @Override
    public DifferentialFunction hardTanhDerivative(DifferentialFunction iX) {
        return sameDiff().setupFunction(new HardTanhDerivative(sameDiff(),iX,null));

    }




    @Override
    public DifferentialFunction sigmoid(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Sigmoid(sameDiff(),iX,null));

    }


    @Override
    public DifferentialFunction sigmoidDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff().setupFunction(new SigmoidDerivative(sameDiff(),sameDiff().setupFunction(iX),sameDiff().setupFunction(wrt)));
    }

    @Override
    public DifferentialFunction swish(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Swish(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction swishDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff().setupFunction(new SwishDerivative(sameDiff(),sameDiff().setupFunction(iX),sameDiff().setupFunction(wrt)));
    }


    @Override
    public DifferentialFunction sign(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Sign(sameDiff(),iX,null));

    }


    @Override
    public DifferentialFunction broadcast(DifferentialFunction iX, int... shape) {
        return sameDiff().setupFunction(new Broadcast(sameDiff(),iX,shape));
    }

    @Override
    public DifferentialFunction repeat(DifferentialFunction iX, int axis) {
        return sameDiff().setupFunction(new Repeat(sameDiff(),iX,axis));

    }

    @Override
    public DifferentialFunction softsign(DifferentialFunction iX) {
        return sameDiff().setupFunction(new SoftSign(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction softsignDerivative(DifferentialFunction iX) {
        return sameDiff().setupFunction(new SoftSignDerivative(sameDiff(),iX,null));

    }





    @Override
    public DifferentialFunction softplus(DifferentialFunction iX) {
        return sameDiff().setupFunction(new SoftPlus(sameDiff(),iX,null));

    }


    @Override
    public DifferentialFunction elu(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ELU(sameDiff(),iX,null));

    }



    @Override
    public DifferentialFunction eluDerivative(DifferentialFunction iX) {
        return sameDiff().setupFunction(new ELUDerivative(sameDiff(),iX,null));

    }




    @Override
    public DifferentialFunction leakyRelu(DifferentialFunction iX, double cutoff) {
        return sameDiff().setupFunction(new LeakyReLU(sameDiff(),iX,false,cutoff));

    }



    @Override
    public DifferentialFunction leakyReluDerivative(DifferentialFunction iX, DifferentialFunction iY, double cutoff) {
        return sameDiff().setupFunction(new LeakyReLUDerivative(sameDiff(),iX,iY,cutoff));

    }

    @Override
    public DifferentialFunction reshape(DifferentialFunction iX, int[] shape) {
        return sameDiff().setupFunction(new Reshape(sameDiff(),iX,shape));
    }

    @Override
    public DifferentialFunction rollAxis(Variable  iX, int axis) {
        return sameDiff().setupFunction(new RollAxis(sameDiff(),iX,axis));
    }

    @Override
    public DifferentialFunction cosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff().setupFunction(new CosineSimilarity(sameDiff(),iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction euclideanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff().setupFunction(new EuclideanDistance(sameDiff(),iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction manhattanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff().setupFunction(new ManhattanDistance(sameDiff(),iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction lossBinaryXENT(DifferentialFunction iX,
                                               DifferentialFunction i_y,
                                               int... dimensions) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction lossCosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction lossHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossKLD(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossL1(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossL2(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossMAE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossMAPE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossMSE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossMCXENT(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossMSLE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossNegativeLogLikelihood(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossPoisson(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction lossSquaredHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    @Override
    public DifferentialFunction mmul(DifferentialFunction x,
                                     DifferentialFunction y,
                                     MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return sameDiff().setupFunction(new Mmul(sameDiff(),x,y,mMulTranspose));
    }

    @Override
    public DifferentialFunction mmul(DifferentialFunction x,
                                     DifferentialFunction y) {
        return mmul(x,y,MMulTranspose.allFalse());
    }

    @Override
    public DifferentialFunction tensorMmul(DifferentialFunction x,
                                           DifferentialFunction y,
                                           int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return sameDiff().setupFunction(new TensorMmul(sameDiff(),x,y,dimensions));
    }


    @Override
    public DifferentialFunction softmaxDerivative(DifferentialFunction functionInput, DifferentialFunction wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new SoftMaxDerivative(sameDiff(),functionInput,wrt));
    }

    @Override
    public DifferentialFunction logSoftmax(DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return sameDiff().setupFunction(new LogSoftMax(sameDiff(),i_v,null));

    }

    @Override
    public DifferentialFunction logSoftmaxDerivative(DifferentialFunction arg, DifferentialFunction wrt) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff().setupFunction(new SoftMaxDerivative(sameDiff(),arg,wrt));
    }

    @Override
    public DifferentialFunction selu(DifferentialFunction arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff().setupFunction(new SELU(sameDiff(),arg,null));
    }

    @Override
    public DifferentialFunction seluDerivative(DifferentialFunction arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff().setupFunction(new SELUDerivative(sameDiff(),arg,null));
    }

    @Override
    public DifferentialFunction rsub(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new RSubOp(sameDiff(),differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction rdiv(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new RDivOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rdivi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new RDivOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rsubi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new RSubOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction add(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new AddOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction addi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new AddOp(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction sub(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new SubOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction subi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new SubOp(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction mul(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new MulOp(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction muli(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new MulOp(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction div(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new DivOp(sameDiff(),differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction divi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new DivOp(sameDiff(),differentialFunction,i_v,true));
    }

    @Override
    public DifferentialFunction rsub(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarReverseSubtraction(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rdiv(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarReverseDivision(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rdivi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarReverseDivision(sameDiff(),differentialFunction,i_v,true));
    }

    @Override
    public DifferentialFunction rsubi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarReverseSubtraction(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction add(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarAdd(sameDiff(),differentialFunction,i_v,false));
    }

    @Override
    public DifferentialFunction addi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarAdd(sameDiff(),differentialFunction,i_v,true));
    }

    @Override
    public DifferentialFunction sub(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarSubtraction(sameDiff(),differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction subi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarSubtraction(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction mul(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarMultiplication(sameDiff(),differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction muli(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarMultiplication(sameDiff(),differentialFunction,i_v,true));

    }

    @Override
    public DifferentialFunction div(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarDivision(sameDiff(),differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction divi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff().setupFunction(new ScalarDivision(sameDiff(),differentialFunction,i_v,true));
    }

    /**
     *
     * @param func
     * @return
     */
    public int getInputLength(DifferentialFunction func) {
        validateDifferentialFunctionsameDiff(func);
        int[] inputShape = func.arg().shape;
        return ArrayUtil.prod(inputShape);



    }


    /**
     *
     * @param func
     * @param input
     * @param axes
     * @return
     */
    public DifferentialFunction doGradChoose(DifferentialFunction func,
                                             DifferentialFunction input,
                                             int...axes) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);

        DifferentialFunction repeatedGrad = doRepeat(func,input,axes);
        DifferentialFunction resultRepeated = doRepeat(func.args()[0],input,axes);
        DifferentialFunction argMaxLocations = eq(input,resultRepeated);
        return div(mul(argMaxLocations,repeatedGrad),sum(argMaxLocations,axes));


    }


    /**
     *
     * @param func
     * @param input
     * @param axes
     * @return
     */
    public  DifferentialFunction doRepeat(DifferentialFunction func,
                                          DifferentialFunction input,
                                          int...axes) {
        int[] inputShape = input.getResultShape();
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);
        return broadcast(func,inputShape);



    }

    @Override
    public String toString() {
        return "DifferentialFunctionFactory{" +
                "methodNames=" + methodNames +
                '}';
    }

    private void validateDifferentialFunctionsameDiff(
            DifferentialFunction function) {
        Preconditions.checkState(function.getSameDiff() ==
                        this.getSameDiff(),
                "Function applications must be contained " +
                        "in same sameDiff(). The left " + function +"" +
                        " must match this function " + this);
        Preconditions.checkState(sameDiff ==
                this.getSameDiff(),"Function applications m" +
                "ust be " +
                "contained in same sameDiff. The left " + function +" " +
                "must " +
                "match this function " + this);

    }



}
