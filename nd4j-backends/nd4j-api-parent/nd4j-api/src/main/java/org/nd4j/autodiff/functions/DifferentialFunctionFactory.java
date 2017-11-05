package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import lombok.Data;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.*;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.lang.reflect.Method;
import java.util.*;

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
    public Constant val(SDVariable iX) {
        return sameDiff().setupFunction(new Constant(sameDiff(), iX,
                iX.getShape(),new int[]{sameDiff().graph().nextVertexId()}));
    }


    @Override
    public SDVariable var(String iName, SDVariable iX) {
        return sameDiff().setupFunction(SDVariable.builder()
                .shape(iX.getShape())
                .varName(iName)
                .sameDiff(sameDiff())
                .vertexId(new int[]{sameDiff().graph().nextVertexId()})
                .build());
    }

    @Override
    public SDVariable zero(int[] shape) {
        return sameDiff().setupFunction(sameDiff.zero("one-" + UUID.randomUUID().toString(),shape));
    }

    @Override
    public SDVariable one(int[] shape) {
        return sameDiff().setupFunction(sameDiff.one("one-" + UUID.randomUUID().toString(),shape));
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
    public DifferentialFunction neq(DifferentialFunction iX, double i_y) {
        return sameDiff().setupFunction(new ScalarNotEquals(sameDiff(),iX,i_y));

    }

    @Override
    public DifferentialFunction neqi(DifferentialFunction iX, double i_y) {
        return sameDiff().setupFunction(new ScalarNotEquals(sameDiff(),iX,i_y,true));

    }


    @Override
    public DifferentialFunction neqi(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff().setupFunction(new NotEqualTo(sameDiff(),iX,i_y,true));

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
        return sameDiff().setupFunction(new SigmoidDerivative(sameDiff(),iX,wrt));
    }

    @Override
    public DifferentialFunction swish(DifferentialFunction iX) {
        return sameDiff().setupFunction(new Swish(sameDiff(),iX,null));

    }

    @Override
    public DifferentialFunction swishDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff().setupFunction(new SwishDerivative(sameDiff(),iX,wrt));
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
    public DifferentialFunction rollAxis(SDVariable iX, int axis) {
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

    @Override
    public DifferentialFunction gt(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new GreaterThan(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction lt(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new LessThan(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction gti(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new GreaterThan(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction lti(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new LessThan(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction gte(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new GreaterThanOrEqual(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction lte(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new LessThanOrEqual(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction gtei(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new GreaterThanOrEqual(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction ltOrEqi(DifferentialFunction functionInput, DifferentialFunction functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return sameDiff().setupFunction(new LessThanOrEqual(sameDiff(),functionInput,functionInput1,true));
    }



    @Override
    public DifferentialFunction gt(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarGreaterThan(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction lt(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarLessThan(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction gti(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarGreaterThan(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction lti(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarLessThan(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction gte(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarGreaterThanOrEqual(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction lte(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarLessThanOrEqual(sameDiff(),functionInput,functionInput1,false));
    }

    @Override
    public DifferentialFunction gtei(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarGreaterThanOrEqual(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction ltei(DifferentialFunction functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff().setupFunction(new ScalarLessThanOrEqual(sameDiff(),functionInput,functionInput1,true));
    }

    @Override
    public DifferentialFunction eq(DifferentialFunction iX, double i_y) {
        return sameDiff().setupFunction(new ScalarEquals(sameDiff(),iX,i_y));
    }

    @Override
    public DifferentialFunction eqi(DifferentialFunction iX, double i_y) {
        return sameDiff().setupFunction(new ScalarEquals(sameDiff(),iX,i_y,true));
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
     * Adds function edges to the same diff graph
     * based on inputs and the current target op.
     * Note that the op *must* have a vertex id defined.
     * If not, an {@link ND4JIllegalStateException}
     * is thrown
     * @param op the operation to add edges to
     */
    public void addFunctionEdges(DifferentialFunction op) {
        DifferentialFunction[] inputs = op.args();
        for (DifferentialFunction input : inputs) {
            validateFunctionReference(input);
            validateDifferentialFunctionGraph(input);
        }

        if(op.vertexId == null) {
            throw new ND4JIllegalStateException("Op must have a vertex id defined!");
        }


        /**
         * Note here that we need to ensure the vertex is properly added.
         * The output variable creation can create skipped vertices.
         */
        if(sameDiff.graph().getVertex(op.vertexId[0]) == null) {
            SDVariable var = sameDiff.var(op.opName() + "-" + UUID.randomUUID().toString(),op.shape,new ZeroInitScheme('f'),op.vertexId,0);
            NDArrayVertex ndArrayVertex = new NDArrayVertex(sameDiff,op.vertexId[0],0,var);
            sameDiff.graph().addVertex(ndArrayVertex);
        }

        String opName = op.opName();

        List<int[]> outputShapes = op.calculateOutputShape();
        int[] outputVertexIds = new int[outputShapes.size()];
        List<Integer> inputIdsList = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            DifferentialFunction differentialFunction = inputs[i];
            if(differentialFunction instanceof SDVariable || differentialFunction.outputs().size() == 1 && differentialFunction.outputs().get(0) == differentialFunction) {
                inputIdsList.addAll(Ints.asList(differentialFunction.vertexId));
            }
            else {
                List<DifferentialFunction> outputs = differentialFunction.outputs();
                for (DifferentialFunction output : outputs) {
                    if(output == differentialFunction)
                        continue;
                    for (int vertexId : output.getOutputVertexIds()) {
                        if (!inputIdsList.contains(vertexId))
                            inputIdsList.add(vertexId);
                    }
                }
            }


        }


        /**
         * Need to do something about in place operations.
         * Due to how variables are handled, we need to make sure array references are updated properly.
         *
         */
        DifferentialFunction[] outputFunctions = new DifferentialFunction[outputShapes.size()];
        SDVariable[] resultInfo = new SDVariable[outputShapes.size()];
        if(outputShapes.size() > 1) {
            throw new ND4JIllegalStateException("Automatically generating edges assumes *only* 1 output for now. Consider using DynamicCustomOp for multi output");
        }
        for (int i = 0; i < outputShapes.size(); i++) {
            SDVariable variable = sameDiff.var(sameDiff.generateVariableName(opName, false,op.args()),outputShapes.get(i),new ZeroInitScheme('f'),op.getVertexId(),op.depth());
            outputVertexIds[i] = variable.getVertexId()[0];
            resultInfo[i] = variable;
            outputFunctions[i] = variable;

        }


        int[] inputIds = Ints.toArray(inputIdsList);

        Op.Type opType = op.opType();

        String[] vertexIds = sameDiff.generateVertexIds(Ints.concat(inputIds, outputVertexIds));
        OpState opState = OpState.builder()
                .opType(opType).inPlace(op.isInPlace())
                .opName(opName)
                .id(opName + "(" + vertexIds + ")")
                .vertexIds(sameDiff.generateVertexIds(Ints.concat(inputIds, outputVertexIds)))
                .extraArgs(op.getExtraArgs())
                .build();


        /**
         * Create 1 opstate with all of the vertex ids
         * with all inputs and outputs representing the edge.
         */
        sameDiff.graph().addEdge(
                inputIds,
                outputVertexIds,
                opState, true);


        op.opState = opState;

    }

    public void validateDifferentialFunctionsameDiff(
            List<DifferentialFunction> function) {
        for(DifferentialFunction differentialFunction : function)
            validateDifferentialFunctionsameDiff(differentialFunction);
    }



    public void validateDifferentialFunctionsameDiff(
            DifferentialFunction function) {

        Preconditions.checkState(function != null,"Passed in function was null.");
        Preconditions.checkState(function.getSameDiff() == sameDiff);

        Preconditions.checkState(function.getSameDiff() ==
                        this.getSameDiff(),
                "Function applications must be contained " +
                        "in same sameDiff. The left " + function  +
                        " must match this function " + this);
        Preconditions.checkState(sameDiff ==
                this.getSameDiff(),"Function applications m" +
                "ust be " +
                "contained in same sameDiff. The left " + function +" " +
                "must " +
                "match this function " + this);

    }



    public void validateDifferentialFunctionGraph(DifferentialFunction function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }



    public void validateFunctionReference(List<DifferentialFunction> reference) {
        for(int i = 0; i < reference.size(); i++) {
            validateFunctionReference(reference.get(i));
        }

    }


    public void validateFunctionReference(DifferentialFunction reference) {
        if(reference instanceof SDVariable)
            return;

        if(sameDiff.getFunctionForVertexId(reference.getVertexId()) != null) {
            DifferentialFunction get = sameDiff.getFunctionForVertexId(reference.getVertexId());
            Preconditions.checkState(reference.equals(get), "Found invalid reference " + reference + " for vertex id "
                    + reference.getVertexId());
        }


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





}
