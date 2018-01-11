package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
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

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 *
 */
@Data
public class DifferentialFunctionFactory   {

    protected SameDiff sameDiff;
    private static Map<String,Method> methodNames;

    /**
     *
     * @param sameDiff
     */
    public DifferentialFunctionFactory(SameDiff sameDiff) {
        if (sameDiff != null) {
            this.sameDiff = sameDiff;
            if(methodNames == null) {
                methodNames = new HashMap<>();
                Method[] methods = getClass().getDeclaredMethods();
                for (Method method : methods)
                    methodNames.put(method.getName().toLowerCase(), method);
            }
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }


    }

    public SameDiff sameDiff() {
        return sameDiff;
    }



    public SDVariable invoke(String name, Object[] args) {
        try {
            return (SDVariable ) methodNames.get(name).invoke(this,args);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }





    public Constant val(SDVariable iX) {
        return new Constant(sameDiff(), iX,
                iX.getShape());
    }



    public SDVariable var(String iName, SDVariable iX) {
        return SDVariable.builder()
                .shape(iX.getShape())
                .varName(iName)
                .sameDiff(sameDiff())
                .build();
    }


    public SDVariable zero(int[] shape) {
        return sameDiff.zero("one-" + UUID.randomUUID().toString(),shape);
    }

    public SDVariable zerosLike(String name, SDVariable input){
        validateDifferentialFunctionsameDiff(input);
        return new ZerosLike(name, sameDiff(),input).outputVariables()[0];
    }


    public SDVariable one(int[] shape) {
        return sameDiff.one("one-" + UUID.randomUUID().toString(),shape);
    }

    public SDVariable onesLike(String name, SDVariable input){
        validateDifferentialFunctionsameDiff(input);
        return new OnesLike(name, sameDiff(),input).outputVariables()[0];
    }




    public SDVariable tile(SDVariable iX, int[] repeat) {
        if(repeat == null) {
            throw new ND4JIllegalStateException("Repeat must not be null!");
        }
        return new Tile(sameDiff(),iX,repeat).outputVariables()[0];
    }




    public SDVariable sum(SDVariable i_x,
                          int... dimensions) {
        return new Sum(sameDiff(),i_x,dimensions).outputVariables()[0];
    }


    public SDVariable prod(SDVariable i_x, int... dimensions) {
        return new Prod(sameDiff(),i_x,dimensions).outputVariables()[0];
    }


    public SDVariable mean(SDVariable i_x, int... dimensions) {
        return new Mean(sameDiff(),i_x,dimensions).outputVariables()[0];
    }


    public SDVariable std(SDVariable i_x,
                          boolean biasCorrected,
                          int... dimensions) {
        return new StandardDeviation(sameDiff(),i_x,dimensions,biasCorrected).outputVariables()[0];
    }


    public SDVariable variance(SDVariable i_x,
                               boolean biasCorrected,
                               int... dimensions) {
        return new  Variance(sameDiff(),i_x,dimensions,biasCorrected).outputVariables()[0];

    }


    public SDVariable max(SDVariable i_x, int... dimensions) {
        return new Max(sameDiff(),i_x,dimensions).outputVariables()[0];

    }


    public SDVariable min(SDVariable i_x, int... dimensions) {
        return new Min(sameDiff(),i_x,dimensions).outputVariables()[0];

    }


    public SDVariable norm1(SDVariable i_x, int... dimensions) {
        return new  Norm1(sameDiff(),i_x,dimensions).outputVariables()[0];

    }


    public SDVariable norm2(SDVariable i_x, int... dimensions) {
        return new  Norm2(sameDiff(),i_x,dimensions).outputVariables()[0];

    }


    public SDVariable normmax(SDVariable i_x, int... dimensions) {
        return new NormMax(sameDiff(),i_x,dimensions).outputVariables()[0];

    }

    /**
     * Add 1s as required to the array make an array possible to be broadcast with the original (pre-reduce) array.
     *
     * Example: if doing [a,b,c].sum(1), result is [a,c]. To 'undo' this in a way that can be auto-broadcast,
     * we want to expand as required - i.e., [a,c] -> [a,1,c] which can be auto-broadcast with the original [a,b,c].
     * This is typically only used with reduction operations backprop.
     *
     * @param origRank   Rank of the original array, before the reduction was executed
     * @param reduceDims Dimensions that the original array was reduced from
     * @param toExpand   Array to add 1s to the shape to (such that it can be
     * @return Reshaped array.
     */
    public SDVariable reductionBroadcastableWithOrigShape(int origRank, int[] reduceDims, SDVariable toExpand){
        if(Shape.isWholeArray(origRank, reduceDims)){
            //Output is [1,1] which is already broadcastable
            return toExpand;
        } else if(origRank == 2 && reduceDims.length == 1){
            //In this case: [a,b] -> [1,b] or [a,b] -> [a,1]
            //both are already broadcastable
            return toExpand;
        } else {
            //Example: [a,b,c].sum(1) -> [a,c]... want [a,1,c]
            for(int d : reduceDims){
                toExpand = sameDiff().expandDims(toExpand, d);
            }
            return toExpand;
        }
    }


    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return new GradientBackwardsMarker(sameDiff(),iX,sameDiff.scalar(iX.getVarName() + "-pairgrad" ,1.0)).outputVariables()[0];
    }

    public SDVariable abs(SDVariable iX) {
        return new Abs(sameDiff(),iX,null).outputVariables()[0];
    }



    public SDVariable neg(SDVariable iX) {
        return new Negative(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable cos(SDVariable iX) {
        return new  Cos(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable sin(SDVariable iX) {
        return new Sin(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable tan(SDVariable iX) {
        return new Tan(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable permute(SDVariable iX, int... dimensions) {
        return new Permute(sameDiff(),iX,dimensions).outputVariables()[0];
    }



    public SDVariable transpose(SDVariable iX) {
        return new Transpose(sameDiff(),iX).outputVariables()[0];
    }


    public SDVariable acos(SDVariable iX) {
        return new  ACos(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable asin(SDVariable iX) {
        return new ASin(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable atan(SDVariable iX) {
        return new ATan(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable cosh(SDVariable iX) {
        return new Cosh(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable sinh(SDVariable iX) {
        return new Sinh(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable tanh(SDVariable iX) {
        return new Tanh(sameDiff(),iX,null).outputVariables()[0];
    }



    public SDVariable tanhDerivative(SDVariable iX, SDVariable wrt) {
        return new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(sameDiff(),iX,wrt).outputVariables()[0];
    }


    public SDVariable acosh(SDVariable iX) {
        return new ACosh(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable asinh(SDVariable iX) {
        return new  ASinh(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable atanh(SDVariable iX) {
        return new ATanh(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable exp(SDVariable iX) {
        return new Exp(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable log(SDVariable iX) {
        return new Log(sameDiff(),iX,null).outputVariables()[0];
    }




    public SDVariable or(SDVariable iX, SDVariable i_y) {
        return new Or(sameDiff(),iX,i_y).outputVariables()[0];
    }



    public SDVariable eq(SDVariable iX, SDVariable i_y) {
        return new EqualTo(sameDiff(),new SDVariable[]{iX,i_y},false).outputVariables()[0];
    }



    public SDVariable neq(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(),iX,i_y).outputVariables()[0];

    }


    public SDVariable neqi(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(),iX,i_y,true).outputVariables()[0];

    }



    public SDVariable neqi(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(),new SDVariable[]{iX,i_y},true).outputVariables()[0];

    }

    public SDVariable neq(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(),new SDVariable[]{iX,i_y},false).outputVariables()[0];

    }




    public SDVariable pow(SDVariable iX, double i_y) {
        return new Pow(sameDiff(),iX,false,i_y).outputVariables()[0];

    }


    public SDVariable sqrt(SDVariable iX) {
        return new Sqrt(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable square(SDVariable iX) {
        return new Pow(sameDiff(),iX,false,2.0).outputVariables()[0];
    }


    public SDVariable cube(SDVariable iX) {
        return new Cube(sameDiff(),iX,null).outputVariables()[0];
    }

    public SDVariable cubeDerivative(SDVariable iX) {
        return new CubeDerivative(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable floor(SDVariable iX) {
        return new Floor(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable relu(SDVariable iX, double cutoff) {
        return new RectifedLinear(sameDiff(),iX,false,cutoff).outputVariables()[0];
    }




    public SDVariable softmax(SDVariable iX) {
        return new SoftMax(sameDiff(),new SDVariable[]{iX}).outputVariables()[0];
    }


    public SDVariable hardTanh(SDVariable iX) {
        return new HardTanh(sameDiff(),iX,null).outputVariables()[0];
    }




    public SDVariable hardTanhDerivative(SDVariable iX) {
        return new HardTanhDerivative(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable sigmoid(SDVariable iX) {
        return new Sigmoid(sameDiff(),iX,null).outputVariables()[0];
    }


    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new SigmoidDerivative(sameDiff(),iX,wrt).outputVariables()[0];
    }


    public SDVariable logSigmoid(SDVariable iX) {
        return new LogSigmoid(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable logSigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new LogSigmoidDerivative(sameDiff(),iX,wrt).outputVariables()[0];
    }

    public SDVariable powDerivative(SDVariable iX, SDVariable wrt, double pow) {
        return new PowDerivative(sameDiff(),iX,wrt, pow).outputVariables()[0];
    }


    public SDVariable swish(SDVariable iX) {
        return new Swish(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable swishDerivative(SDVariable iX, SDVariable wrt) {
        return new SwishDerivative(sameDiff(),iX,wrt).outputVariables()[0];
    }



    public SDVariable sign(SDVariable iX) {
        return new Sign(sameDiff(),iX,null).outputVariables()[0];

    }



    public SDVariable expandDims(SDVariable iX,int axis) {
        return new ExpandDims(sameDiff(),new SDVariable[]{iX},axis).outputVariables()[0];
    }


    public SDVariable broadcast(SDVariable iX, int... shape) {
        return new Broadcast(sameDiff(),iX,shape).outputVariables()[0];
    }


    public SDVariable repeat(SDVariable iX, int axis) {
        return new Repeat(sameDiff(),iX,axis).outputVariables()[0];

    }


    public SDVariable softsign(SDVariable iX) {
        return new SoftSign(sameDiff(),iX,null).outputVariables()[0];

    }


    public SDVariable softsignDerivative(SDVariable iX) {
        return new SoftSignDerivative(sameDiff(),iX,null).outputVariables()[0];

    }






    public SDVariable softplus(SDVariable iX) {
        return new SoftPlus(sameDiff(),iX,null).outputVariables()[0];

    }



    public SDVariable elu(SDVariable iX) {
        return new ELU(sameDiff(),iX,null).outputVariables()[0];

    }




    public SDVariable eluDerivative(SDVariable iX) {
        return new ELUDerivative(sameDiff(),iX,null).outputVariables()[0];

    }





    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return new LeakyReLU(sameDiff(),iX,false,cutoff).outputVariables()[0];

    }




    public SDVariable leakyReluDerivative(SDVariable iX, SDVariable iY, double cutoff) {
        return new LeakyReLUDerivative(sameDiff(),iX,iY,cutoff).outputVariables()[0];

    }


    public SDVariable reshape(SDVariable iX, int[] shape) {
        return new Reshape(sameDiff(),iX,shape).outputVariables()[0];
    }


    public SDVariable rollAxis(SDVariable iX, int axis) {
        return new RollAxis(sameDiff(),iX,axis).outputVariables()[0];
    }


    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new CosineSimilarity(sameDiff(),iX,i_y,dimensions).outputVariables()[0];
    }


    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new EuclideanDistance(sameDiff(),iX,i_y,dimensions).outputVariables()[0];
    }


    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new ManhattanDistance(sameDiff(),iX,i_y,dimensions).outputVariables()[0];
    }


    public SDVariable lossBinaryXENT(SDVariable iX,
                                     SDVariable i_y,
                                     int... dimensions) {
        throw new UnsupportedOperationException();
    }


    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();
    }


    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMAPE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }



    public SDVariable mmul(SDVariable x,
                           SDVariable y,
                           MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new Mmul(sameDiff(),x,y,mMulTranspose).outputVariables()[0];
    }


    public SDVariable mmul(SDVariable x,
                           SDVariable y) {
        return mmul(x,y,MMulTranspose.allFalse());
    }


    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new TensorMmul(sameDiff(),x,y,dimensions).outputVariables()[0];
    }



    public SDVariable softmaxDerivative(SDVariable functionInput, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new SoftMaxDerivative(sameDiff(),functionInput,wrt).outputVariables()[0];
    }


    public SDVariable logSoftmax(SDVariable i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return new LogSoftMax(sameDiff(),i_v,null).outputVariables()[0];

    }


    public SDVariable logSoftmaxDerivative(SDVariable arg, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(arg);
        return new LogSoftMaxDerivative(sameDiff(),arg,wrt).outputVariables()[0];
    }


    public SDVariable selu(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELU(sameDiff(),arg,null).outputVariables()[0];
    }


    public SDVariable seluDerivative(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELUDerivative(sameDiff(),arg,null).outputVariables()[0];
    }


    public SDVariable rsub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(),differentialFunction,i_v).outputVariables()[0];
    }


    public SDVariable rdiv(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},false).outputVariables()[0];

    }


    public SDVariable rdivi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},true).outputVariables()[0];

    }


    public SDVariable rsubi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(),differentialFunction,i_v).outputVariables()[0];

    }


    public SDVariable add(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},false).outputVariables()[0];

    }


    public SDVariable addi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},true).outputVariables()[0];

    }


    public SDVariable sub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},false).outputVariables()[0];

    }


    public SDVariable subi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},true).outputVariables()[0];

    }


    public SDVariable mul(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},false).outputVariables()[0];

    }


    public SDVariable muli(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},true).outputVariables()[0];

    }


    public SDVariable div(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},false).outputVariables()[0];
    }


    public SDVariable divi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(),new SDVariable[]{differentialFunction,i_v},true).outputVariables()[0];
    }


    public SDVariable rsub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(),differentialFunction,i_v).outputVariables()[0];

    }


    public SDVariable rdiv(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(),differentialFunction,i_v).outputVariables()[0];

    }


    public SDVariable rdivi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];
    }


    public SDVariable rsubi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];

    }


    public SDVariable add(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(),differentialFunction,i_v,false).outputVariables()[0];
    }


    public SDVariable addi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];
    }


    public SDVariable sub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(),differentialFunction,i_v).outputVariables()[0];
    }


    public SDVariable subi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];

    }


    public SDVariable mul(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(),differentialFunction,i_v).outputVariables()[0];

    }


    public SDVariable muli(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];

    }


    public SDVariable div(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(),differentialFunction,i_v).outputVariables()[0];
    }


    public SDVariable divi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(),differentialFunction,i_v,true).outputVariables()[0];
    }


    public SDVariable gt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(),new SDVariable[]{functionInput,functionInput1},false).outputVariables()[0];
    }


    public SDVariable lt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(),new SDVariable[]{functionInput,functionInput1},false).outputVariables()[0];
    }


    public SDVariable gti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(),new SDVariable[]{functionInput,functionInput1},true).outputVariables()[0];
    }


    public SDVariable lti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(),new SDVariable[]{functionInput,functionInput1},true).outputVariables()[0];
    }


    public SDVariable gte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(),new SDVariable[]{functionInput,functionInput1},false).outputVariables()[0];
    }


    public SDVariable lte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(),new SDVariable[]{functionInput,functionInput1},false).outputVariables()[0];
    }


    public SDVariable gtei(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(),new SDVariable[]{functionInput,functionInput1},true).outputVariables()[0];
    }


    public SDVariable ltOrEqi(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(),new SDVariable[]{functionInput,functionInput1},true).outputVariables()[0];
    }




    public SDVariable gt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(),functionInput,functionInput1,false).outputVariables()[0];
    }


    public SDVariable lt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(),functionInput,functionInput1,false).outputVariables()[0];
    }


    public SDVariable gti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(),functionInput,functionInput1,true).outputVariables()[0];
    }


    public SDVariable lti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(),functionInput,functionInput1,true).outputVariables()[0];
    }


    public SDVariable gte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(),functionInput,functionInput1,false).outputVariables()[0];
    }


    public SDVariable lte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(),functionInput,functionInput1,false).outputVariables()[0];
    }


    public SDVariable gtei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(),functionInput,functionInput1,true).outputVariables()[0];
    }


    public SDVariable ltei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(),functionInput,functionInput1,true).outputVariables()[0];
    }


    public SDVariable eq(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(),iX,i_y).outputVariables()[0];
    }

    public SDVariable eqi(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(),iX,i_y,true).outputVariables()[0];
    }


    /**
     *
     * @param func
     * @return
     */
    public int getInputLength(SDVariable func) {
        validateDifferentialFunctionsameDiff(func);
        int[] inputShape = func.arg().getShape();
        return ArrayUtil.prod(inputShape);
    }

    public int getReductionLength(DifferentialFunction func){
        int[] inputShape = func.arg().getShape();
        if(Shape.isWholeArray(inputShape, func.getDimensions())){
            return ArrayUtil.prod(inputShape);
        }
        int prod = 1;
        for(int i : func.getDimensions()){
            prod *= inputShape[i];
        }
        return prod;
    }






    public void validateDifferentialFunctionsameDiff(
            SDVariable function) {

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



    public void validateDifferentialFunctionGraph(SDVariable function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }






    /**
     *
     * @param func
     * @param input
     * @return
     */
    public SDVariable doGradChoose(SDVariable func,
                                   SDVariable input) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);

        SDVariable repeatedGrad = doRepeat(func,input);
        SDVariable resultRepeated = doRepeat(func.args()[0],input);
        SDVariable argMaxLocations = eq(input,resultRepeated);
        return div(mul(argMaxLocations,repeatedGrad),sum(argMaxLocations).outputVariables()[0]);
    }


    /**
     *
     * @param func
     * @param input
     * @return
     */
    public  SDVariable doRepeat(SDVariable func,
                                SDVariable input) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);
        return tile(func,input.getShape());
    }


    public String toString() {
        return "DifferentialFunctionFactory{" +
                "methodNames=" + methodNames +
                '}';
    }





}
