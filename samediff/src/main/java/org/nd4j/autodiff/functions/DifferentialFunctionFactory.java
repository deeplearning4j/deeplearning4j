package org.nd4j.autodiff.functions;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.impl.binary.reduce.EuclideanDistance;
import org.nd4j.autodiff.functions.impl.binary.reduce.ManhattanDistance;
import org.nd4j.autodiff.functions.impl.binary.transform.*;
import org.nd4j.autodiff.functions.impl.binary.transform.gradient.*;
import org.nd4j.autodiff.functions.impl.binary.transform.scalar.*;
import org.nd4j.autodiff.functions.impl.unary.reduce.Prod;
import org.nd4j.autodiff.functions.impl.unary.transform.*;
import org.nd4j.autodiff.functions.impl.unary.transform.shape.*;
import org.nd4j.autodiff.functions.mmul.Mmul;
import org.nd4j.autodiff.functions.mmul.TensorMmul;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

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


    @Override
    public DifferentialFunction invoke(String name, Object[] args) {
        try {
            return (DifferentialFunction ) methodNames.get(name).invoke(this,args);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Constant  val(ArrayField iX) {
        return sameDiff.setupFunction(new Constant(sameDiff, iX,
                iX.getInput().getShape()));
    }



    @Override
    public Variable  var(String iName, ArrayField iX, PreEvaluator  preEvaluator) {
        Preconditions.checkArgument(iX.getOps() == sameDiff,"Same diff must be the same.");
        return sameDiff.setupFunction(new Variable(sameDiff,iName, iX, preEvaluator));
    }

    @Override
    public Variable  var(String iName, ArrayField iX) {
        Preconditions.checkArgument(iX.getOps() == sameDiff,"Same diff must be the same.");
        return sameDiff.setupFunction(new Variable(sameDiff,iName, iX));
    }

    @Override
    public Zero  zero(int[] shape) {
        return sameDiff.setupFunction(new Zero(sameDiff,shape));
    }

    @Override
    public One  one(int[] shape) {
        return sameDiff.setupFunction(new One(sameDiff,shape));
    }

    @Override
    public DifferentialFunction tile(DifferentialFunction iX, int[] repeat) {
        return sameDiff.setupFunction(new Tile(sameDiff,iX,repeat));
    }


    @Override
    public DifferentialFunction valueArrayOf(DifferentialFunction iX, int[] shape) {
        return sameDiff.setupFunction(new ValueArrayOf(sameDiff,iX,shape,null));
    }

    @Override
    public DifferentialFunction sum(DifferentialFunction i_x,
                                    int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Sum(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction prod(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new Prod(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction mean(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Mean(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction std(DifferentialFunction i_x,
                                    boolean biasCorrected,
                                    int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.StandardDeviation(sameDiff,i_x,dimensions,biasCorrected));
    }

    @Override
    public DifferentialFunction variance(DifferentialFunction i_x,
                                         boolean biasCorrected,
                                         int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Variance(sameDiff,i_x,dimensions,biasCorrected));

    }

    @Override
    public DifferentialFunction max(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Max(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction min(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Min(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction norm1(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Norm1(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction norm2(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Norm2(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction normmax(DifferentialFunction i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.NormMax(sameDiff,i_x,dimensions));

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
    public DifferentialFunction expandDims(DifferentialFunction iX,int axis) {
        return sameDiff.setupFunction(new ExpandDims(sameDiff,iX,null,axis));
    }



    @Override
    public DifferentialFunction abs(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Abs(sameDiff,iX,null));
    }


    @Override
    public DifferentialFunction neg(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Neg(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction cos(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Cos(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction sin(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sin(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction tan(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Tan(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction permute(DifferentialFunction iX, int... dimensions) {
        return sameDiff.setupFunction(new Permute(sameDiff,iX,dimensions));
    }


    @Override
    public DifferentialFunction transpose(DifferentialFunction iX) {
        return sameDiff.setupFunction(new Transpose(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction acos(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.ACos(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction asin(DifferentialFunction iX) {
        return sameDiff.setupFunction(new ASin(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction atan(DifferentialFunction iX) {
        return sameDiff.setupFunction(new ATan(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction cosh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Cosh(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction sinh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new Sinh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction tanh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new Tanh(sameDiff,iX,null));
    }


    @Override
    public DifferentialFunction tanhDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff.setupFunction(new TanhDerivative(sameDiff,iX,wrt));
    }

    @Override
    public DifferentialFunction acosh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new ACosh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction asinh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new  org.nd4j.autodiff.functions.impl.unary.transform.ASinh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction atanh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new ATanh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction exp(DifferentialFunction iX) {
        return sameDiff.setupFunction(new Exp(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction log(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Log(sameDiff,iX,null));
    }



    @Override
    public DifferentialFunction or(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Or(sameDiff,iX,i_y));
    }


    @Override
    public DifferentialFunction eq(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Eq(sameDiff,iX,i_y));
    }

    @Override
    public DifferentialFunction neq(DifferentialFunction iX, DifferentialFunction i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Neq(sameDiff,iX,i_y));

    }

    @Override
    public DifferentialFunction pow(DifferentialFunction iX, double i_y) {
        return sameDiff.setupFunction(new ScalarPow(sameDiff,iX,new Object[]{i_y}));
    }

    @Override
    public DifferentialFunction sqrt(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sqrt(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction square(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Square(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction floor(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Floor(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction relu(DifferentialFunction iX, double cutoff) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Floor(sameDiff,iX,new Object[]{cutoff}));

    }



    @Override
    public DifferentialFunction softmax(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Softmax(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction hardTanh(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.HardTanh(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction hardTanhDerivative(DifferentialFunction iX) {
        return sameDiff.setupFunction(new HardTanhDerivative(sameDiff,iX,null));

    }




    @Override
    public DifferentialFunction sigmoid(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sigmoid(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction sigmoidDerivative(DifferentialFunction iX, DifferentialFunction wrt) {
        return sameDiff.setupFunction(new SigmoidDerivative(sameDiff,sameDiff.setupFunction(iX),sameDiff.setupFunction(wrt)));

    }


    @Override
    public DifferentialFunction sign(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sign(sameDiff,iX,null));

    }


    @Override
    public DifferentialFunction broadcast(DifferentialFunction iX, int... shape) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.shape.Broadcast(sameDiff,iX,shape));
    }

    @Override
    public DifferentialFunction repeat(DifferentialFunction iX, int axis) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.shape.Repeat(sameDiff,iX,axis));

    }

    @Override
    public DifferentialFunction softsign(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.SoftSign(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction softsignDerivative(DifferentialFunction iX) {
        return sameDiff.setupFunction(new SoftSignDerivative(sameDiff,iX,null));

    }





    @Override
    public DifferentialFunction softplus(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.SoftPlus(sameDiff,iX,null));

    }


    @Override
    public DifferentialFunction elu(DifferentialFunction iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Elu(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction eluDerivative(DifferentialFunction iX) {
        return sameDiff.setupFunction(new EluDerivative(sameDiff,iX,null));

    }




    @Override
    public DifferentialFunction leakyRelu(DifferentialFunction iX, double cutoff) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.LeakyRelu(sameDiff,iX,cutoff));

    }



    @Override
    public DifferentialFunction leakyReluDerivative(DifferentialFunction iX, DifferentialFunction iY, double cutoff) {
        return sameDiff.setupFunction(new LeakyReluDerivative(sameDiff,iX,iY,cutoff));

    }

    @Override
    public DifferentialFunction reshape(DifferentialFunction iX, int[] shape) {
        return sameDiff.setupFunction(new Reshape(sameDiff,iX,shape));
    }

    @Override
    public DifferentialFunction rollAxis(Variable  iX, int axis) {
        return sameDiff.setupFunction(new RollAxis(sameDiff,iX,axis));
    }

    @Override
    public DifferentialFunction cosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.reduce.CosineSimilarity(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction euclideanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new EuclideanDistance(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction manhattanDistance(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new ManhattanDistance(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction lossBinaryXENT(DifferentialFunction iX,
                                               DifferentialFunction i_y,
                                               int... dimensions) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction lossCosineSimilarity(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossCosineSimilarity(iX,i_y,dimensions);
            }

            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossCosineSimilarity";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossHinge(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossHinge";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossKLD(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossKLD(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossKLD";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossL1(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossL1(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossL1";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossL2(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossL2(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossL2";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossMAE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMAE(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMAE";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossMAPE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMAPE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMAPE";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossMSE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMSE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMSE";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossMCXENT(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMCXENT(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMCXENT";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossMSLE(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMSLE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMSLE";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossNegativeLogLikelihood(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossNegativeLogLikelihood(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossNegativeLogLikelihood";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossPoisson(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossPoisson(iX,i_y,dimensions);
            }

            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossPoisson";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction lossSquaredHinge(DifferentialFunction iX, DifferentialFunction i_y, int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossSquaredHinge(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossSquaredHinge";
            }


            @Override
            public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
                return null;
            }
        });
    }


    @Override
    public DifferentialFunction mmul(DifferentialFunction x,
                                     DifferentialFunction y,
                                     MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return sameDiff.setupFunction(new Mmul(sameDiff,x,y,mMulTranspose));
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
        return sameDiff.setupFunction(new TensorMmul(sameDiff,x,y,dimensions));
    }


    @Override
    public DifferentialFunction softmaxDerivative(DifferentialFunction functionInput, DifferentialFunction wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff.setupFunction(new SoftMaxDerivative(sameDiff,functionInput,wrt));
    }

    @Override
    public DifferentialFunction logSoftmax(DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return sameDiff.setupFunction(new LogSoftMax(sameDiff,i_v,null));

    }

    @Override
    public DifferentialFunction selu(DifferentialFunction arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff.setupFunction(new SELU(sameDiff,arg,null));
    }

    @Override
    public DifferentialFunction seluDerivative(DifferentialFunction arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff.setupFunction(new SELUDerivative(sameDiff,arg,null));
    }

    @Override
    public DifferentialFunction rsub(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RSub(sameDiff,differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction rdiv(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RDiv(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rdivi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RDiv(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction rsubi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RSub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction add(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Add(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction addi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Add(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction sub(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Sub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction subi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Sub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction mul(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Mul(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction muli(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Mul(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction div(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Div(sameDiff,sameDiff.setupFunction(differentialFunction),sameDiff.setupFunction(i_v)));
    }

    @Override
    public DifferentialFunction divi(DifferentialFunction differentialFunction, DifferentialFunction i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Div(sameDiff,differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction rsub(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction rdiv(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRDiv(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction rdivi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction rsubi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction add(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarAdd(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction addi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarAdd(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction sub(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarSub(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction subi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction mul(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarMul(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction muli(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarMul(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction div(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction divi(DifferentialFunction differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    /**
     *
     * @param func
     * @return
     */
    public int getInputLength(DifferentialFunction func) {
        validateDifferentialFunctionsameDiff(func);
        ArrayField arrayField = func.getValue(true);
        int[] inputShape = arrayField.getInput().getShape();
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
                        "in same sameDiff. The left " + function +"" +
                        " must match this function " + this);
        Preconditions.checkState(sameDiff ==
                this.getSameDiff(),"Function applications m" +
                "ust be " +
                "contained in same sameDiff. The left " + function +" " +
                "must " +
                "match this function " + this);

    }



}
