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
 * @param <X>
 */
@Data
public class DifferentialFunctionFactory<X extends Field<ArrayField>> implements FunctionFactory<ArrayField>  {

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
    public DifferentialFunction<ArrayField> invoke(String name, Object[] args) {
        try {
            return (DifferentialFunction<ArrayField> ) methodNames.get(name).invoke(this,args);
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
    public One<ArrayField>  one(int[] shape) {
        return sameDiff.setupFunction(new One<>(sameDiff,shape));
    }

    @Override
    public DifferentialFunction<ArrayField> tile(DifferentialFunction<ArrayField> iX,
                                                 int[] repeat) {
        return sameDiff.setupFunction(new Tile(sameDiff,iX,repeat));
    }


    @Override
    public DifferentialFunction<ArrayField> valueArrayOf(DifferentialFunction<ArrayField> iX, int[] shape) {
        return sameDiff.setupFunction(new ValueArrayOf(sameDiff,iX,shape,null));
    }

    @Override
    public DifferentialFunction<ArrayField> sum(DifferentialFunction<ArrayField> i_x,
                                                int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Sum(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> prod(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new Prod(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> mean(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Mean(sameDiff,i_x,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> std(DifferentialFunction<ArrayField> i_x,
                                                boolean biasCorrected,
                                                int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.StandardDeviation(sameDiff,i_x,dimensions,biasCorrected));
    }

    @Override
    public DifferentialFunction<ArrayField> variance(DifferentialFunction<ArrayField> i_x,
                                                     boolean biasCorrected,
                                                     int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Variance(sameDiff,i_x,dimensions,biasCorrected));

    }

    @Override
    public DifferentialFunction<ArrayField> max(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Max(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction<ArrayField> min(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Min(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction<ArrayField> norm1(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Norm1(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction<ArrayField> norm2(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.reduce.Norm2(sameDiff,i_x,dimensions));

    }

    @Override
    public DifferentialFunction<ArrayField> normmax(DifferentialFunction<ArrayField> i_x, int... dimensions) {
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
    public DifferentialFunction<ArrayField> doNormGrad(DifferentialFunction<ArrayField> func,
                                                       DifferentialFunction<ArrayField> input,
                                                       String type,
                                                       int... axes) {

        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);
        DifferentialFunction<ArrayField> result;
        if(Shape.isWholeArray(axes)) {
            result = input;
        }
        else if(axes.length > 1) {
            if(axes[0] > axes[1]) {
                axes[0]--;
            }

            result = expandDims(expandDims(func.div(input).mul(func.args()[0]),axes[0]),axes[1]);
        }
        else {
            result = expandDims(func.div(input).mul(func.args()[0]),axes[0]);
        }

        return result;
    }



    @Override
    public DifferentialFunction<ArrayField> expandDims(DifferentialFunction<ArrayField> iX,int axis) {
        return sameDiff.setupFunction(new ExpandDims(sameDiff,iX,null,axis));
    }



    @Override
    public DifferentialFunction<ArrayField> abs(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Abs(sameDiff,iX,null));
    }


    @Override
    public DifferentialFunction<ArrayField> neg(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Neg(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> cos(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Cos(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> sin(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sin(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> tan(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Tan(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> permute(DifferentialFunction<ArrayField> iX, int... dimensions) {
        return sameDiff.setupFunction(new Permute(sameDiff,iX,dimensions));
    }


    @Override
    public DifferentialFunction<ArrayField> transpose(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new Transpose(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> acos(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.ACos(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> asin(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new ASin(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> atan(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new ATan(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> cosh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Cosh(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> sinh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new Sinh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> tanh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new Tanh(sameDiff,iX,null));
    }


    @Override
    public DifferentialFunction<ArrayField> tanhDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> wrt) {
        return sameDiff.setupFunction(new TanhDerivative(sameDiff,iX,wrt));
    }

    @Override
    public DifferentialFunction<ArrayField> acosh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new ACosh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> asinh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new  org.nd4j.autodiff.functions.impl.unary.transform.ASinh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> atanh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new ATanh(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> exp(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new Exp(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> log(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Log(sameDiff,iX,null));
    }



    @Override
    public DifferentialFunction<ArrayField> or(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Or(sameDiff,iX,i_y));
    }


    @Override
    public DifferentialFunction<ArrayField> eq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Eq(sameDiff,iX,i_y));
    }

    @Override
    public DifferentialFunction<ArrayField> neq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.transform.Neq(sameDiff,iX,i_y));

    }

    @Override
    public DifferentialFunction<ArrayField> pow(DifferentialFunction<ArrayField> iX, double i_y) {
        return sameDiff.setupFunction(new ScalarPow(sameDiff,iX,new Object[]{i_y}));
    }

    @Override
    public DifferentialFunction<ArrayField> sqrt(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sqrt(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> square(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Square(sameDiff,iX,null));
    }

    @Override
    public DifferentialFunction<ArrayField> floor(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Floor(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> relu(DifferentialFunction<ArrayField> iX, double cutoff) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Floor(sameDiff,iX,new Object[]{cutoff}));

    }



    @Override
    public DifferentialFunction<ArrayField> softmax(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Softmax(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> hardTanh(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.HardTanh(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction<ArrayField> hardTanhDerivative(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new HardTanhDerivative(sameDiff,iX,null));

    }




    @Override
    public DifferentialFunction<ArrayField> sigmoid(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sigmoid(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction<ArrayField> sigmoidDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> wrt) {
        return sameDiff.setupFunction(new SigmoidDerivative(sameDiff,sameDiff.setupFunction(iX),sameDiff.setupFunction(wrt)));

    }


    @Override
    public DifferentialFunction<ArrayField> sign(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Sign(sameDiff,iX,null));

    }


    @Override
    public DifferentialFunction<ArrayField> broadcast(DifferentialFunction<ArrayField> iX, int... shape) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.shape.Broadcast(sameDiff,iX,shape));
    }

    @Override
    public DifferentialFunction<ArrayField> repeat(DifferentialFunction<ArrayField> iX, int axis) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.shape.Repeat(sameDiff,iX,axis));

    }

    @Override
    public DifferentialFunction<ArrayField> softsign(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.SoftSign(sameDiff,iX,null));

    }

    @Override
    public DifferentialFunction<ArrayField> softsignDerivative(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new SoftSignDerivative(sameDiff,iX,null));

    }





    @Override
    public DifferentialFunction<ArrayField> softplus(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.SoftPlus(sameDiff,iX,null));

    }


    @Override
    public DifferentialFunction<ArrayField> elu(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.Elu(sameDiff,iX,null));

    }



    @Override
    public DifferentialFunction<ArrayField> eluDerivative(DifferentialFunction<ArrayField> iX) {
        return sameDiff.setupFunction(new EluDerivative(sameDiff,iX,null));

    }




    @Override
    public DifferentialFunction<ArrayField> leakyRelu(DifferentialFunction<ArrayField> iX, double cutoff) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.unary.transform.LeakyRelu(sameDiff,iX,cutoff));

    }



    @Override
    public DifferentialFunction<ArrayField> leakyReluDerivative(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> iY, double cutoff) {
        return sameDiff.setupFunction(new LeakyReluDerivative(sameDiff,iX,iY,cutoff));

    }

    @Override
    public DifferentialFunction<ArrayField> reshape(DifferentialFunction<ArrayField> iX, int[] shape) {
        return sameDiff.setupFunction(new Reshape(sameDiff,iX,shape));
    }

    @Override
    public DifferentialFunction<ArrayField> rollAxis(Variable  iX, int axis) {
        return sameDiff.setupFunction(new RollAxis(sameDiff,iX,axis));
    }

    @Override
    public DifferentialFunction<ArrayField> cosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return sameDiff.setupFunction(new org.nd4j.autodiff.functions.impl.binary.reduce.CosineSimilarity(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> euclideanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return sameDiff.setupFunction(new EuclideanDistance(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> manhattanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return sameDiff.setupFunction(new ManhattanDistance(sameDiff,iX,i_y,dimensions));
    }

    @Override
    public DifferentialFunction<ArrayField> lossBinaryXENT(DifferentialFunction<ArrayField> iX,
                                                           DifferentialFunction<ArrayField> i_y,
                                                           int... dimensions) {
        return sameDiff.setupFunction(new AbstractBinaryReduceFunction(sameDiff,iX,i_y,dimensions) {
            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossBinaryXENT(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable> variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossBinaryXENT";
            }


            @Override
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                DifferentialFunction<ArrayField> numerator = i_y.sub(iX);
                DifferentialFunction<ArrayField> denominator = i_y.mul(i_y.rsub(1.0));
                DifferentialFunction<ArrayField> dLda = denominator.div(denominator);

                /**
                 *   INDArray output = activationFn.getActivation(preOutput.dup(), true));

                 INDArray numerator = output.sub(labels));
                 INDArray denominator = output.mul(output.rsubi(1))); // output * (1-output)
                 INDArray dLda = numerator.divi(denominator));

                 if(mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
                 //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
                 //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
                 //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
                 // error prone - but buy us a tiny bit of performance
                 LossUtil.applyMask(dLda, mask));
                 }

                 INDArray grad = activationFn.backprop(preOutput, dLda).getFirst()); //TODO activation functions with weights

                 //Weighted loss function
                 if (weights != null) {
                 if (weights.length() != output.size(1)) {
                 throw new IllegalStateException("Weights vector (length " + weights.length()
                 + ") does not match output.size(1)=" + output.size(1)));
                 }

                 grad.muliRowVector(weights));
                 }

                 if (mask != null) {
                 LossUtil.applyMask(grad, mask));
                 }


                 */
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossCosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossKLD(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossL1(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossL2(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAPE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossMCXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSLE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossNegativeLogLikelihood(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossPoisson(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }

    @Override
    public DifferentialFunction<ArrayField> lossSquaredHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
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
            public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
                return null;
            }
        });
    }


    @Override
    public DifferentialFunction<ArrayField> mmul(DifferentialFunction<ArrayField> x,
                                                 DifferentialFunction<ArrayField> y,
                                                 MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return sameDiff.setupFunction(new Mmul(sameDiff,x,y,mMulTranspose));
    }

    @Override
    public DifferentialFunction<ArrayField> mmul(DifferentialFunction<ArrayField> x,
                                                 DifferentialFunction<ArrayField> y) {
        return mmul(x,y,MMulTranspose.allFalse());
    }

    @Override
    public DifferentialFunction<ArrayField> tensorMmul(DifferentialFunction<ArrayField> x,
                                                       DifferentialFunction<ArrayField> y,
                                                       int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return sameDiff.setupFunction(new TensorMmul<>(sameDiff,x,y,dimensions));
    }


    @Override
    public DifferentialFunction<ArrayField> softmaxDerivative(DifferentialFunction<ArrayField> functionInput, DifferentialFunction<ArrayField> wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return sameDiff.setupFunction(new SoftMaxDerivative(sameDiff,functionInput,wrt));
    }

    @Override
    public DifferentialFunction<ArrayField> logSoftmax(DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return sameDiff.setupFunction(new LogSoftMax(sameDiff,i_v,null));

    }

    @Override
    public DifferentialFunction<ArrayField> selu(DifferentialFunction<ArrayField> arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff.setupFunction(new SELU(sameDiff,arg,null));
    }

    @Override
    public DifferentialFunction<ArrayField> seluDerivative(DifferentialFunction<ArrayField> arg) {
        validateDifferentialFunctionsameDiff(arg);
        return sameDiff.setupFunction(new SELUDerivative(sameDiff,arg,null));
    }

    @Override
    public DifferentialFunction<ArrayField> rsub(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RSub(sameDiff,differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction<ArrayField> rdiv(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RDiv(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> rdivi(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RDiv(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> rsubi(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new RSub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Add(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> addi(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Add(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> sub(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Sub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> subi(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Sub(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Mul(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> muli(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Mul(sameDiff,differentialFunction,i_v));

    }

    @Override
    public DifferentialFunction<ArrayField> div(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Div(sameDiff,sameDiff.setupFunction(differentialFunction),sameDiff.setupFunction(i_v)));
    }

    @Override
    public DifferentialFunction<ArrayField> divi(DifferentialFunction<ArrayField> differentialFunction, DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new Div(sameDiff,differentialFunction,i_v));
    }

    @Override
    public DifferentialFunction<ArrayField> rsub(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> rdiv(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRDiv(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> rdivi(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction<ArrayField> rsubi(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarRSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarAdd(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction<ArrayField> addi(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarAdd(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction<ArrayField> sub(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarSub(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction<ArrayField> subi(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarSub(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarMul(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> muli(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarMul(sameDiff,differentialFunction,new Object[]{i_v}));

    }

    @Override
    public DifferentialFunction<ArrayField> div(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    @Override
    public DifferentialFunction<ArrayField> divi(DifferentialFunction<ArrayField> differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return sameDiff.setupFunction(new ScalarDiv(sameDiff,differentialFunction,new Object[]{i_v}));
    }

    /**
     *
     * @param func
     * @return
     */
    public int getInputLength(DifferentialFunction<ArrayField> func) {
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
    public DifferentialFunction<ArrayField> doGradChoose(DifferentialFunction<ArrayField> func,
                                                         DifferentialFunction<ArrayField> input,
                                                         int...axes) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);

        DifferentialFunction<ArrayField> repeatedGrad = doRepeat(func,input,axes);
        DifferentialFunction<ArrayField> resultRepeated = doRepeat(func.args()[0],input,axes);
        DifferentialFunction<ArrayField> argMaxLocations = eq(input,resultRepeated);
        return argMaxLocations.mul(repeatedGrad).div(sum(argMaxLocations,axes));


    }


    /**
     *
     * @param func
     * @param input
     * @param axes
     * @return
     */
    public  DifferentialFunction<ArrayField> doRepeat(DifferentialFunction<ArrayField> func,
                                                      DifferentialFunction<ArrayField> input,
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
            DifferentialFunction<ArrayField> function) {
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
