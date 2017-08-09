package org.nd4j.autodiff.functions;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.mmul.Mmul;
import org.nd4j.autodiff.functions.mmul.TensorMmul;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.Negative;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

/**
 *
 * @param <X>
 */
@Data
public class DifferentialFunctionFactory<X extends Field<ArrayField> > implements FunctionFactory<ArrayField>  {

    protected SameDiff sameDiff;
    private Map<String,Method> methodNames;

    public DifferentialFunctionFactory(SameDiff sameDiff) {
        if (sameDiff != null) {
            this.sameDiff = sameDiff;
            methodNames = new HashMap<>();
            Method[] methods = getClass().getDeclaredMethods();
            for(Method method : methods)
                methodNames.put(method.getName(),method);
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
    public Constant<ArrayField>  val(ArrayField iX) {
        if(iX instanceof ArrayField) {

            return new Constant<>(sameDiff, iX,
                    ((ArrayField) iX).getInput().getShape());
        }
        else
            throw new IllegalStateException("Illegal type. Must be ArrayField");
    }


    @Override
    public Variable<ArrayField>  var(String iName, ArrayField iX, PreEvaluator<ArrayField>  preEvaluator) {
        return new Variable<>(sameDiff,iName, iX, preEvaluator);
    }

    @Override
    public Variable<ArrayField>  var(String iName, ArrayField iX) {
        return new Variable<>(sameDiff,iName, iX);
    }

    @Override
    public Zero<ArrayField>  zero(int[] shape) {
        return new Zero<>(sameDiff,shape);
    }

    @Override
    public One<ArrayField>  one(int[] shape) {
        return new One<>(sameDiff,shape);
    }

    @Override
    public DifferentialFunction<ArrayField> tile(DifferentialFunction<ArrayField> iX,
                                        int[] repeat) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,
                new Object[]{repeat}) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().tile(arg().getValue(true),repeat);
            }

            @Override
            public double getReal() {
                throw new UnsupportedOperationException();
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                validateDifferentialFunctionsameDiff(i_v);
                throw new UnsupportedOperationException();
            }


            @Override
            public String functionName() {
                return "tile";
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> valueArrayOf(DifferentialFunction<ArrayField> iX, int[] shape) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,new Object[]{shape}) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().valueArrayOf((ArrayField) arg().getValue(true),shape);
            }

            @Override
            public double getReal() {
                throw new UnsupportedOperationException();
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                throw new UnsupportedOperationException();
            }


            @Override
            public String functionName() {
                return "valueArray";
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> sum(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sum(arg().doGetValue(),dimensions);
            }

            @Override
            public String functionName() {
                return new org.nd4j.linalg.api.ops.impl.accum.Sum().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doRepeat(this,i_v1,dimensions).mul(arg().diff(i_v1));
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> prod(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().prod((ArrayField) arg().doGetValue(),dimensions);
            }


            @Override
            public String functionName() {
                return new org.nd4j.linalg.api.ops.impl.accum.Prod().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doRepeat(this,i_v1,dimensions).div(one(getResultShape()).mul(getInputLength(i_v1)));
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> mean(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().mean((ArrayField) arg().doGetValue(),dimensions);
            }


            @Override
            public String functionName() {
                return new Mean().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doRepeat(this,i_v1,dimensions).div(one(i_v1.getResultShape()).mul(getInputLength(i_v1)));
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> std(DifferentialFunction<ArrayField> i_x,
                                       boolean biasCorrected,
                                       int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().std(arg().doGetValue(),
                        biasCorrected ,
                        dimensions);
            }

            @Override
            public String functionName() {
                return new StandardDeviation().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                int inputs = getInputLength(i_v1);
                DifferentialFunction<ArrayField> g =  doRepeat(this,i_v1,dimensions);
                return g.mul(arg().sub(DifferentialFunctionFactory.this.mean(arg(),dimensions))).div(one(g.getResultShape()).mul(inputs));
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> variance(DifferentialFunction<ArrayField> i_x,
                                            boolean biasCorrected,
                                            int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().variance(arg().doGetValue(),
                        biasCorrected, dimensions);
            }


            @Override
            public String functionName() {
                return new Variance().name();
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                int inputs = getInputLength(i_v1);
                DifferentialFunction<ArrayField> g =  doRepeat(this,i_v1,dimensions);
                return one(getResultShape()).mul(2).mul(g).mul(arg().sub(DifferentialFunctionFactory.this.mean(arg(),dimensions))).div(one(getResultShape()).mul(inputs));
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> max(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().max((ArrayField) arg().doGetValue(),dimensions);
            }


            @Override
            public String functionName() {
                return new Max().name();
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doGradChoose(this,i_v1,dimensions);
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> min(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().min((ArrayField) arg().doGetValue(),dimensions);
            }

            @Override
            public String functionName() {
                return new Min().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doGradChoose(this,i_v1,dimensions);
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> norm1(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().norm1((ArrayField) arg().doGetValue(),dimensions);
            }


            @Override
            public String functionName() {
                return new Norm1().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return doNormGrad(this,i_v1,"norm1",dimensions);
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> norm2(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().norm2((ArrayField) arg().doGetValue(),dimensions);
            }


            @Override
            public String functionName() {
                return new Norm2().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doNormGrad(this,i_v1,"norm2",dimensions);
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> normmax(DifferentialFunction<ArrayField> i_x, int... dimensions) {
        return new AbstractReduceUnaryFunction<ArrayField> (sameDiff,i_x,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().normmax((ArrayField) arg().doGetValue(),dimensions);
            }

            @Override
            public String functionName() {
                return new NormMax().name();
            }



            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                validateDifferentialFunctionsameDiff(i_v1);
                return doNormGrad(this,i_v1,"max",dimensions);
            }
        };
    }

    private DifferentialFunction<ArrayField> doNormGrad(DifferentialFunction<ArrayField> func,
                                               DifferentialFunction<ArrayField> input,
                                               String type,
                                               int...axes) {

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
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().expandDims(arg().getValue(true),axis);
            }

            @Override
            public double getReal() {
                return Math.abs(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                validateDifferentialFunctionsameDiff(i_v);
                return arg().div(DifferentialFunctionFactory.this.abs(arg()));
            }


            @Override
            public String functionName() {
                return "expandDims";
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> abs(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().abs((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.abs(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return arg().div(DifferentialFunctionFactory.this.abs(arg()));
            }


            @Override
            public String functionName() {
                return new Abs().name();
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> neg(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().neg((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return -arg().getReal();
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                validateDifferentialFunctionsameDiff(i_v);
                return i_v.negate().mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new Negative().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> cos(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().cos((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.cos(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                validateDifferentialFunctionsameDiff(i_v);
                return (DifferentialFunctionFactory.this.sin(arg()).mul(arg().diff(i_v))).negate();
            }


            @Override
            public String functionName() {
                return new Cos().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> sin(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sin((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.sin(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                validateDifferentialFunctionsameDiff(i_v);
                return DifferentialFunctionFactory.this.cos(arg()).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new Sin().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> tan(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().tan((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.tan(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return (new PolynomialTerm<>(sameDiff,1, DifferentialFunctionFactory.this.cos(arg()), -2)).mul(arg().diff(i_v));
            }



            @Override
            public String functionName() {
                return new Tan().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> permute(DifferentialFunction<ArrayField> iX, int... dimensions) {
        if(iX.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) iX.getValue(true);
            return new AbstractUnaryFunction<ArrayField> (sameDiff,
                    iX,
                    ArrayUtil.reverseCopy(arrayField.getInput().getShape()),
                    OpState.OpType.SHAPE,
                    null) {

                @Override
                public ArrayField doGetValue() {
                    return sameDiff.getArrayFactory().permute((ArrayField) arg().getValue(true),dimensions);
                }

                @Override
                public double getReal() {
                    return Math.tan(arg().getReal());
                }

                @Override
                public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                    return this;
                }



                @Override
                public String functionName() {
                    return "permute";
                }
            };
        }

        throw new IllegalStateException("Need the shape. This is only possible with ArrayField");
    }


    @Override
    public DifferentialFunction<ArrayField> transpose(DifferentialFunction<ArrayField> iX) {
        if(iX.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) iX.getValue(true);
            return new AbstractUnaryFunction<ArrayField> (sameDiff,
                    iX,ArrayUtil.reverseCopy(arrayField.getInput().getShape()),
                    OpState.OpType.SHAPE,null) {

                @Override
                public ArrayField doGetValue() {
                    return sameDiff.getArrayFactory().transpose((ArrayField) arg().getValue(true));
                }

                @Override
                public double getReal() {
                    return Math.tan(arg().getReal());
                }

                @Override
                public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                    return this;
                }



                @Override
                public String functionName() {
                    return "transpose";
                }
            };
        }

        throw new IllegalStateException("Need the shape. This is only possible with ArrayField");
    }

    @Override
    public DifferentialFunction<ArrayField> acos(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().acos((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.acos(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).div(DifferentialFunctionFactory.this.sqrt(one(getResultShape()).sub(arg().pow(2)))).negate();
            }


            @Override
            public String functionName() {
                return new ACos().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> asin(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().asin((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.asin(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).div(DifferentialFunctionFactory.this.sqrt(one(getResultShape()).sub(arg().pow(2))));
            }



            @Override
            public String functionName() {
                return new ASin().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> atan(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().atan((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.atan(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).div(one(getResultShape()).add(arg().pow(2)));
            }

            @Override
            public String functionName() {
                return new ATan().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> cosh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().cosh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.cosh(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.sinh(arg());
            }

            @Override
            public String functionName() {
                return new Cosh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> sinh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sinh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.sinh(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.cosh(arg());
            }


            @Override
            public String functionName() {
                return new Sinh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> tanh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().tanh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.tanh(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.one(getResultShape()).div(DifferentialFunctionFactory.this.cosh(arg())).pow(2);
            }

            @Override
            public String functionName() {
                return new Tanh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> acosh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().acosh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                throw new IllegalStateException("");
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.one(getResultShape()).div(DifferentialFunctionFactory.this.sqrt(arg().sub(one(getResultShape()))).mul(DifferentialFunctionFactory.this.sqrt(arg().add(one(getResultShape())))));
            }

            @Override
            public String functionName() {
                return new ACosh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> asinh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().asinh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                throw new IllegalStateException();
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.one(getResultShape()).div(DifferentialFunctionFactory.this.sqrt(arg().pow(2).add(one(getResultShape()))));
            }


            @Override
            public String functionName() {
                return new ASinh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> atanh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().atanh((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                throw new IllegalStateException();
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).div(one(getResultShape()).sub(arg().pow(2)));
            }


            @Override
            public String functionName() {
                return new ATanh().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> exp(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().exp((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.exp(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.exp(arg()).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new Exp().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> log(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().log((ArrayField) arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.log(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return new Inverse<>(sameDiff,arg()).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new Log().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> or(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return new AbstractBinaryFunction<ArrayField> (sameDiff,iX, i_y) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().or(larg().getValue(true), rarg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.pow(larg().getReal(), rarg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                Constant<ArrayField>  ym1 = DifferentialFunctionFactory.this
                        .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
                return rarg().mul(DifferentialFunctionFactory.this.pow(larg(), ym1))
                        .mul(larg().diff(i_v));
            }

            @Override
            public String toString() {
                return "or(" + larg().toString() + ", " + rarg().toString() + ")";
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return "or(" + larg().doGetFormula(variables) + ","
                        + rarg().doGetFormula(variables) + ")";
            }

            @Override
            public String functionName() {
                return new Or().name();
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> eq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return new AbstractBinaryFunction<ArrayField> (sameDiff,iX, i_y) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().eq(larg().getValue(true), rarg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.pow(larg().getReal(), rarg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                Constant<ArrayField>  ym1 = DifferentialFunctionFactory.this
                        .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
                return rarg().mul(DifferentialFunctionFactory.this.pow(larg(), ym1))
                        .mul(larg().diff(i_v));
            }

            @Override
            public String toString() {
                return "eq(" + larg().toString() + ", " + rarg().toString() + ")";
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return "eq(" + larg().doGetFormula(variables) + ","
                        + rarg().doGetFormula(variables) + ")";
            }

            @Override
            public String functionName() {
                return new Not().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> neq(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y) {
        return new AbstractBinaryFunction<ArrayField> (sameDiff,iX, i_y) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().neq(larg().getValue(true), rarg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.pow(larg().getReal(), rarg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                Constant<ArrayField>  ym1 = DifferentialFunctionFactory.this
                        .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
                return rarg().mul(DifferentialFunctionFactory.this.pow(larg(), ym1))
                        .mul(larg().diff(i_v));
            }

            @Override
            public String toString() {
                return "neq(" + larg().toString() + ", " + rarg().toString() + ")";
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return "neq(" + larg().doGetFormula(variables) + ","
                        + rarg().doGetFormula(variables) + ")";
            }

            @Override
            public String functionName() {
                return new Not().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> pow(DifferentialFunction<ArrayField> iX, Constant<ArrayField>  i_y) {
        return new AbstractBinaryFunction<ArrayField> (sameDiff,iX, i_y) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().pow(larg().getValue(true), rarg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.pow(larg().getReal(), rarg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                Constant<ArrayField>  ym1 = DifferentialFunctionFactory.this
                        .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
                return rarg().mul(DifferentialFunctionFactory.this.pow(larg(), ym1))
                        .mul(larg().diff(i_v));
            }

            @Override
            public String toString() {
                return "pow(" + larg().toString() + ", " + rarg().toString() + ")";
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return "pow(" + larg().doGetFormula(variables) + ","
                        + rarg().doGetFormula(variables) + ")";
            }

            @Override
            public String functionName() {
                return new Pow().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> sqrt(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sqrt(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.sqrt(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return ((DifferentialFunctionFactory.this.sqrt(arg()).inverse())
                        .div(val(sameDiff.getArrayFactory().one(getResultShape()).mul(2L))))
                        .mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new Sqrt().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> square(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().square(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.pow(arg().getReal(), 2);
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return arg().mul(val(sameDiff.getArrayFactory().one(getResultShape()).mul(2L)))
                        .mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new Pow().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> floor(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().floor(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                throw new RuntimeException("not allowed");
            }

            @Override
            public String functionName() {
                return new Floor().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> relu(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().relu(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return val(sameDiff.getArrayFactory().step(arg().getValue(true))).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new RectifedLinear().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> softmax(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().softmax(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                DifferentialFunction<ArrayField> val = val(getValue(true));
                return val.mul(one(getResultShape()).sub(val)).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new SoftMax().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> hardTanh(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().hardTanh(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.hardTanhDerivative(val(getValue(true))).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new HardTanh().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> hardTanhDerivative(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().hardTanhDerivative(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new HardTanhDerivative().name();
            }
        };
    }




    @Override
    public DifferentialFunction<ArrayField> sigmoid(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sigmoid(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.sigmoidDerivative(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new Sigmoid().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> sigmoidDerivative(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sigmoidDerivative(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return one(getResultShape()).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new SigmoidDerivative().name();
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> sign(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().sign(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return zero(getResultShape());
            }

            @Override
            public String functionName() {
                return new Sign().name();
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> broadcast(DifferentialFunction<ArrayField> iX, int... shape) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().broadcast(arg().getValue(true),shape);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                throw new UnsupportedOperationException();
            }

            @Override
            public String functionName() {
                return "broadcast";
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> repeat(DifferentialFunction<ArrayField> iX, int axis) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().repeat(arg().getValue(true),axis);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                throw new UnsupportedOperationException();
            }

            @Override
            public String functionName() {
                return "repeat";
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> softsign(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().softsign(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return softsignDerivative().mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new SoftSign().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> softsignDerivative(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().softsignDeriviative(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return zero(getResultShape());
            }


            @Override
            public String functionName() {
                return new SoftSignDerivative().name();
            }
        };
    }





    @Override
    public DifferentialFunction<ArrayField> softplus(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().softplus(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.sigmoid(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String functionName() {
                return new SoftPlus().name();
            }
        };
    }


    @Override
    public DifferentialFunction<ArrayField> elu(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().elu(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.eluDerivative(arg()).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new ELU().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> eluDerivative(DifferentialFunction<ArrayField> iX) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().eluDerivative(arg().getValue(true));
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return zero(getResultShape());
            }

            @Override
            public String functionName() {
                return new ELUDerivative().name();
            }
        };
    }




    @Override
    public DifferentialFunction<ArrayField> leakyRelu(DifferentialFunction<ArrayField> iX, double cutoff) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,new Object[]{cutoff}) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().leakyRelu(arg().getValue(true),cutoff);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return DifferentialFunctionFactory.this.leakyReluDerivative(arg(),cutoff).mul(arg().diff(i_v));
            }


            @Override
            public String functionName() {
                return new LeakyReLU().name();
            }
        };
    }



    @Override
    public DifferentialFunction<ArrayField> leakyReluDerivative(DifferentialFunction<ArrayField> iX, double cutoff) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,new Object[]{cutoff}) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().leakyReluDerivative((ArrayField) arg().getValue(true),cutoff);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return zero(getResultShape());
            }

            @Override
            public String functionName() {
                return new LeakyReLUDerivative().name();
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> reshape(DifferentialFunction<ArrayField> iX, int[] shape) {
        shape = Shape.resolveNegativeShapeIfNeccessary(shape);
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,shape, OpState.OpType.SHAPE,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().reshape((ArrayField) arg().getValue(true),shape);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return this;
            }

            @Override
            public String functionName() {
                return "reshape";
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> rollAxis(Variable<ArrayField>  iX, int axis) {
        return new AbstractUnaryFunction<ArrayField> (sameDiff,iX,null, OpState.OpType.SHAPE,null) {

            @Override
            public ArrayField doGetValue() {
                return sameDiff.getArrayFactory().rollAxis((ArrayField) arg().getValue(true),axis);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
                return this;
            }

            @Override
            public String functionName() {
                return "rollAxis";
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> cosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff, iX, i_y, dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().cosineSimilarity(iX, i_y, dimensions);
            }

            private DifferentialFunction<ArrayField> formula() {
                DifferentialFunction<ArrayField> numerator = larg().mul(rarg());
                DifferentialFunction<ArrayField> denom = DifferentialFunctionFactory.this.sqrt(larg().pow(2).mul(rarg().pow(2)));

                return numerator.div(denom);
            }

            @Override
            public double getReal() {
                return formula().getReal();
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return larg().doGetFormula(variables) + " * " + rarg().doGetFormula(variables) + "/" +
                        "sqrt(pow(" + larg().doGetFormula(variables) + ", 2) * pow(" + rarg().doGetFormula(variables) + ", 2))";
            }

            @Override
            public String functionName() {
                return new CosineSimilarity().name();
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return formula().diff(i_v1);
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> euclideanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().euclideanDistance(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "euclideanDistance";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> manhattanDistance(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().manhattanDistance(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "manhattanDistance";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossBinaryXENT(DifferentialFunction<ArrayField> iX,
                                                  DifferentialFunction<ArrayField> i_y,
                                                  int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField>(sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossBinaryXENT(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField>> variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossBinaryXENT";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                DifferentialFunction<ArrayField> numerator = i_y.sub(iX);
                DifferentialFunction<ArrayField> denominator = i_y.mul(i_y.rsub(1.0));
                DifferentialFunction<ArrayField> dLda = denominator.div(denominator);

                /**
                 *   INDArray output = activationFn.getActivation(preOutput.dup(), true);

                 INDArray numerator = output.sub(labels);
                 INDArray denominator = output.mul(output.rsubi(1)); // output * (1-output)
                 INDArray dLda = numerator.divi(denominator);

                 if(mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
                 //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
                 //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
                 //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
                 // error prone - but buy us a tiny bit of performance
                 LossUtil.applyMask(dLda, mask);
                 }

                 INDArray grad = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation functions with weights

                 //Weighted loss function
                 if (weights != null) {
                 if (weights.length() != output.size(1)) {
                 throw new IllegalStateException("Weights vector (length " + weights.length()
                 + ") does not match output.size(1)=" + output.size(1));
                 }

                 grad.muliRowVector(weights);
                 }

                 if (mask != null) {
                 LossUtil.applyMask(grad, mask);
                 }


                 */
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossCosineSimilarity(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossCosineSimilarity(iX,i_y,dimensions);
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossCosineSimilarity";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossHinge(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossHinge";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossKLD(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossKLD(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossKLD";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossL1(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossL1(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossL1";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossL2(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossL2(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossL2";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMAE(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMAE";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAPE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMAPE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMAPE";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMSE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMSE";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossMCXENT(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMCXENT(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMCXENT";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSLE(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossMSLE(iX,i_y,dimensions);
            }



            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossMSLE";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossNegativeLogLikelihood(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossNegativeLogLikelihood(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossNegativeLogLikelihood";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossPoisson(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossPoisson(iX,i_y,dimensions);
            }

            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossPoisson";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> lossSquaredHinge(DifferentialFunction<ArrayField> iX, DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return new AbstractBinaryReduceFunction<ArrayField> (sameDiff,iX,i_y,dimensions) {
            @Override
            protected ArrayField doGetValue() {
                return sameDiff.getArrayFactory().lossSquaredHinge(iX,i_y,dimensions);
            }


            @Override
            public String doGetFormula(List<Variable<ArrayField> > variables) {
                return null;
            }

            @Override
            public String functionName() {
                return "lossSquaredHinge";
            }


            @Override
            public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
                return null;
            }
        };
    }

    @Override
    public DifferentialFunction<ArrayField> mmul(int argNum,
                                        DifferentialFunction<ArrayField> x,
                                        DifferentialFunction<ArrayField> y) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new Mmul<>(sameDiff,x,y,argNum);
    }

    @Override
    public DifferentialFunction<ArrayField> tensorMmul(DifferentialFunction<ArrayField> x,
                                              DifferentialFunction<ArrayField> y,
                                              int[][] dimensions,
                                              int argNum) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new TensorMmul<>(sameDiff,x,y,dimensions,argNum);
    }



    private int getInputLength(DifferentialFunction<ArrayField> func) {
        validateDifferentialFunctionsameDiff(func);
        if(func.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) func.getValue(true);
            int[] inputShape = arrayField.getInput().getShape();
            return ArrayUtil.prod(inputShape);
        }

        throw new IllegalStateException("Only able to compute on array field");
    }

    private DifferentialFunction<ArrayField> doGradChoose(DifferentialFunction<ArrayField> func,
                                                 DifferentialFunction<ArrayField> input,int...axes) {
        if(input.getValue(true) instanceof ArrayField) {
            validateDifferentialFunctionsameDiff(func);
            validateDifferentialFunctionsameDiff(input);

            DifferentialFunction<ArrayField> repeatedGrad = doRepeat(func,input,axes);
            DifferentialFunction<ArrayField> resultRepeated = doRepeat(func.args()[0],input,axes);
            DifferentialFunction<ArrayField> argMaxLocations = eq(input,resultRepeated);
            return argMaxLocations.mul(repeatedGrad).div(sum(argMaxLocations,axes));
        }

        throw new UnsupportedOperationException("Must be an ArrayField argument");

    }


    private DifferentialFunction<ArrayField> doRepeat(DifferentialFunction<ArrayField> func,
                                             DifferentialFunction<ArrayField> input,
                                             int...axes) {
        if(input.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = input.getValue(true);
            int[] inputShape = arrayField.getInput().getShape();
            if(Shape.isWholeArray(inputShape,axes)) {
                validateDifferentialFunctionsameDiff(func);
                validateDifferentialFunctionsameDiff(input);
                return valueArrayOf(input,inputShape);
            }

            for(int i = 0; i < inputShape.length; i++) {
                inputShape[axes[i]] = 1;
            }

            return broadcast(func,inputShape);

        }

        throw new UnsupportedOperationException("Must be an ArrayField " +
                "argument");

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
