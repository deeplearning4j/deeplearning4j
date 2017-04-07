package org.nd4j.autodiff.autodiff;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.autodiff.AbstractFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;

public class DifferentialFunctionFactory<X extends Field<X>> {

    protected AbstractFactory<X> mFactory;
    protected Graph<NDArrayInformation,OpState> graph;
    public DifferentialFunctionFactory(Graph<NDArrayInformation,OpState> graph,AbstractFactory<X> mFactory) {
        if (mFactory != null) {
            this.mFactory = mFactory;
            this.graph = graph;
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    public Constant<X> val(X iX) {
        return new Constant<>(mFactory.graph(),iX, mFactory);
    }

    public ConstantVector<X> val(X... iX) {
        int size = iX.length;
        List<Constant<X>> list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            list.add(val(iX[i]));
        }
        return new ConstantVector<>(graph,mFactory, list);
    }

    // ZeroVector
    public ConstantVector<X> zero(int iSize) {
        List<Constant<X>> list = new ArrayList<>(iSize);
        for (int i = 0; i < iSize; i++) {
            list.add(zero());
        }
        return new ConstantVector<>(graph,mFactory, list);
    }

    public Variable<X> var(String iName, X iX, PreEvaluator<X> preEvaluator) {
        return new Variable<>(mFactory.graph(),iName, iX, mFactory, preEvaluator);
    }

    public Variable<X> var(String iName, X iX) {
        return new Variable<>(mFactory.graph(),iName, iX, mFactory);
    }

    public VariableVector<X> var(String iName, X... iX) {
        int size = iX.length;
        List<Variable<X>> list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            list.add(var(iName + String.valueOf(i), iX[i]));
        }
        return new VariableVector<>(graph,mFactory, list);
    }

    public VariableVector<X> var(String iName, int iSize) {
        List<Variable<X>> list = new ArrayList<>(iSize);
        for (int i = 0; i < iSize; i++) {
            list.add(var(iName + String.valueOf(i), mFactory.zero()));
        }
        return new VariableVector<>(graph,mFactory, list);
    }

    public DifferentialVectorFunction<X> function(DifferentialFunction<X>... iX) {
        int size = iX.length;
        List<DifferentialFunction<X>> list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            list.add(iX[i]);
        }
        return new DifferentialVectorFunction<>(graph,mFactory, list);
    }

    public Zero<X> zero() {
        return new Zero<>(graph,mFactory);
    }

    public One<X> one() {
        return new One<>(graph,mFactory);
    }

    public DifferentialFunction<X> cos(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.cos(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.cos(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return (sin(arg()).mul(arg().diff(i_v))).negate();
            }

            @Override
            public String toString() {
                return "cos(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "cos(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> sin(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sin(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.sin(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return cos(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "sin(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sin(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> tan(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.tan(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.tan(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return (new PolynomialTerm<>(mFactory.graph(),1, cos(arg()), -2)).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "tan(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "tan(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> acos(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.acos(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.acos(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(sqrt(one().minus(arg().pow(2)))).negate();
            }

            @Override
            public String toString() {
                return "acos(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "acos(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> asin(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.asin(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.asin(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(sqrt(one().minus(arg().pow(2))));
            }

            @Override
            public String toString() {
                return "asin(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "asin(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> atan(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.atan(arg().getValue());
            }

                @Override
            public double getReal() {
                return Math.atan(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(one().plus(arg().pow(2)));
            }

            @Override
            public String toString() {
                return "atan(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "atan(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> cosh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.cosh(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.cosh(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return sinh(arg());
            }

            @Override
            public String toString() {
                return "cosh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "cosh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> sinh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sinh(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.sinh(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return cosh(arg());
            }

            @Override
            public String toString() {
                return "sinh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sinh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> tanh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.tanh(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.tanh(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(cosh(arg())).pow(2);
            }

            @Override
            public String toString() {
                return "tanh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "tanh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> acosh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.acosh(arg().getValue());
            }

            @Override
            public double getReal() {
                throw new IllegalStateException("");
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(sqrt(arg().minus(one())).mul(sqrt(arg().plus(one()))));
            }

            @Override
            public String toString() {
                return "acosh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "acosh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> asinh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.asinh(arg().getValue());
            }

            @Override
            public double getReal() {
                throw new IllegalStateException();
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(sqrt(arg().pow(2).plus(one())));
            }

            @Override
            public String toString() {
                return "asinh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "asinh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> atanh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.atanh(arg().getValue());
            }

            @Override
            public double getReal() {
                throw new IllegalStateException();
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().div(one().minus(arg().pow(2)));
            }

            @Override
            public String toString() {
                return "atanh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "atanh(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> exp(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.exp(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.exp(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return exp(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "exp(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "exp(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> log(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.log(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.log(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return new Inverse<>(graph,arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "log(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "log(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> pow(DifferentialFunction<X> iX, Constant<X> i_y) {
        return new AbstractBinaryFunction<X>(mFactory.graph(),iX, i_y) {

            @Override
            public X getValue() {
                return mFactory.pow(larg().getValue(), rarg().getValue());
            }

            @Override
            public double getReal() {
                return Math.pow(larg().getReal(), rarg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                Constant<X> ym1 = DifferentialFunctionFactory.this
                        .val(rarg().getValue().minus(mFactory.one()));
                return rarg().mul(DifferentialFunctionFactory.this.pow(larg(), ym1))
                        .mul(larg().diff(i_v));
            }

            @Override
            public String toString() {
                return "pow(" + larg().toString() + ", " + rarg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "pow(" + larg().getFormula(variables) + ","
                        + rarg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> sqrt(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sqrt(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.sqrt(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return ((sqrt(arg()).inverse())
                        .div(val(mFactory.one().mul(2L))))
                        .mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "sqrt(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sqrt(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> square(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.square(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.pow(arg().getReal(), 2);
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return arg().mul(val(mFactory.one().mul(2L)))
                        .mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "square(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "pow(" + arg().getFormula(variables) + ", 2d )";
            }
        };
    }

    public DifferentialFunction<X> floor(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.floor(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                throw new RuntimeException("not allowed");
            }

            @Override
            public String toString() {
                return "floor(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "floor(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> relu(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.relu(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return val(mFactory.step(arg().getValue())).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "relu(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "relu(" + arg().getFormula(variables) + ")";
            }
        };
    }



    public DifferentialFunction<X> softmax(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.softmax(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                DifferentialFunction<X> val = val(getValue());
                return val.mul(one().minus(val)).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "softmax(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "softmax(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> hardTanh(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.hardTanh(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return hardTanhDerivative(val(getValue())).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "hardtanh(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "hardtanh(" + arg().getFormula(variables) + ")";
            }
        };
    }



    public DifferentialFunction<X> hardTanhDerivative(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.hardTanhDerivative(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "hardtanhDerivative(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "hardtanhDerivative(" + arg().getFormula(variables) + ")";
            }
        };
    }




    public DifferentialFunction<X> sigmoid(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sigmoid(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return sigmoidDerivative(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "sigmoid(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sigmoid(" + arg().getFormula(variables) + ")";
            }
        };
    }



    public DifferentialFunction<X> sigmoidDerivative(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sigmoidDerivative(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return one().mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "sigmoidDerivative(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sigmoidDerivative(" + arg().getFormula(variables) + ")";
            }
        };
    }


    public DifferentialFunction<X> sign(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.sign(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return zero();
            }

            @Override
            public String toString() {
                return "sign(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "sign(" + arg().getFormula(variables) + ")";
            }
        };
    }


    public DifferentialFunction<X> softsign(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.softsign(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return softsignDerivative(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "softsign(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "softsign(" + arg().getFormula(variables) + ")";
            }
        };
    }

    public DifferentialFunction<X> softsignDerivative(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.softsignDeriviative(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return zero();
            }

            @Override
            public String toString() {
                return "softsignDerivative(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "softsignDerivative(" + arg().getFormula(variables) + ")";
            }
        };
    }





    public DifferentialFunction<X> softplus(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.softplus(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return sigmoid(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "softplus(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "softplus(" + arg().getFormula(variables) + ")";
            }
        };
    }


    public DifferentialFunction<X> elu(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.elu(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return eluDerivative(arg()).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "elu(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "elu(" + arg().getFormula(variables) + ")";
            }
        };
    }



    public DifferentialFunction<X> eluDerivative(DifferentialFunction<X> iX) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.eluDerivative(arg().getValue());
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return zero();
            }

            @Override
            public String toString() {
                return "elu(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "elu(" + arg().getFormula(variables) + ")";
            }
        };
    }




    public DifferentialFunction<X> leakyRelu(DifferentialFunction<X> iX,double cutoff) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.leakyRelu(arg().getValue(),cutoff);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return leakyReluDerivative(arg(),cutoff).mul(arg().diff(i_v));
            }

            @Override
            public String toString() {
                return "elu(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "elu(" + arg().getFormula(variables) + ")";
            }
        };
    }



    public DifferentialFunction<X> leakyReluDerivative(DifferentialFunction<X> iX,double cutoff) {
        return new AbstractUnaryFunction<X>(mFactory.graph(),iX) {

            @Override
            public X getValue() {
                return mFactory.leakyReluDerivative(arg().getValue(),cutoff);
            }

            @Override
            public double getReal() {
                return Math.floor(arg().getReal());
            }

            @Override
            public DifferentialFunction<X> diff(Variable<X> i_v) {
                return zero();
            }

            @Override
            public String toString() {
                return "leakyReluDerivative(" + arg().toString() + ")";
            }

            @Override
            public String getFormula(List<Variable<X>> variables) {
                return "leakyReluDerivative(" + arg().getFormula(variables) + ")";
            }
        };
    }
}
