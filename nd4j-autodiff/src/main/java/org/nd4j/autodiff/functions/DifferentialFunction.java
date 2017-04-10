package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


@AllArgsConstructor
public abstract class DifferentialFunction<X extends Field<X>>
        implements Field<DifferentialFunction<X>>,
        Differential<X, DifferentialFunction<X>> {

    protected Graph<NDArrayInformation,OpState> graph;

    /**
     * Get the value of this function
     * @return
     */
    protected abstract X doGetValue();


    /**
     * Equivalent to calling getValue(true)
     * which will by default freeze the graph
     * @return
     */
    public  X getValue() {
        return getValue(true);
    }

    /**
     * Get the value specifying
     * whether to freeze the graph or not
     * @param freeze whether to freeze the graph or not,
     *               this means whether to add nodes to the internal
     *               computation graph or not
     * @return the value of this function
     */
    public  X getValue(boolean freeze) {
        boolean graphAlreadyFrozen = graph.isFrozen();
       //if graph is already frozen leave it frozen
        if(freeze && !graphAlreadyFrozen) {
            graph.freeze();
        }

        X val = doGetValue();

        if(freeze && !graphAlreadyFrozen) {
            graph.unfreeze();
        }

        return val;
    }
    @Override
    public abstract double getReal();


    public  String getFormula(List<Variable<X>> variables) {
        graph.freeze();
        String ret = doGetFormula(variables);
        graph.unfreeze();
        return ret;
    }

    public abstract String doGetFormula(List<Variable<X>> variables);

    public abstract String functionName();

    @Override
    public abstract String toString();

    public boolean isPrecisionOK(int precision) {
        return (13 - precision) > Math.log10(getReal()) + 1;
    }


    public boolean isConstant() {
        return false;
    }


    public boolean isVariable() {
        return false;
    }

    @Override
    public abstract DifferentialFunction<X> diff(Variable<X> i_v1);

    @Override
    public DifferentialFunction<X> plus(DifferentialFunction<X> i_v) {
        return i_v.plused(this);
    }

    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return new Sum<>(graph,i_v, this);
    }

    @Override
    public DifferentialFunction<X> minus(DifferentialFunction<X> i_v) {
        return plus(i_v.negate());
    }

    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        return i_v.muled(this);
    }

    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return new Product<>(graph,i_v, this);
    }

    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        return mul(i_v.inverse());
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return new Inverse<>(graph,this);
    }

    @Override
    public DifferentialFunction<X> negate() {
        return new Negative<>(graph,this);
    }

    @Override
    public DifferentialFunction<X> mul(double i_n) {
        return new PolynomialTerm<X>(graph,i_n, this, 1);
    }

    @Override
    public DifferentialFunction<X> pow(int i_n) {
        return new PolynomialTerm<>(graph,1L, this, i_n);
    }

}
