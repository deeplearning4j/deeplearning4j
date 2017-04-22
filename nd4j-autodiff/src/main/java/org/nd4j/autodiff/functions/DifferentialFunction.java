package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


@AllArgsConstructor
@Data
@NoArgsConstructor
public abstract class DifferentialFunction<X extends Field<X>>
        implements Field<DifferentialFunction<X>>,
        Differential<X, DifferentialFunction<X>> {

    protected Graph<NDArrayInformation,OpState> graph;
    protected OpState opState;

    public DifferentialFunction(Graph<NDArrayInformation, OpState> graph) {
        this.graph = graph;
    }

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
        DifferentialFunction<X> ret = i_v.plused(this);
        return ret;
    }

    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = new Sum<>(graph,i_v, this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> minus(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = plus(i_v.negate());
        return ret;
    }

    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = i_v.muled(this);
        return ret;
    }

    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = new Product<>(graph,i_v, this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = mul(i_v.inverse());
        return ret;
    }

    @Override
    public DifferentialFunction<X> inverse() {
        DifferentialFunction<X> ret = new Inverse<>(graph,this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> negate() {
        DifferentialFunction<X> ret = new Negative<>(graph,this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> mul(double i_n) {
        PolynomialTerm<X> ret =  new PolynomialTerm<>(graph,i_n, this, 1);
        return ret;
    }

    @Override
    public DifferentialFunction<X> pow(int i_n) {
        PolynomialTerm<X> ret = new PolynomialTerm<>(graph,1L, this, i_n);
        return ret;
    }


}
