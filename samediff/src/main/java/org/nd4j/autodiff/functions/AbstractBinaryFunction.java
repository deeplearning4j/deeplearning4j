package org.nd4j.autodiff.functions;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

@NoArgsConstructor
public abstract class AbstractBinaryFunction<X extends Field<ArrayField>> extends DifferentialFunction<ArrayField> {

    protected DifferentialFunction<ArrayField> m_x1;

    protected DifferentialFunction<ArrayField> m_x2;

    public AbstractBinaryFunction(SameDiff sameDiff,
                                  DifferentialFunction<ArrayField> i_v1,
                                  DifferentialFunction<ArrayField> i_v2) {
        this(sameDiff,i_v1,i_v2, OpState.OpType.TRANSFORM);
    }

    public AbstractBinaryFunction(SameDiff sameDiff,
                                  DifferentialFunction<ArrayField> i_v1,
                                  DifferentialFunction<ArrayField> i_v2,
                                  OpState.OpType opType) {
        super(sameDiff,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            m_x1 = i_v1;
            m_x2 = i_v2;
            validateDifferentialFunctionsameDiff(i_v1);
            validateDifferentialFunctionsameDiff(i_v2);
            validateFunctionReference(i_v1);
            validateFunctionReference(i_v2);
            this.sameDiff = sameDiff;

            addEdges(sameDiff,i_v1,i_v2,functionName(),opType,i_v1.getResultShape(),null);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }

    public AbstractBinaryFunction(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public AbstractBinaryFunction(SameDiff sameDiff,
                                  DifferentialFunction<ArrayField> i_v1,
                                  DifferentialFunction<ArrayField> i_v2,
                                  OpState.OpType opType, Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {
            m_x1 = i_v1;
            m_x2 = i_v2;
            validateDifferentialFunctionsameDiff(i_v1);
            validateDifferentialFunctionsameDiff(i_v2);

            this.sameDiff = sameDiff;

            addEdges(sameDiff,i_v1,i_v2,functionName(),opType,i_v1.getResultShape(),null);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }


    @Override
    public DifferentialFunction<ArrayField>[] args() {
        return new DifferentialFunction[] {larg(),rarg()};
    }

    @Override
    public DifferentialFunction<ArrayField> arg() {
        return larg();
    }

    public DifferentialFunction<ArrayField> larg() {
        if(m_x1 == this)
            return sameDiff.setupFunction(m_x1.dup());
        return sameDiff.setupFunction(m_x1);
    }


    public DifferentialFunction<ArrayField> rarg() {
        if(m_x2 == this)
            return sameDiff.setupFunction(m_x2.dup());
        return sameDiff.setupFunction(m_x2);
    }


    @Override
    public String toString() {
        return functionName() + "(" + larg().toString() + ", " + rarg().toString() + ")";
    }

    @Override
    public String doGetFormula(List<Variable<ArrayField> > variables) {
        return functionName() + "(" + larg().doGetFormula(variables) + ","
                + rarg().doGetFormula(variables) + ")";
    }

    @Override
    public DifferentialFunction<ArrayField> dup() {
        try {
            return getClass().getConstructor(sameDiff.getClass(),DifferentialFunction.class,DifferentialFunction.class).newInstance(sameDiff,larg(),
                    rarg());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
