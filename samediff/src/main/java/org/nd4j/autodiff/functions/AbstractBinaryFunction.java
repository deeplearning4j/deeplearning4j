package org.nd4j.autodiff.functions;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiff;

@NoArgsConstructor
public abstract class AbstractBinaryFunction<X extends Field<ArrayField>> extends DifferentialFunction<ArrayField> {

    protected DifferentialFunction<ArrayField> m_x1;
    
    protected DifferentialFunction<ArrayField> m_x2;

    public AbstractBinaryFunction(SameDiff sameDiff,
                                  DifferentialFunction<ArrayField> i_v1,
                                  DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            m_x1 = i_v1;
            m_x2 = i_v2;
            validateDifferentialFunctionsameDiff(i_v1);
            validateDifferentialFunctionsameDiff(i_v2);

            this.sameDiff = sameDiff;

            addEdges(sameDiff,i_v1,i_v2,functionName());
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }

    public AbstractBinaryFunction(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
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
            return m_x1.dup();
        return m_x1;
    }


    public DifferentialFunction<ArrayField> rarg() {
        if(m_x2 == this)
            return m_x2.dup();
        return m_x2;
    }



    @Override
    public DifferentialFunction<ArrayField> dup() {
        try {
            return getClass().getConstructor(sameDiff.getClass(),larg()
                    .getClass(),rarg().getClass()).newInstance(sameDiff,larg(),
                    rarg());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
