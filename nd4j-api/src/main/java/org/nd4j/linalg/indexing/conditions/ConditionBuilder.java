package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.util.ArrayUtil;

/**
 * Mini dsl for building conditions
 *
 * @author Adam Gibson
 */
public class ConditionBuilder {

    private Condition soFar;


    public ConditionBuilder or(Condition...conditions) {
        if(soFar == null)
            soFar = new Or(conditions);
        else {
            soFar = new Or(ArrayUtil.combine(conditions,new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder and(Condition...conditions) {
        if(soFar == null)
            soFar = new And(conditions);
        else {
            soFar = new And(ArrayUtil.combine(conditions,new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder eq(Condition...conditions) {
        if(soFar == null)
            soFar = new ConditionEquals(conditions);
        else {
            soFar = new ConditionEquals(ArrayUtil.combine(conditions,new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder not() {
        if(soFar == null)
            throw new IllegalStateException("No condition to take the opposite of");
        soFar = new Not(soFar);
        return this;
    }

    public Condition build() {
        return soFar;
    }



}
