package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Returns true when all of the conditions equal each other
 * @author Adam Gibson
 */
public class ConditionEquals implements Condition {

    private Condition[] conditions;

    public ConditionEquals(Condition... conditions) {
        this.conditions = conditions;
    }

    @Override
    public Boolean apply(Number input) {
        boolean ret = conditions[0].apply(input);
        for(int i = 1; i < conditions.length; i++) {
            ret = ret == conditions[i].apply(input);
        }
        return ret;
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        boolean ret = conditions[0].apply(input);
        for(int i = 1; i < conditions.length; i++) {
            ret = ret == conditions[i].apply(input);
        }
        return ret;
    }
}
