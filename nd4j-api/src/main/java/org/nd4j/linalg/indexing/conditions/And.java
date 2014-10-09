package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class And implements Condition {

    private Condition[] conditions;

    public And(Condition...conditions) {
        this.conditions = conditions;
    }

    @Override
    public Boolean apply(Number input) {
        boolean ret = conditions[0].apply(input);
        //short circuit: no need to check anything else
        if(!ret)
            return false;
        for(int i = 1; i < conditions.length; i++) {
            ret = ret && conditions[i].apply(input);
        }
        return ret;
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        boolean ret = conditions[0].apply(input);
        //short circuit: no need to check anything else
        if(!ret)
            return false;
        for(int i = 1; i < conditions.length; i++) {
            ret = ret && conditions[i].apply(input);
        }
        return ret;
    }
}
