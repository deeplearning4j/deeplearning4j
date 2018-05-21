package org.datavec.api.transform.condition.column;

import lombok.Data;
import org.datavec.api.transform.condition.SequenceConditionMode;
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.Writable;

/**
 * Created by agibsonccc on 11/26/16.
 */
@Data
public class BooleanColumnCondition extends BaseColumnCondition {
    protected BooleanColumnCondition(String columnName, SequenceConditionMode sequenceConditionMode) {
        super(columnName, sequenceConditionMode);
    }

    /**
     * Returns whether the given element
     * meets the condition set by this operation
     *
     * @param writable the element to test
     * @return true if the condition is met
     * false otherwise
     */
    @Override
    public boolean columnCondition(Writable writable) {
        BooleanWritable booleanWritable = (BooleanWritable) writable;
        return booleanWritable.get();
    }

    /**
     * Condition on arbitrary input
     *
     * @param input the input to return
     *              the condition for
     * @return true if the condition is met
     * false otherwise
     */
    @Override
    public boolean condition(Object input) {
        Boolean bool = (Boolean) input;
        return bool;
    }

    @Override
    public String toString() {
        return getClass().toString();
    }
}
