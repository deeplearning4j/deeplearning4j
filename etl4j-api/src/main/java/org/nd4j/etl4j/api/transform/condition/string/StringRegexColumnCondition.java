package org.nd4j.etl4j.api.transform.condition.string;

import org.nd4j.etl4j.api.transform.condition.SequenceConditionMode;
import org.nd4j.etl4j.api.transform.condition.column.BaseColumnCondition;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.writable.Writable;

/**
 * Condition that applies to the values in a String column, using a provided regex.
 * Condition return true if the String matches the regex, or false otherwise<br>
 *
 * <b>Note:</b> Uses Writable.toString(), hence can potentially be applied to non-String columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringRegexColumnCondition extends BaseColumnCondition {

    private final String regex;

    public StringRegexColumnCondition(String columnName, String regex){
        this(columnName, regex, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    public StringRegexColumnCondition(String columnName, String regex, SequenceConditionMode sequenceConditionMode ){
        super(columnName, sequenceConditionMode);
        this.regex = regex;
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return writable.toString().matches(regex);
    }
}
