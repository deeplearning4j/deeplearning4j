package org.nd4j.etl4j.api.transform.condition.column;

import org.canova.api.io.data.NullWritable;
import org.canova.api.writable.Writable;

/**
 * Condition that applies to the values in any column. Specifically, condition is true
 * if the Writable value is a NullWritable, and false for any other value
 *
 * @author Alex Black
 */
public class NullWritableColumnCondition extends BaseColumnCondition {

    public NullWritableColumnCondition(String column) {
        super(column, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return writable instanceof NullWritable;
    }
}
