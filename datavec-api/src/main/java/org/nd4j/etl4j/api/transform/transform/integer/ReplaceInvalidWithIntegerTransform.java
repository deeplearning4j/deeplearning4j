package org.nd4j.etl4j.api.transform.transform.integer;

import org.nd4j.etl4j.api.io.data.IntWritable;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Replace an invalid (non-integer) value in a column with a specified integer
 */
public class ReplaceInvalidWithIntegerTransform extends BaseIntegerTransform {

    private final int intValue;

    public ReplaceInvalidWithIntegerTransform(String column, int intValue) {
        super(column);
        this.intValue = intValue;
    }

    @Override
    public Writable map(Writable writable) {
        if(inputSchema.getMetaData(columnNumber).isValid(writable)){
            return writable;
        } else {
            return new IntWritable(intValue);
        }
    }
}
