package org.datavec.api.transform.transform.integer;

import org.datavec.api.io.data.IntWritable;
import org.datavec.api.writable.Writable;

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
