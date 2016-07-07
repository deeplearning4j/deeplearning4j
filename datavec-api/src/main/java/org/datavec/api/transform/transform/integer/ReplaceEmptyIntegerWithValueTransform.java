package org.datavec.api.transform.transform.integer;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.io.data.IntWritable;
import org.datavec.api.writable.Writable;

/**
 * Replace an empty/missing integer with a certain value.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ReplaceEmptyIntegerWithValueTransform extends BaseIntegerTransform {

    private final int newValueOfEmptyIntegers;

    public ReplaceEmptyIntegerWithValueTransform(String columnName, int newValueOfEmptyIntegers) {
        super(columnName);
        this.newValueOfEmptyIntegers = newValueOfEmptyIntegers;
    }

    @Override
    public Writable map(Writable writable) {
        String s = writable.toString();
        if (s == null || s.isEmpty()) return new IntWritable(newValueOfEmptyIntegers);
        return writable;
    }
}
