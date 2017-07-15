package org.datavec.api.transform.transform.integer;

import lombok.NoArgsConstructor;
import org.datavec.api.transform.transform.string.BaseStringTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Convert any value to an Integer.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class ConvertToInteger extends BaseIntegerTransform {
    public ConvertToInteger(String column) {
        super(column);
    }

    /**
     * Transform the writable in to a
     * string
     *
     * @param writable the writable to transform
     * @return the string form of this writable
     */
    @Override
    public IntWritable map(Writable writable) {
        return new IntWritable(Integer.parseInt(writable.toString()));
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        return Integer.parseInt(input.toString());
    }
}
