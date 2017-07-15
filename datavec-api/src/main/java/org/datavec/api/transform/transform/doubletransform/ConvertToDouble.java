package org.datavec.api.transform.transform.doubletransform;

import lombok.NoArgsConstructor;
import org.datavec.api.transform.transform.integer.BaseIntegerTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

/**
 * Convert any value to an Integer.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class ConvertToDouble extends BaseDoubleTransform {
    public ConvertToDouble(String column) {
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
    public DoubleWritable map(Writable writable) {
        return new DoubleWritable(Double.parseDouble(writable.toString()));
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
        return Double.parseDouble(input.toString());
    }
}
