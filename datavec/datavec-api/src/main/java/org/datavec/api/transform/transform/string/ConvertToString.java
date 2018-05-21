package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Convert any value to a string.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
public class ConvertToString extends BaseStringTransform {
    public ConvertToString(String column) {
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
    public Text map(Writable writable) {
        return new Text(writable.toString());
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
        return input.toString();
    }
}
