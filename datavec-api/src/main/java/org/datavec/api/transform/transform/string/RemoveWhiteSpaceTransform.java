package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.io.data.Text;
import org.datavec.api.writable.Writable;

/**
 * String transform that removes all whitespace charaters
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class RemoveWhiteSpaceTransform extends BaseStringTransform {


    public RemoveWhiteSpaceTransform(String columnName) {
        super(columnName);
    }

    @Override
    public Text map(Writable writable) {
        String value = writable.toString().replaceAll("\\s","");
        return new Text(value);
    }
}
