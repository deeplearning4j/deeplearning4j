package org.datavec.api.transform.transform.string;

import lombok.Data;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Change case (to, e.g, all lower case) of String column.
 *
 * @author dave@skymind.io
 */
@Data
public class ChangeCaseStringTransform extends BaseStringTransform {
    public enum CaseType {
        LOWER, UPPER
    }

    private final CaseType caseType;

    public ChangeCaseStringTransform(String column) {
        super(column);
        this.caseType = CaseType.LOWER; // default is all lower case
    }

    public ChangeCaseStringTransform(@JsonProperty("column") String column,
                    @JsonProperty("caseType") CaseType caseType) {
        super(column);
        this.caseType = caseType;
    }

    private String mapHelper(String input) {
        String result;
        switch (caseType) {
            case UPPER:
                result = input.toUpperCase();
                break;
            case LOWER:
            default:
                result = input.toLowerCase();
                break;
        }
        return result;
    }

    @Override
    public Text map(Writable writable) {
        return new Text(mapHelper(writable.toString()));
    }

    @Override
    public Object map(Object input) {
        return mapHelper(input.toString());
    }
}
