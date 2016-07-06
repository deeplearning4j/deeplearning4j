package org.nd4j.etl4j.api.transform.metadata;

import org.nd4j.etl4j.api.transform.ColumnType;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Metadata for an String column
 *
 * @author Alex Black
 */
public class StringMetaData implements ColumnMetaData {

    //regex + min/max length are nullable: null -> no restrictions on these
    private final String regex;
    private final Integer minLength;
    private final Integer maxLength;

    /**
     * Default constructor with no restrictions on allowable strings
     */
    public StringMetaData() {
        this(null, null, null);
    }

    /**
     * @param mustMatchRegex Nullable. If not null: this is a regex that each string must match in order for the entry
     *                       to be considered valid.
     * @param minLength      Min allowable String length. If null: no restriction on min String length
     * @param maxLength      Max allowable String length. If null: no restriction on max String length
     */
    public StringMetaData(String mustMatchRegex, Integer minLength, Integer maxLength) {
        this.regex = mustMatchRegex;
        this.minLength = minLength;
        this.maxLength = maxLength;
    }


    @Override
    public ColumnType getColumnType() {
        return ColumnType.String;
    }

    @Override
    public boolean isValid(Writable writable) {
        String str = writable.toString();
        int len = str.length();
        if (minLength != null && len < minLength) return false;
        if (maxLength != null && len > maxLength) return false;

        return regex == null || str.matches(regex);
    }

    @Override
    public StringMetaData clone() {
        return new StringMetaData(regex, minLength, maxLength);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StringMetaData(");
        if (minLength != null) sb.append("minLengthAllowed=").append(minLength);
        if (maxLength != null) {
            if (minLength != null) sb.append(",");
            sb.append("maxLengthAllowed=").append(maxLength);
        }
        if (regex != null) {
            if (minLength != null || maxLength != null) sb.append(",");
            sb.append("regex=").append(regex);
        }
        sb.append(")");
        return sb.toString();
    }

}
