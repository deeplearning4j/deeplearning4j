package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * Replaces String values that match regular expressions.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ReplaceStringTransform extends BaseStringTransform {

    private final Map<String, String> map;

    /**
     * Constructs a new ReplaceStringTransform using the specified
     * @param columnName Name of the column
     * @param map Key: regular expression; Value: replacement value
     */
    public ReplaceStringTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("map") Map<String, String> map) {
        super(columnName);
        this.map = map;
    }

    @Override
    public Text map(final Writable writable) {
        String value = writable.toString();
        value = replaceAll(value);
        return new Text(value);
    }

    @Override
    public Object map(final Object o) {
        String value = o.toString();
        value = replaceAll(value);
        return value;
    }

    private String replaceAll(String value) {
        if (map != null && !map.isEmpty()) {
            for (Map.Entry<String, String> entry : map.entrySet()) {
                value = value.replaceAll(entry.getKey(), entry.getValue());
            }
        }
        return value;
    }

}
