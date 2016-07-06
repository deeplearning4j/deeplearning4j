package org.nd4j.etl4j.api.transform.transform.string;

import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This method maps all String values, except those is the specified list, to a single String  value
 *
 * @author Alex Black
 */
public class MapAllStringsExceptListTransform extends BaseStringTransform {

    private final Set<String> exclusionSet;
    private final String newValue;

    public MapAllStringsExceptListTransform(String column, String newValue, List<String> exceptions) {
        super(column);
        this.newValue = newValue;
        this.exclusionSet = new HashSet<>(exceptions);
    }

    @Override
    public Text map(Writable writable) {
        String str = writable.toString();
        if(exclusionSet.contains(str)){
            return new Text(str);
        } else {
            return new Text(newValue);
        }
    }
}
