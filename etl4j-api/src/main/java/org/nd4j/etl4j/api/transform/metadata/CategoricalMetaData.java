package org.nd4j.etl4j.api.transform.metadata;

import io.skymind.echidna.api.ColumnType;
import org.canova.api.writable.Writable;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Metadata for categorical columns.
 * Here, each
 */
public class CategoricalMetaData implements ColumnMetaData {

    private List<String> stateNames;
    private Set<String> stateNamesSet;  //For fast lookup

    public CategoricalMetaData(String... stateNames) {
        this(Arrays.asList(stateNames));
    }

    public CategoricalMetaData(List<String> stateNames) {
        this.stateNames = stateNames;
        stateNamesSet = new HashSet<>(stateNames);
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Categorical;
    }

    @Override
    public boolean isValid(Writable writable) {
        return stateNamesSet.contains(writable.toString());
    }

    @Override
    public CategoricalMetaData clone() {
        return new CategoricalMetaData(stateNames);
    }

    public List<String> getStateNames() {
        return stateNames;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("CategoricalMetaData(stateNames=[");
        boolean first = true;
        for (String s : stateNamesSet) {
            if (!first) sb.append(",");
            sb.append("\"").append(s).append("\"");
            first = false;
        }
        sb.append("])");
        return sb.toString();
    }
}
