package io.skymind.echidna.api.transform.categorical;

import io.skymind.echidna.api.metadata.CategoricalMetaData;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.transform.BaseColumnTransform;

import java.util.*;

/**
 * Convert an integer column to a categorical column, using a provided {@code Map<Integer,String>}
 *
 * @author Alex Black
 */
public class IntegerToCategoricalTransform extends BaseColumnTransform {

    private final Map<Integer, String> map;

    public IntegerToCategoricalTransform(String columnName, Map<Integer, String> map) {
        super(columnName);
        this.map = map;
    }

    public IntegerToCategoricalTransform(String columnName, List<String> list) {
        super(columnName);
        this.map = new LinkedHashMap<>();
        int i = 0;
        for (String s : list) map.put(i++, s);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType) {
        return new CategoricalMetaData(new ArrayList<>(map.values()));
    }

    @Override
    public Writable map(Writable columnWritable) {
        return new Text(map.get(columnWritable.toInt()));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("IntegerToCategoricalTransform(map=[");
        List<Integer> list = new ArrayList<>(map.keySet());
        Collections.sort(list);
        boolean first = true;
        for (Integer i : list) {
            if (!first) sb.append(",");
            sb.append(i).append("=\"").append(map.get(i)).append("\"");
            first = false;
        }
        sb.append("])");
        return sb.toString();
    }
}
