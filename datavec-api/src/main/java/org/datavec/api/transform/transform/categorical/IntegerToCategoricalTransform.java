/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.transform.categorical;

import lombok.Data;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Convert an integer column to a categorical column, using a provided {@code Map<Integer,String>}
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "columnNumber"})
@Data
public class IntegerToCategoricalTransform extends BaseColumnTransform {

    private final Map<Integer, String> map;

    public IntegerToCategoricalTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("map") Map<Integer, String> map) {
        super(columnName);
        this.map = map;
    }

    public IntegerToCategoricalTransform(String columnName, List<String> list) {
        super(columnName);
        this.map = new LinkedHashMap<>();
        int i = 0;
        for (String s : list)
            map.put(i++, s);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new CategoricalMetaData(newColumnName, new ArrayList<>(map.values()));
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
            if (!first)
                sb.append(",");
            sb.append(i).append("=\"").append(map.get(i)).append("\"");
            first = false;
        }
        sb.append("])");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        if (!super.equals(o))
            return false;

        IntegerToCategoricalTransform o2 = (IntegerToCategoricalTransform) o;

        return map != null ? map.equals(o2.map) : o2.map == null;

    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (map != null ? map.hashCode() : 0);
        return result;
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
        return new Text(map.get(input.toString()));
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> values = (List<?>) sequence;
        List<List<Integer>> ret = new ArrayList<>();
        for (Object obj : values) {
            ret.add((List<Integer>) map(obj));
        }
        return ret;
    }
}
