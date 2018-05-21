/*-
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.transform.geo;

import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Computes the Euclidean distance between coordinates found in two columns, divided by an optional third for normalization purposes.
 * A new column (with the specified name) is added as the final column of the output. No other columns are modified.
 *
 * @author saudet
 */
public class CoordinatesDistanceTransform extends BaseColumnsMathOpTransform {

    public final static String DEFAULT_DELIMITER = ":";
    protected String delimiter = DEFAULT_DELIMITER;

    public CoordinatesDistanceTransform(String newColumnName, String firstColumn, String secondColumn,
                    String stdevColumn) {
        this(newColumnName, firstColumn, secondColumn, stdevColumn, DEFAULT_DELIMITER);
    }

    public CoordinatesDistanceTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("firstColumn") String firstColumn, @JsonProperty("secondColumn") String secondColumn,
                    @JsonProperty("stdevColumn") String stdevColumn, @JsonProperty("delimiter") String delimiter) {
        super(newColumnName, MathOp.Add /* dummy op */,
                        stdevColumn != null ? new String[] {firstColumn, secondColumn, stdevColumn}
                                        : new String[] {firstColumn, secondColumn});
        this.delimiter = delimiter;
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName, Schema inputSchema) {
        return new DoubleMetaData(newColumnName);
    }

    @Override
    protected Writable doOp(Writable... input) {
        String[] first = input[0].toString().split(delimiter);
        String[] second = input[1].toString().split(delimiter);
        String[] stdev = columns.length > 2 ? input[2].toString().split(delimiter) : null;

        double dist = 0;
        for (int i = 0; i < first.length; i++) {
            double d = Double.parseDouble(first[i]) - Double.parseDouble(second[i]);
            double s = stdev != null ? Double.parseDouble(stdev[i]) : 1;
            dist += (d * d) / (s * s);
        }
        return new DoubleWritable(Math.sqrt(dist));
    }

    @Override
    public String toString() {
        return "CoordinatesDistanceTransform(newColumnName=\"" + newColumnName + "\",columns="
                        + Arrays.toString(columns) + ",delimiter=" + delimiter + ")";
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
        List row = (List) input;
        String[] first = row.get(0).toString().split(delimiter);
        String[] second = row.get(1).toString().split(delimiter);
        String[] stdev = columns.length > 2 ? row.get(2).toString().split(delimiter) : null;

        double dist = 0;
        for (int i = 0; i < first.length; i++) {
            double d = Double.parseDouble(first[i]) - Double.parseDouble(second[i]);
            double s = stdev != null ? Double.parseDouble(stdev[i]) : 1;
            dist += (d * d) / (s * s);
        }
        return Math.sqrt(dist);
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<List> seq = (List<List>) sequence;
        List<Double> ret = new ArrayList<>();
        for (Object step : seq)
            ret.add((Double) map(step));
        return ret;
    }
}
