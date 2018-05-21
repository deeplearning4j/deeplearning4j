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

package org.datavec.spark.transform.sparkfunction.sequence;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.DataFrames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Combiner function for use in {@link DataFrames#toRecordsSequence(Dataset<Row>)}
 * <p>
 * Assumption here: first two columns are the sequence UUID and the sequence index, as per
 * {@link DataFrames#toDataFrameSequence(Schema, JavaRDD)}
 *
 * @author Alex Black
 */
public class DataFrameToSequenceMergeCombiner
                implements Function2<List<List<Writable>>, List<List<Writable>>, List<List<Writable>>> {

    @Override
    public List<List<Writable>> call(List<List<Writable>> v1, List<List<Writable>> v2) throws Exception {
        List<List<Writable>> out = new ArrayList<>(v1.size() + v2.size());
        out.addAll(v1);
        out.addAll(v2);
        Collections.sort(out, new Comparator<List<Writable>>() {
            @Override
            public int compare(List<Writable> o1, List<Writable> o2) {
                return Integer.compare(o1.get(1).toInt(), o2.get(1).toInt());
            }
        });
        return out;
    }
}
