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

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.DataFrames;

import java.util.ArrayList;
import java.util.List;

/**
 * A create combiner function for use in {@link DataFrames#toRecordsSequence(Dataset<Row>)}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DataFrameToSequenceCreateCombiner implements Function<Iterable<Row>, List<List<Writable>>> {

    private final Schema schema;

    @Override
    public List<List<Writable>> call(Iterable<Row> rows) throws Exception {
        List<List<Writable>> retSeq = new ArrayList<>();
        for (Row v1 : rows) {
            List<Writable> ret = DataFrames.rowToWritables(schema, v1);
            retSeq.add(ret);
        }
        return retSeq;
    }
}
