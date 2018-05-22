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

package org.datavec.spark.transform.sequence;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Spark function to map an example to a pair, by using some of the column values as the key.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class SparkMapToPairByMultipleColumnsFunction
                implements PairFunction<List<Writable>, List<Writable>, List<Writable>> {

    private final int[] keyColumnIdxs;

    @Override
    public Tuple2<List<Writable>, List<Writable>> call(List<Writable> writables) throws Exception {
        List<Writable> keyOut = new ArrayList<>(keyColumnIdxs.length);
        for (int keyColumnIdx : keyColumnIdxs) {
            keyOut.add(writables.get(keyColumnIdx));
        }
        return new Tuple2<>(keyOut, writables);
    }
}
