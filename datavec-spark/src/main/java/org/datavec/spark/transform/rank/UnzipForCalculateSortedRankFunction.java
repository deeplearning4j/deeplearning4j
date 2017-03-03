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

package org.datavec.spark.transform.rank;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple helper function for use in executing CalculateSortedRank
 *
 * @author Alex Black
 */
public class UnzipForCalculateSortedRankFunction
                implements Function<Tuple2<Tuple2<Writable, List<Writable>>, Long>, List<Writable>> {
    @Override
    public List<Writable> call(Tuple2<Tuple2<Writable, List<Writable>>, Long> v1) throws Exception {
        List<Writable> inputWritables = new ArrayList<>(v1._1()._2());
        inputWritables.add(new LongWritable(v1._2()));
        return inputWritables;
    }
}
