/*
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

package org.datavec.spark.transform.join;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.transform.join.Join;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.Iterator;
import java.util.List;

/**
 * Execute a join
 */
public class ExecuteJoinFlatMapFunction implements FlatMapFunction<Tuple2<List<Writable>,Iterable<JoinValue>>, List<Writable>> {

    private final ExecuteJoinFlatMapFunctionAdapter adapter;

    public ExecuteJoinFlatMapFunction(Join join) {
        this.adapter = new ExecuteJoinFlatMapFunctionAdapter(join);
    }

    @Override
    public Iterator<List<Writable>> call(Tuple2<List<Writable>, Iterable<JoinValue>> t2) throws Exception {
        return adapter.call(t2).iterator();
    }
}
