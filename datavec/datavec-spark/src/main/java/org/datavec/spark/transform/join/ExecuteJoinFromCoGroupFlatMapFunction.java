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

package org.datavec.spark.transform.join;

import org.datavec.api.transform.join.Join;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import scala.Tuple2;

import java.util.List;

/**
 * Execute a join
 *
 * @author Alex Black
 */
public class ExecuteJoinFromCoGroupFlatMapFunction extends
                BaseFlatMapFunctionAdaptee<Tuple2<List<Writable>, Tuple2<Iterable<List<Writable>>, Iterable<List<Writable>>>>, List<Writable>> {

    public ExecuteJoinFromCoGroupFlatMapFunction(Join join) {
        super(new ExecuteJoinFromCoGroupFlatMapFunctionAdapter(join));
    }
}
