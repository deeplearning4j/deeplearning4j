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

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.schema.Schema;
import scala.Tuple2;

import java.util.List;

/**
 * Map an example to a Tuple2<String,JoinValue> for use in a {@link Join}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class MapToJoinValuesFunction implements PairFunction<List<Writable>,String,JoinValue> {

    private boolean left;
    private Join join;

    @Override
    public Tuple2<String, JoinValue> call(List<Writable> writables) throws Exception {

        Schema schema;
        String[] keyColumns;
        if(left){
            schema = join.getLeftSchema();
            keyColumns = join.getKeyColumnsLeft();
        } else {
            schema = join.getRightSchema();
            keyColumns = join.getKeyColumnsRight();
        }

        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(String key : keyColumns){
            int idx = schema.getIndexOfColumn(key);
            if(!first) sb.append("_");
            sb.append(writables.get(idx).toString());
            first = false;
        }

        return new Tuple2<>(sb.toString(), new JoinValue(left,writables));
    }
}
