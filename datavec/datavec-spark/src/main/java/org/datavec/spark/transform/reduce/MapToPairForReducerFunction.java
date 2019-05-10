/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform.reduce;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.List;


@AllArgsConstructor
public class MapToPairForReducerFunction implements PairFunction<List<Writable>, String, List<Writable>> {
    public static final String GLOBAL_KEY = "";

    private final IAssociativeReducer reducer;

    @Override
    public Tuple2<String, List<Writable>> call(List<Writable> writables) throws Exception {
        List<String> keyColumns = reducer.getKeyColumns();

        if(keyColumns == null){
            //Global reduction
            return new Tuple2<>(GLOBAL_KEY, writables);
        } else {
            Schema schema = reducer.getInputSchema();
            String key;
            if (keyColumns.size() == 1)
                key = writables.get(schema.getIndexOfColumn(keyColumns.get(0))).toString();
            else {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < keyColumns.size(); i++) {
                    if (i > 0)
                        sb.append("_");
                    sb.append(writables.get(schema.getIndexOfColumn(keyColumns.get(i))).toString());
                }
                key = sb.toString();
            }

            return new Tuple2<>(key, writables);
        }
    }
}
