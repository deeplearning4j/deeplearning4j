/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.List;

@AllArgsConstructor
public class ColumnToKeyPairTransform implements PairFunction<List<Writable>, Writable, Long> {
    private final int columnIndex;

    @Override
    public Tuple2<Writable, Long> call(List<Writable> list) throws Exception {
        return new Tuple2<>(list.get(columnIndex), 1L);
    }
}
