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

package org.datavec.spark.transform.analysis;

import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

/**
 * Created by Alex on 4/03/2016.
 */
public class CategoricalToPairFunction implements PairFunction<Writable, String, Integer> {
    @Override
    public Tuple2<String, Integer> call(Writable writable) throws Exception {
        return new Tuple2<>(writable.toString(), 1);
    }
}
