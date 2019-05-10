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

package org.datavec.spark.transform.misc.comparator;

import lombok.AllArgsConstructor;
import scala.Tuple2;

import java.io.Serializable;
import java.util.Comparator;

/**
 * Simple comparator: Compare {@code Tuple2<T,Long>} by Long value
 */
@AllArgsConstructor
public class Tuple2Comparator<T> implements Comparator<Tuple2<T, Long>>, Serializable {

    private final boolean ascending;

    @Override
    public int compare(Tuple2<T, Long> o1, Tuple2<T, Long> o2) {
        if (ascending)
            return Long.compare(o1._2(), o2._2);
        else
            return -Long.compare(o1._2(), o2._2);
    }
}
