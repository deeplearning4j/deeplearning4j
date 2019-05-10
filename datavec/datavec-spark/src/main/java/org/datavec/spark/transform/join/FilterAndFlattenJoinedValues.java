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

package org.datavec.spark.transform.join;

import org.datavec.api.transform.join.Join;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;

import java.util.List;

/**
 * Doing two things here:
 * (a) filter out any unnecessary values, and
 * (b) extract the List<Writable> values from the JoinedValue
 *
 * @author Alex Black
 */
public class FilterAndFlattenJoinedValues extends BaseFlatMapFunctionAdaptee<JoinedValue, List<Writable>> {

    public FilterAndFlattenJoinedValues(Join.JoinType joinType) {
        super(new FilterAndFlattenJoinedValuesAdapter(joinType));
    }

}
