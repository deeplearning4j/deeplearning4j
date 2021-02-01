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

package org.datavec.spark.transform.join;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.transform.join.Join;
import org.datavec.api.writable.Writable;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Doing two things here:
 * (a) filter out any unnecessary values, and
 * (b) extract the List<Writable> values from the JoinedValue
 *
 * @author Alex Black
 */
public class FilterAndFlattenJoinedValues implements FlatMapFunction<JoinedValue, List<Writable>> {

    private final Join.JoinType joinType;

    public FilterAndFlattenJoinedValues(Join.JoinType joinType) {
        this.joinType = joinType;
    }

    @Override
    public Iterator<List<Writable>> call(JoinedValue joinedValue) throws Exception {
        boolean keep;
        switch (joinType) {
            case Inner:
                //Only keep joined values where we have both left and right
                keep = joinedValue.isHaveLeft() && joinedValue.isHaveRight();
                break;
            case LeftOuter:
                //Keep all values where left is not missing/null
                keep = joinedValue.isHaveLeft();
                break;
            case RightOuter:
                //Keep all values where right is not missing/null
                keep = joinedValue.isHaveRight();
                break;
            case FullOuter:
                //Keep all values
                keep = true;
                break;
            default:
                throw new RuntimeException("Unknown/not implemented join type: " + joinType);
        }

        if (keep) {
            return Collections.singletonList(joinedValue.getValues()).iterator();
        } else {
            return Collections.emptyIterator();
        }
    }

}
