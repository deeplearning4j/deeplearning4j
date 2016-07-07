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
import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.join.Join;

import java.util.Collections;
import java.util.List;

/**
 * Doing two things here:
 * (a) filter out any unnecessary values, and
 * (b) extract the List<Writable> values from the JoinedValue
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class FilterAndFlattenJoinedValues implements FlatMapFunction<JoinedValue,List<Writable>> {

    private final Join.JoinType joinType;

    @Override
    public Iterable<List<Writable>> call(JoinedValue joinedValue) throws Exception {
        boolean keep;
        switch (joinType){
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

        if(keep){
            return Collections.singletonList(joinedValue.getValues());
        } else {
            return Collections.emptyList();
        }
    }
}
