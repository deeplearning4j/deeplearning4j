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

package org.datavec.api.transform;

import org.datavec.api.transform.reduce.ColumnReduction;
import org.datavec.api.transform.reduce.Reducer;

/**ReduceOp defines the type of column reductions that can be used when reducing
 * a set of values to a single value.<br>
 *
 * Min: take the minimum value<br>
 * Max: take the maximum value<br>
 * Range: output the value max-min<br>
 * Sum: Reduce by summing all values<br>
 * Mean: Reduce by taking the arithmetic mean of the values<br>
 * Stdev: Reduce by calculating the sample standard deviation<br>
 * Count: Reduce by doing a simple count<br>
 * CountUnique: Reduce by counting the number of unique values<br>
 * TakeFirst: Take the first possible  value in the list<br>
 * TakeLast: Take the last possible value in the list<br>
 *
 * <b>Note</b>: For custom reduction operations with {@link Reducer}
 * , use the {@link ColumnReduction}
 * functionality.
 *
 * @author Alex Black
 */
public enum ReduceOp {
    Min,
    Max,
    Range,  //Max - Min
    Sum,
    Mean,
    Stdev,
    Count,
    CountUnique,
    TakeFirst,   //First value
    TakeLast     //Last value

}
