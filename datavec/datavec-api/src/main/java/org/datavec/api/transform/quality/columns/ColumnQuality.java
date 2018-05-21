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

package org.datavec.api.transform.quality.columns;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Base class for the quality of a column
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public abstract class ColumnQuality implements Serializable {

    protected final long countValid;
    protected final long countInvalid;
    protected final long countMissing;
    protected final long countTotal;


    @Override
    public String toString() {
        return "countValid=" + countValid + ", countInvalid=" + countInvalid + ", countMissing=" + countMissing
                        + ", countTotal=" + countTotal;
    }
}
