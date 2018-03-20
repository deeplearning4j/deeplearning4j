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

package org.datavec.local.transforms.transform.filter;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Function for executing filter operations
 * @author Alex Black
 */
@AllArgsConstructor
public class LocalFilterFunction implements Function<List<Writable>, Boolean> {

    private final Filter filter;

    @Override
    public Boolean apply(List<Writable> v1) {
        return !filter.removeExample(v1); //return true to keep example (Filter: return true to remove)
    }
}
