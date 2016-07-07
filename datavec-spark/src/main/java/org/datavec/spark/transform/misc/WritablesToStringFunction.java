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

package org.datavec.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Simple function to map an example to a String format (such as CSV)
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class WritablesToStringFunction implements Function<List<Writable>,String> {

    private final String delim;

    @Override
    public String call(List<Writable> c) throws Exception {

        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(Writable w : c){
            if(!first) sb.append(delim);
            sb.append(w.toString());
            first = false;
        }

        return sb.toString();
    }

}
