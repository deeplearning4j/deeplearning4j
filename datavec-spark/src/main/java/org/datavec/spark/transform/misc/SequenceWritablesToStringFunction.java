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

package org.datavec.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Simple function to map sequence examples to a String format (such as CSV)
 * with given quote around the string value if it contains the delimiter.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class SequenceWritablesToStringFunction implements Function<List<List<Writable>>, String> {

    public static final String DEFAULT_DELIMITER = ",";
    public static final String DEFAULT_TIME_STEP_DELIMITER = "\n";

    private final String delimiter;
    private final String timeStepDelimiter;
    private final String quote;

    /**
     * Function with default delimiters ("," and "\n")
     */
    public SequenceWritablesToStringFunction() {
        this(DEFAULT_DELIMITER);
    }

    /**
     * function with default delimiter ("\n") between time steps
     * @param delim Delimiter between values within a single time step
     */
    public SequenceWritablesToStringFunction(String delim) {
        this(delim, DEFAULT_TIME_STEP_DELIMITER, null);
    }

    /**
     *
     * @param delim             The delimiter between column values in a single time step (for example, ",")
     * @param timeStepDelimiter The delimiter between time steps (for example, "\n")
     */
    public SequenceWritablesToStringFunction(String delim, String timeStepDelimiter) {
        this(delim, timeStepDelimiter, null);
    }

    @Override
    public String call(List<List<Writable>> c) throws Exception {

        StringBuilder sb = new StringBuilder();
        boolean firstLine = true;
        for (List<Writable> timeStep : c) {
            if (!firstLine) {
                sb.append(timeStepDelimiter);
            }
            WritablesToStringFunction.append(timeStep, sb, delimiter, quote);
            firstLine = false;
        }

        return sb.toString();
    }
}
