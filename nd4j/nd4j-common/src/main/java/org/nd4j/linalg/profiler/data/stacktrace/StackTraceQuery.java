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
package org.nd4j.linalg.profiler.data.stacktrace;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StackTraceQuery implements Serializable {
    @Builder.Default
    private int lineNumber = -1;
    private String className;
    private String methodName;
    @Builder.Default
    private int occursWithinLineCount = -1;
    @Builder.Default
    private boolean exactMatch = false;
    @Builder.Default
    private boolean regexMatch = false;

    @Builder.Default
    private int lineNumberBegin = -1;
    @Builder.Default
    private int lineNumberEnd = -1;

    private static Map<String, Pattern> cachedPatterns = new HashMap<>();


    /**
     * Create a list of queries
     * based on the fully qualified class name patterns.
     *
     * @param regex
     * @param classes the classes to create queries for
     * @return the list of queries
     */
    public static List<StackTraceQuery> ofClassPatterns(boolean regex, String... classes) {
        List<StackTraceQuery> ret = new ArrayList<>();
        for (String s : classes) {
            if(regex) {
                cachedPatterns.put(s, Pattern.compile(s));
            }
            ret.add(StackTraceQuery.builder()
                    .regexMatch(regex)
                    .className(s).build());
        }

        return ret;
    }


    /**
     * Returns true if the stack trace element matches the given criteria
     * @param queries the queries to match on
     * @param stackTrace the stack trace to match on
     *                   (note that the stack trace is in reverse order)
     * @return true if the stack trace element matches the given criteria
     */
    public static boolean stackTraceFillsAnyCriteria(List<StackTraceQuery> queries, StackTraceElement[] stackTrace) {
        if(stackTrace == null)
            return false;
        if(queries == null)
            return false;
        for (int j = 0; j < stackTrace.length; j++) {
            StackTraceElement line = stackTrace[j];
            //parse line like this: org.deeplearning4j.nn.layers.recurrent.BidirectionalLayer.backpropGradient(BidirectionalLayer.java:153)

            if (stackTraceElementMatchesCriteria(queries, line, j)) return true;
        }

        return false;
    }



    /**
     * Returns true if the stack trace element matches the given criteria
     * @param queries the queries to match on
     * @param line the stack trace element to match on
     * @param j the index of the line
     * @return true if the stack trace element matches the given criteria
     */
    public static boolean stackTraceElementMatchesCriteria(List<StackTraceQuery> queries, StackTraceElement line, int j) {
        for (StackTraceQuery query : queries) {
            //allow -1 on line number to mean any line number  also allow methods that are unspecified to mean any method
            //also check for the line count occurrence -1 means any
            boolean classNameMatch = isClassNameMatch(query.getClassName(), query, line.getClassName());
            //null or empty method name means any method name, depending on whether an exact match is required
            //return we consider it a match
            boolean methodNameMatch = isClassNameMatch(query.getMethodName(), query, line.getMethodName());
            //< 0 line means any line number
            boolean lineNumberMatch = query.getLineNumber() < 0 || query.getLineNumber() == line.getLineNumber();
            //whether the user specifies if the match is within the stack trace depth. what this is  for is
            //to filter stack trace matches to a certain depth. for example, if you want to match a stack trace
            //that occurs within a certain method, you can specify the depth of the stack trace to match on.
            boolean matchesStackTraceDepth = (query.getOccursWithinLineCount() <= j || query.getOccursWithinLineCount() < 0);
            boolean inLineRange = (query.getLineNumberBegin() <= line.getLineNumber() && query.getLineNumberEnd() >= line.getLineNumber()) || (query.getLineNumberBegin() < 0 && query.getLineNumberEnd() < 0);
            if (classNameMatch
                    && methodNameMatch
                    && lineNumberMatch
                    && inLineRange
                    && matchesStackTraceDepth) {
                return true;

            }

        }
        return false;
    }

    private static boolean isClassNameMatch(String query, StackTraceQuery query1, String line) {
        boolean classNameMatch = (query == null || query.isEmpty()) ||
                (query1.isExactMatch() ? line.equals(query) : line.contains(query)) ||
                (query1.isRegexMatch() ? line.matches(query) : line.contains(query));
        return classNameMatch;
    }



    public static int indexOfFirstDifference(StackTraceElement[] first,StackTraceElement[] second) {
        int min = Math.min(first.length,second.length);
        for(int i = 0; i < min; i++) {
            if(!first[i].equals(second[i])) {
                return i;
            }
        }
        return -1;
    }
}
