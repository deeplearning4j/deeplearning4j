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

package org.nd4j.common.util;

import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.util.ArrayList;
import java.util.List;

/**
 * Utilities for working with stack traces
 * and stack trace elements
 * in a more functional way.
 * This is useful for filtering stack traces
 * and rendering them in a more human readable way.
 * This is useful for debugging and profiling
 * purposes.
 *
 */
public class StackTraceUtils {


    public static StackTraceElement[] reverseCopy(StackTraceElement[] e) {
        StackTraceElement[] copy =  new StackTraceElement[e.length];
        for (int i = 0; i <= e.length / 2; i++) {
            StackTraceElement temp = e[i];
            copy[i] = e[e.length - i - 1];
            copy[e.length - i - 1] = temp;
        }
        return copy;

    }


    /***
     * Returns a potentially reduced stacktrace
     * based on the namepsaces specified
     * in the ignore packages and
     * skipFullPatterns lists
     * @param stackTrace the stack trace to filter
     * @param ignorePackages the packages to ignore
     * @param skipFullPatterns the full patterns to skip
     * @return the filtered stack trace
     */
    public static StackTraceElement[] trimStackTrace(StackTraceElement[] stackTrace, List<StackTraceQuery> ignorePackages, List<StackTraceQuery> skipFullPatterns) {
        if(skipFullPatterns != null && !skipFullPatterns.isEmpty()) {
            if(StackTraceQuery.stackTraceFillsAnyCriteria(skipFullPatterns,stackTrace)) {
                return new StackTraceElement[0];
            }
        }

        if(ignorePackages != null && !ignorePackages.isEmpty()) {
            StackTraceElement[] reverse = reverseCopy(stackTrace);
            List<StackTraceElement> ret = new ArrayList<>();
            //start backwards to find the index of the first non ignored package.
            //we loop backwards to avoid typical unrelated boilerplate
            //like unit tests or ide stack traces
            int startingIndex = -1;
            for(int i = 0; i < reverse.length; i++) {
                if(!StackTraceQuery.stackTraceElementMatchesCriteria(ignorePackages,reverse[i],i)) {
                    startingIndex = i;
                    break;
                }
            }

            //if we didn't find a match, just start at the beginning
            if(startingIndex < 0) {
                startingIndex = 0;
            }

            //loop backwards to present original stack trace
            for(int i = reverse.length - 1; i >= startingIndex; i--) {
                ret.add(reverse[i]);
            }

            return ret.toArray(new StackTraceElement[0]);
        } else {
            List<StackTraceElement> ret = new ArrayList<>();
            for (StackTraceElement stackTraceElement : stackTrace) {
                //note we break because it doesn't make sense to continue rendering when we've hit a package we should be ignoring.
                //this allows a user to specify 1 namespace and ignore anything after it.
               ret.add(stackTraceElement);
            }
            return ret.toArray(new StackTraceElement[0]);
        }

    }


    /**
     * Get the current stack trace as a string.
     * @return
     */
    public static String renderStackTrace(StackTraceElement[] stackTrace, List<StackTraceQuery> ignorePackages, List<StackTraceQuery> skipFullPatterns) {
        StringBuilder stringBuilder = new StringBuilder();
        StackTraceElement[] stackTrace1 = trimStackTrace(stackTrace,ignorePackages,skipFullPatterns);

        for (StackTraceElement stackTraceElement : stackTrace1) {
            stringBuilder.append(stackTraceElement.toString() + "\n");
        }

        return stringBuilder.toString();

    }



    /**
     * Get the current stack trace as a string.
     * @return
     */
    public static String renderStackTrace(StackTraceElement[] stackTrace) {
        return renderStackTrace(stackTrace, null,null );
    }

    /**
     * Get the current stack trace as a string.
     * @return
     */
    public static String currentStackTraceString() {
        Thread currentThread = Thread.currentThread();
        StackTraceElement[] stackTrace = currentThread.getStackTrace();
        return renderStackTrace(stackTrace);
    }

}
