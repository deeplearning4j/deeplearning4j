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

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.util.*;

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


    public final static List<StackTraceQuery> invalidPointOfInvocationClasses = StackTraceQuery.ofClassPatterns(
            false,
            "org.nd4j.linalg.factory.Nd4j",
            "org.nd4j.linalg.api.ndarray.BaseNDArray",
            "org.nd4j.linalg.cpu.nativecpu.CpuNDArrayFactory",
            "org.nd4j.linalg.cpu.nativecpu.NDArray",
            "org.nd4j.linalg.jcublas.JCublasNDArray",
            "org.nd4j.linalg.jcublas.JCublasNDArrayFactory",
            "org.nd4j.linalg.cpu.nativecpu.ops.NativeOpExecutioner",
            "org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner",
            "org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner",
            "org.nd4j.linalg.workspace.BaseWorkspaceMgr",
            "java.lang.Thread",
            "org.nd4j.linalg.factory.BaseNDArrayFactory"
    );
    //regexes for package names that we exclude
    public static List<StackTraceQuery> invalidPointOfInvocationPatterns = queryForProperties();

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

    /**
     * Parent of invocation is an element of the stack trace
     * with a different class altogether.
     * The goal is to be able to segment what is calling a method within the same class.
     * @param elements the elements to get the parent of invocation for
     * @return
     */
    public static Set<StackTraceElement> parentOfInvocation(StackTraceElement[] elements, StackTraceElement pointOfOrigin, StackTraceElement pointOfInvocation) {
        if(elements == null || elements.length < 1)
            return null;

        int pointOfInvocationIndex = -1;
        for(int i = 0; i < elements.length; i++) {
            if(elements[i].equals(pointOfInvocation)) {
                pointOfInvocationIndex = i;
                break;
            }
        }

        if(pointOfInvocationIndex <= 0) {
            return new HashSet<>(Arrays.asList(elements));
        }

        if(pointOfInvocationIndex < 0)
            throw new IllegalArgumentException("Invalid stack trace. Point of invocation not found!");
        int pointOfOriginIndex = -1;
        Set<StackTraceElement> ret = new HashSet<>();
        //loop backwards to find the first non nd4j class
        for(int i = pointOfInvocationIndex + 1; i < elements.length; i++) {
            StackTraceElement element = elements[i];
            if(!StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationClasses,elements[i],i)
                    && !StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationPatterns,elements[i],i) &&
                    !element.getClassName().equals(pointOfOrigin.getClassName())  && !element.getClassName().equals(pointOfInvocation.getClassName())) {
                pointOfOriginIndex = i;
                break;
            }
        }

        if(pointOfOriginIndex < 0) {
            return new HashSet<>(Arrays.asList(elements));
        }
        //this is  what we'll call the "interesting parents", we need to index
        //by multiple parents in order to capture the different parts of the stack tree that could be applicable.
        for(int i = pointOfOriginIndex; i < elements.length; i++) {
            StackTraceElement element = elements[i];

            if(StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationClasses,elements[i],i)
                    || StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationPatterns,elements[i],i) ||
                    element.getClassName().equals(pointOfOrigin.getClassName())  || element.getClassName().equals(pointOfInvocation.getClassName())) {

                break;
            }

            ret.add(elements[i]);
        }

        return ret;
    }

    /**
     * Calls from class is a method that returns
     * all stack trace elements that are from a given class.
     * @param elements the elements to get the calls from class for
     * @param className the class name to get the calls from
     * @return the stack trace elements from the given class
     */
    public static StackTraceElement[] callsFromClass(StackTraceElement[] elements, String className) {
        if(elements == null || elements.length < 1)
            return null;

        List<StackTraceElement> ret = new ArrayList<>();
        for(int i = 0; i < elements.length; i++) {
            if(elements[i].getClassName().equals(className)) {
                ret.add(elements[i]);
            }
        }

        return ret.toArray(new StackTraceElement[0]);
    }

    /**
     * Point of origin is the first non nd4j class in the stack trace.
     * @param elements the elements to get the point of origin for
     * @return
     */
    public static StackTraceElement pointOfOrigin(StackTraceElement[] elements) {
        if(elements == null || elements.length < 1)
            return null;

        int pointOfOriginIndex = 0;
        //loop backwards to find the first non nd4j class
        for(int i = elements.length - 1; i >= 0; i--) {
            if(!StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationClasses,elements[i],i)
                    && !StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationPatterns,elements[i],i)) {
                pointOfOriginIndex = i;
                break;
            }
        }

        return elements[pointOfOriginIndex];
    }

    /**
     *
     * @param elements
     * @return
     */
    public static StackTraceElement pointOfInvocation(StackTraceElement[] elements) {
        if(elements == null || elements.length < 1)
            return null;

        int pointOfInvocationIndex = 0;
        for(int i = 0; i < elements.length; i++) {
            if(!StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationClasses,elements[i],i)
                    && !StackTraceQuery.stackTraceElementMatchesCriteria(invalidPointOfInvocationPatterns,elements[i],i)) {
                pointOfInvocationIndex = i;
                break;
            }
        }

        return elements[pointOfInvocationIndex];
    }

    private static List<StackTraceQuery> queryForProperties() {
        if(System.getProperties().containsKey(ND4JSystemProperties.ND4J_EVENT_LOG_POINT_OF_ORIGIN_PATTERNS)) {
            return StackTraceQuery.ofClassPatterns(true,
                    System.getProperty(ND4JSystemProperties.ND4J_EVENT_LOG_POINT_OF_ORIGIN_PATTERNS).split(","));
        }
        return StackTraceQuery.ofClassPatterns(true,
                "org.junit.*",
                "com.intellij.*",
                "java.*",
                "jdk.*"
        );
    }
}
