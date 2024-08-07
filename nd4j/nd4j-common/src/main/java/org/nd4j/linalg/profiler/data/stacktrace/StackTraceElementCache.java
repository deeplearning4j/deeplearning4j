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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Utility class for easier usage of stack trace elements.
 * Allows caching and direct lookup of stack trace elements
 * by class name, method  name, and line number.
 *
 *
 */
public class StackTraceElementCache {


    private static Map<StackTraceLookupKey,StackTraceElement> cache = new ConcurrentHashMap<>();



    /**
     * Lookup a stack trace element by key.
     * Note you can also directly use {@link #lookup(String, String, int)}
     * This method is mainly for use cases where you have to deal with multiple stack traces and it could be verbose.
     * Note that when looking up stack traces sometimes user space code can be missing.
     * If a cache entry is missing, we'll attempt to cache the current thread's stack trace elements.
     * Since user input is not guaranteed to be valid, we don't just dynamically create stack trace entries.
     *
     * If your stack trace is missing, ensure you call {@link #storeStackTrace(StackTraceElement[])}
     * on your calling thread.
     *
     * Usually this should be  a transparent process included in certain constructors related to
     * the Environment's NDArray logging being set to true.
     * @param key the key to lookup
     */
    public static StackTraceElement lookup(StackTraceLookupKey key) {
        if(!cache.containsKey(key)) {
            storeStackTrace(Thread.currentThread().getStackTrace());
        }
        return cache.get(key);
    }

    /**
     * Get the cache
     * @return the cache
     */
    public static Map<StackTraceLookupKey,StackTraceElement> getCache() {
        return cache;
    }


    /**
     * Store a stack trace in the cache
     * @param stackTrace the stack trace to store
     */
    public static void storeStackTrace(StackTraceElement[] stackTrace) {
        if(stackTrace == null) {
            return;
        }
        for (StackTraceElement stackTraceElement : stackTrace) {
            if(stackTrace != null)
                storeStackTraceElement(stackTraceElement);
        }
    }

    /**
     * Store a stack trace element in the cache
     * @param stackTraceElement the stack trace element to store
     */
    public static void storeStackTraceElement(StackTraceElement stackTraceElement) {
        if(stackTraceElement == null) {
            return;
        }
        StackTraceLookupKey key = StackTraceLookupKey.builder()
                .className(stackTraceElement.getClassName())
                .methodName(stackTraceElement.getMethodName())
                .lineNumber(stackTraceElement.getLineNumber()).build();
        cache.put(key,stackTraceElement);
    }


    /**
     * Check if the cache contains a stack trace element
     * @param className the class name to check
     * @param methodName the method name to check
     * @param lineNumber the line number to check
     * @return
     */
    public static boolean containsKey(String className,String methodName,int lineNumber) {
        StackTraceLookupKey key = StackTraceLookupKey.builder().className(className).methodName(methodName).lineNumber(lineNumber).build();
        return cache.containsKey(key);
    }

    /**
     * Lookup a stack trace element by class name, method name, and line number
     * @param className the class name to check
     * @param methodName the method name to check
     * @param lineNumber the line number to check
     * @return the stack trace element if it exists, or null if it does not exist
     */
    public static StackTraceElement lookup(String className,String methodName,int lineNumber) {
        StackTraceLookupKey key = StackTraceLookupKey.builder().className(className).methodName(methodName).lineNumber(lineNumber).build();
        return cache.get(key);
    }


}
