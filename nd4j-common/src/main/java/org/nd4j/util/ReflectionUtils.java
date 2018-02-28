/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.nd4j.util;


import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.lang.reflect.Constructor;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * General reflection utils
 */
public class ReflectionUtils {

    /**
     * Cache of constructors for each class. Pins the classes so they
     * can't be garbage collected until ReflectionUtils can be collected.
     */
    protected static final Map<Class<?>, Constructor<?>> CONSTRUCTOR_CACHE = new ConcurrentHashMap<>();

    protected ReflectionUtils() {}



    static private ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();

    public static void setContentionTracing(boolean val) {
        threadBean.setThreadContentionMonitoringEnabled(val);
    }

    private static String getTaskName(long id, String name) {
        if (name == null) {
            return Long.toString(id);
        }
        return id + " (" + name + ")";
    }

    /**
     * Print all of the thread's information and stack traces.
     *
     * @param stream the stream to
     * @param title a string title for the stack trace
     */
    public static void printThreadInfo(PrintWriter stream, String title) {
        final int STACK_DEPTH = 20;
        boolean contention = threadBean.isThreadContentionMonitoringEnabled();
        long[] threadIds = threadBean.getAllThreadIds();
        stream.println("Process Thread Dump: " + title);
        stream.println(threadIds.length + " active threads");
        for (long tid : threadIds) {
            ThreadInfo info = threadBean.getThreadInfo(tid, STACK_DEPTH);
            if (info == null) {
                stream.println("  Inactive");
                continue;
            }
            stream.println("Thread " + getTaskName(info.getThreadId(), info.getThreadName()) + ":");
            Thread.State state = info.getThreadState();
            stream.println("  State: " + state);
            stream.println("  Blocked count: " + info.getBlockedCount());
            stream.println("  Waited count: " + info.getWaitedCount());
            if (contention) {
                stream.println("  Blocked time: " + info.getBlockedTime());
                stream.println("  Waited time: " + info.getWaitedTime());
            }
            if (state == Thread.State.WAITING) {
                stream.println("  Waiting on " + info.getLockName());
            } else if (state == Thread.State.BLOCKED) {
                stream.println("  Blocked on " + info.getLockName());
                stream.println("  Blocked by " + getTaskName(info.getLockOwnerId(), info.getLockOwnerName()));
            }
            stream.println("  Stack:");
            for (StackTraceElement frame : info.getStackTrace()) {
                stream.println("    " + frame.toString());
            }
        }
        stream.flush();
    }

    private static long previousLogTime = 0;


    /**
     * Return the correctly-typed {@link Class} of the given object.
     *
     * @param o object whose correctly-typed <code>Class</code> is to be obtained
     * @return the correctly typed <code>Class</code> of the given object.
     */
    @SuppressWarnings("unchecked")
    public static <T> Class<T> getClass(T o) {
        return (Class<T>) o.getClass();
    }

    // methods to support testing
    static void clearCache() {
        CONSTRUCTOR_CACHE.clear();
    }

    static int getCacheSize() {
        return CONSTRUCTOR_CACHE.size();
    }



}
