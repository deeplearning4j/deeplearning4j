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
package org.nd4j.linalg.profiler.data.array.event;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.event.dict.NDArrayEventDictionary;
import org.nd4j.linalg.profiler.data.array.event.dict.NDArrayEventStackTraceBreakDown;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceElementCache;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Data
@NoArgsConstructor
@Builder
public class NDArrayEvent implements Serializable  {

    private StackTraceElement[] stackTrace;
    private static final AtomicLong arrayCounter = new AtomicLong(0);

    /**
     * When {@link Environment#isFuncTracePrintJavaOnly()} is true,
     * this will be recorded on PUT calls only. WHen working with op execution
     * we will have this information.
     */
    private StackTraceElement[] arrayWriteCreationStackTrace;
    //for views
    private StackTraceElement[] parentArrayCreationStackTrace;
    private NDArrayEventType ndArrayEventType;
    /**
     * This is mainly for view creation.
     * When arrays are created, they are given an id.
     * When creating a view the parent array is the
     * array the view was derived from.
     */
    @Builder.Default
    private long parentArrayId = -1;
    @Builder.Default
    private long childArrayId = -1;
    private NDArrayMetaData dataAtEvent;
    private WorkspaceUseMetaData childWorkspaceUseMetaData;
    private WorkspaceUseMetaData parentWorkspace;
    @Builder.Default
    private long eventTimeStamp = System.nanoTime();
    private StackTraceElement pointOfInvocation;
    private StackTraceElement pointOfOrigin;
    private List<StackTraceElement> parentPointOfInvocation;
    private String scopeName;
    @Builder.Default
    private int scopeIndex = -1;

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
    @Builder.Default
    private long eventId = -1;

    public NDArrayEvent(final StackTraceElement[] stackTrace, final StackTraceElement[] arrayWriteCreationStackTrace,
                        final StackTraceElement[] parentArrayCreationStackTrace, final NDArrayEventType ndArrayEventType,
                        final long parentArrayId,
                        final long childArrayId, final NDArrayMetaData dataAtEvent,
                        final WorkspaceUseMetaData childWorkspaceUseMetaData, final WorkspaceUseMetaData parentWorkspace,
                        final long eventTimeStamp, final StackTraceElement pointOfInvocation,
                        final StackTraceElement pointOfOrigin,
                        final List<StackTraceElement> parentPointOfInvocation,
                        String scopeName,int scopeIndex,long eventId) {
        this.stackTrace = stackTrace;
        this.arrayWriteCreationStackTrace = arrayWriteCreationStackTrace;
        this.parentArrayCreationStackTrace = parentArrayCreationStackTrace;
        this.ndArrayEventType = ndArrayEventType;
        this.parentArrayId = parentArrayId;
        this.childArrayId = childArrayId;
        this.dataAtEvent = dataAtEvent;
        this.childWorkspaceUseMetaData = childWorkspaceUseMetaData;
        this.parentWorkspace = parentWorkspace;
        this.eventTimeStamp = eventTimeStamp;
        this.pointOfInvocation = pointOfInvocation(stackTrace);
        this.pointOfOrigin = pointOfOrigin(stackTrace);
        this.parentPointOfInvocation = parentOfInvocation(stackTrace,this.pointOfOrigin,this.pointOfInvocation);
        this.eventId = arrayCounter.incrementAndGet();
        //store the stack trace for easier lookup later
        StackTraceElementCache.storeStackTrace(stackTrace);
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


    /**
     * A getter for a {@link StackTraceElement} array
     * that can handle grouping functions by wrapping the
     * array in a {@link StackTraceKey} that uses the to string of the
     * stack trace element array as the key.
     * @return
     */
    public StackTraceKey getStackTraceKey() {
        return new StackTraceKey(stackTrace);
    }

    /**
     * Render events by session and line number.
     * This map is created using {@link Nd4jEventLog#arrayEventsByMethod(String, String, boolean)}
     *
     * @param className             the class name to render
     * @param methodName            the method name to get the grouped events for.,
     * @param eventType             the event type to render
     * @param classesPackagesToSkip the classes and packages to skip, these are regular expressions typically of the form
     *                              package name: .*package_name.* or class name: .*ClassName.*
     * @param globalSkips           the global skips to apply to all stack trace elements. If any element matches the stack trace avoid rendering.
     * @param organizeByInvocation
     * @return the rendered events by session and line number
     */
    public static NDArrayEventDictionary groupedEvents(
            String className,
            String methodName,
            NDArrayEventType eventType,
            List<StackTraceQuery> classesPackagesToSkip,
            List<StackTraceQuery> globalSkips, boolean organizeByInvocation) {
        return groupedEvents(Nd4j.getExecutioner().getNd4jEventLog().arrayEventsByMethod(className,methodName, organizeByInvocation),
                eventType,classesPackagesToSkip,globalSkips);

    }

    /**
     * Render events by session and line number.
     * This map is created using {@link Nd4jEventLog#arrayEventsByMethod(String, String, boolean)}
     * The class name and method are implicit in the returned map and thus only sorted by line number.
     * @param eventsBySessionAndLineNumber the events to render
     * @param eventType the event type to render
     * @param classesPackagesToSkip the classes and packages to skip, these are regular expressions typically of the form
     *                              package name: .*package_name.* or class name: .*ClassName.*
     * @param globalSkips the global skips to apply to all stack trace elements. If any element matches the stack trace avoid rendering.
     * @return the rendered events by session and line number
     */
    public static NDArrayEventDictionary groupedEvents(
            NDArrayEventDictionary eventsBySessionAndLineNumber,
            NDArrayEventType eventType,
            List<StackTraceQuery> classesPackagesToSkip,
            List<StackTraceQuery> globalSkips) {
        NDArrayEventDictionary ret = new NDArrayEventDictionary();
        //sorted by line number with each map being the session index and the list of events
        for(val entry : eventsBySessionAndLineNumber.entrySet()) {
            if(!entry.getValue().isEmpty()) {
                for(val entry1 : entry.getValue().entrySet()) {
                    //filter by relevant event type
                    entry1.getValue().stream()
                            .filter(input -> !shouldSkipEvent(eventType, globalSkips, input))
                            .filter(input -> !shouldSkipEvent(eventType, classesPackagesToSkip, input))
                            .collect(Collectors.groupingBy(NDArrayEvent::getPointOfOrigin)).entrySet().stream()
                            .forEach(entry2 -> {
                                Map<StackTraceElement,List<NDArrayEvent>> differencesGrouped = new LinkedHashMap<>();
                                NDArrayEvent first = entry2.getValue().get(0);
                                StackTraceElement firstDiff = null;
                                for(int i = 1; i < entry2.getValue().size(); i++) {
                                    int firstDiffIdx = StackTraceQuery.indexOfFirstDifference(first.getStackTrace(),entry2.getValue().get(i).getStackTrace());
                                    if(firstDiffIdx >= 0 && firstDiff == null) {
                                        firstDiff = first.getStackTrace()[firstDiffIdx];
                                        differencesGrouped.put(firstDiff,new ArrayList<>());
                                        differencesGrouped.get(firstDiff).add(first);
                                    }
                                    //this is the case where we bumped in to a stack trace with the same path.
                                    if(firstDiffIdx < 0) {
                                        if(firstDiff != null) {
                                            differencesGrouped.get(firstDiff).add(entry2.getValue().get(i));
                                        } else {
                                            differencesGrouped.put(entry2.getValue().get(i).getStackTrace()[0],new ArrayList<>());
                                            differencesGrouped.get(entry2.getValue().get(i).getStackTrace()[0]).add(entry2.getValue().get(i));
                                        }

                                    } else {
                                        StackTraceElement diffInComp = entry2.getValue().get(i).getStackTrace()[firstDiffIdx];
                                        if(!differencesGrouped.containsKey(diffInComp)) {
                                            differencesGrouped.put(diffInComp,new ArrayList<>());
                                        }

                                        differencesGrouped.get(diffInComp).add(entry2.getValue().get(i));
                                    }

                                }



                                //append to string grouped by similar stack trace differences
                                //this allows easier reading of the different events sorted
                                //by a common stack trace
                                if(!differencesGrouped.isEmpty()) {
                                    //events sorted by the common parts of the stack trace
                                    differencesGrouped.values().stream().flatMap(input -> input.stream()).forEach(events -> {
                                        ret.addEvent(events);
                                    });
                                }
                                else {
                                    Map<StackTraceElement, List<NDArrayEvent>> collect = entry2.getValue().stream().collect(Collectors.groupingBy(NDArrayEvent::getPointOfOrigin));
                                    collect.values().stream().flatMap(input -> input.stream()).forEach(events -> {
                                        ret.addEvent(events);
                                    });
                                }
                            });
                }
            }

        }

        return ret;

    }




    /**
     * Break down events that occur
     * in a given class and method
     * with the given event type.
     * This is a short cut method for calling
     * {@Link #groupedEvents(String, String, NDArrayEventType, List, List, boolean)}
     * followed by {@link NDArrayEventDictionary#stackTraceBreakdowns()}
     * @param className the class name to break down
     * @param methodName the method name to break down
     * @param eventType the event type to break down
     * @param classesPackagesToSkip the classes and packages to skip, these are regular expressions typically of the form
     * @param globalSkips the global skips to apply to all stack trace elements. If any element matches the stack trace avoid rendering.
     * @param organizeByInvocation whether to organize by invocation or not
     * @return
     */
    public static NDArrayEventStackTraceBreakDown stacktraceBreakDowns(String className,
                                                                       String methodName,
                                                                       NDArrayEventType eventType,
                                                                       List<StackTraceQuery> classesPackagesToSkip,
                                                                       List<StackTraceQuery> globalSkips, boolean organizeByInvocation) {
        return groupedEvents(Nd4j.getExecutioner().getNd4jEventLog().arrayEventsByMethod(className,methodName, organizeByInvocation),
                eventType,classesPackagesToSkip,globalSkips).stackTraceBreakdowns();
    }


    private static boolean shouldSkipEvent(NDArrayEventType eventType, List<StackTraceQuery> globalSkips, NDArrayEvent input) {
        if(globalSkips == null || globalSkips.isEmpty())
            return input.getNdArrayEventType() == eventType;
        else {
            return input.getNdArrayEventType() == eventType && !
                    StackTraceQuery.stackTraceFillsAnyCriteria(globalSkips, input.getStackTrace());
        }

    }

    /**
     * Parent of invocation is an element of the stack trace
     * with a different class altogether.
     * The goal is to be able to segment what is calling a method within the same class.
     * @param elements
     * @return
     */
    public static List<StackTraceElement> parentOfInvocation(StackTraceElement[] elements,StackTraceElement pointOfOrigin,StackTraceElement pointOfInvocation) {
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
            return Arrays.asList(elements);
        }

        if(pointOfInvocationIndex < 0)
            throw new IllegalArgumentException("Invalid stack trace. Point of invocation not found!");
        int pointOfOriginIndex = -1;
        List<StackTraceElement> ret = new ArrayList<>();
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
            return Arrays.asList(elements);
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
     *
     * @param elements
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
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("=========================================\n");
        sb.append("NDArrayEvent: \n");
        sb.append("NDArrayEventType: " + ndArrayEventType + "\n");
        if(stackTrace != null) {
            sb.append("-----------------------------------------\n");
            sb.append("StackTrace: " + stackTrace + "\n");
            sb.append("-----------------------------------------\n");

        }

        if(arrayWriteCreationStackTrace != null) {
            sb.append("-----------------------------------------\n");
            sb.append("Array Write Creation Stack Trace: " + arrayWriteCreationStackTrace + "\n");
            sb.append("-----------------------------------------\n");

        }

        if(parentArrayCreationStackTrace != null) {
            sb.append("-----------------------------------------\n");
            sb.append("Parent Array Creation Stack Trace: " + parentArrayCreationStackTrace + "\n");
            sb.append("-----------------------------------------\n");

        }


        sb.append("=========================================\n");
        return sb.toString();
    }
}
