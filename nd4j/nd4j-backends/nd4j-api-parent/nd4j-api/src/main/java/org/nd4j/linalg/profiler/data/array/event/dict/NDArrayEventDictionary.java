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
package org.nd4j.linalg.profiler.data.array.event.dict;

import org.jetbrains.annotations.NotNull;
import org.nd4j.common.util.StackTraceUtils;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class NDArrayEventDictionary extends ConcurrentHashMap<StackTraceElement,Map<StackTraceElement,List<NDArrayEvent>>> {

    private boolean organizeByPointOfInvocation = false;

    public NDArrayEventDictionary() {
        this(false);
    }

    public NDArrayEventDictionary(boolean organizeByPointOfInvocation) {
        super();
        this.organizeByPointOfInvocation = organizeByPointOfInvocation;
    }



    /**
     * Break down the comparison
     * of different stack trace elements.
     * This
     * @param pointOfInvocation
     * @return
     */
    public Table<StackTraceElement,StackTraceElement, List<NDArrayEvent>> breakdownComparison(StackTraceElement pointOfInvocation) {
        Map<StackTraceElement, List<NDArrayEvent>> stackTraceElementListMap = groupElementsByInnerPoint(pointOfInvocation);
        Table<StackTraceElement,StackTraceElement, List<NDArrayEvent>> ret = HashBasedTable.create();
        for(Entry<StackTraceElement, List<NDArrayEvent>> stackTraceElementListEntry : stackTraceElementListMap.entrySet()) {
            /**
             * How do we  convert this to just be based on the grouping of the difference
             * of the stack trace element.
             */
            Map<StackTraceElement, List<NDArrayEvent>> listMap = segmentByStackTrace(stackTraceElementListEntry.getValue());
            listMap.entrySet().stream().forEach(entry -> {
                ret.put(stackTraceElementListEntry.getKey(), entry.getKey(), entry.getValue());
            });
        }

        return ret;
    }


    /**
     * Breaks all events for all stack traces present in this dictionary
     * in to a table for each stack trace looking up the stack trace element
     * mapped by the 3 points of interest:
     * 1. The point of invocation: the point where the method is invoked
     * 2. The point of origin the origin point where the user invoked the code in their business logic
     * 3. The parent of invocation: a different class above the point of invocation outside the class
     * to allow differentiation in more complex nested code.
     *
     * Each entry contains a list of events for the given stack trace element
     * The map has the following structure:
     * 1. The key is the point of invocation
     * 2. The row key is the point of origin
     * 3. The column key is the parent of invocation

     * @return
     */
    public NDArrayEventStackTraceBreakDown stackTraceBreakdowns() {
        NDArrayEventStackTraceBreakDown ret = new NDArrayEventStackTraceBreakDown();
        for(StackTraceElement stackTraceElement : keySet()) {
            ret.put(stackTraceElement,breakdownComparison(stackTraceElement));
        }

        return ret;
    }


    /**
     * Segment by stack trace grouping at 3 levels:
     * 1. The point of invocation: the point where the method is invoked
     * 2. The point of origin the origin point where the user invoked the code in their business logic
     * 3. The parent of invocation: a different class above the point of invocation outside the class
     * to allow differentiation in more complex nested code.
     * @param events the events to segment
     * @return the segmented events
     */
    public Map<StackTraceElement, List<NDArrayEvent>> segmentByStackTrace(List<NDArrayEvent> events) {

        Set<StackTraceElement> elements = new HashSet<>();
        for(NDArrayEvent event : events) {
            elements.addAll(event.getParentPointOfInvocation());
        }
       Map<StackTraceElement,List<NDArrayEvent>> ret = new HashMap<>();
        for(NDArrayEvent event : events) {
          for(StackTraceElement stackTraceElement : elements) {
              if(event.getParentPointOfInvocation().contains(stackTraceElement)) {
                  if(!ret.containsKey(stackTraceElement)) {
                      ret.put(stackTraceElement,new ArrayList<>());
                  }

                  ret.get(stackTraceElement).add(event);
              }
          }
        }

        return ret;
    }



    /**
     * Display events for a given point of origin
     * @param pointOfOrigin the point of origin to display events for
     * @param eventType the event type to display
     * @param packagesToSkip the packages to skip this is applicable when
     *                       pruning stack traces using {@link StackTraceUtils#trimStackTrace(StackTraceElement[], List, List)}
     * @param globalSkips the global skips to apply when pruning stack traces
     *                    using {@link StackTraceUtils#trimStackTrace(StackTraceElement[], List, List)}
     * @return the string representation of the events
     */
    public String displayEvents(StackTraceElement pointOfOrigin,
                                NDArrayEventType eventType,
                                List<StackTraceQuery> packagesToSkip,
                                List<StackTraceQuery> globalSkips) {
        StringBuilder builder = new StringBuilder();
        if(!containsKey(pointOfOrigin)) {
            return "No events found for point of origin " + pointOfOrigin;
        }

        Map<StackTraceElement, List<NDArrayEvent>> collect = groupElementsByInnerPoint(pointOfOrigin);

        if(!collect.containsKey(pointOfOrigin)) {
            return "No events found for point of origin " + pointOfOrigin;
        }

        builder.append("Point of origin: " + pointOfOrigin + "\n");
        for(Entry<StackTraceElement, List<NDArrayEvent>> stackTraceElement : collect.entrySet()) {
            for(NDArrayEvent event : stackTraceElement.getValue()) {
                if(event.getNdArrayEventType() == eventType) {
                    StackTraceElement[] pruned = StackTraceUtils.trimStackTrace(event.getStackTrace(),packagesToSkip,globalSkips);
                    builder.append("Comparison point: " + stackTraceElement.getKey() + "\n");
                    builder.append("Data: " + event.getDataAtEvent() + "\n");
                    builder.append("Stack trace: " + StackTraceUtils.renderStackTrace(pruned) + "\n");

                }
            }
        }

        return builder.toString();
    }

    @NotNull
    private Map<StackTraceElement, List<NDArrayEvent>> groupElementsByInnerPoint(StackTraceElement pointOfOrigin) {
        Map<StackTraceElement, List<NDArrayEvent>> stackTraceElementListMap = get(pointOfOrigin);

        if(!containsKey(pointOfOrigin)) {
            return new HashMap<>();
        }

        Map<StackTraceElement, List<NDArrayEvent>> collect = stackTraceElementListMap.values().stream()
                .flatMap(input -> input.stream())
                .sorted(Comparator.comparingLong(NDArrayEvent::getEventId))
                .collect(Collectors.groupingBy(NDArrayEvent::getPointOfInvocation, Collectors.toList()));
        return collect;
    }


    public List<NDArrayEvent> eventsForOrigin(StackTraceElement pointOfOrigin, NDArrayEventType eventType) {
        if (organizeByPointOfInvocation) {
            List<NDArrayEvent> ret = new ArrayList<>();
            if (containsKey(pointOfOrigin)) {
                for (List<NDArrayEvent> ndArrayEvent : get(pointOfOrigin).values()) {
                    ret.addAll(ndArrayEvent.stream().filter(e -> e.getNdArrayEventType() == eventType).collect(Collectors.toList()));
                }

                Collections.sort(ret, Comparator.comparingLong(NDArrayEvent::getEventId));

            }

            return ret;

        } else {

            Set<NDArrayEvent> ret = new LinkedHashSet<>();
            //return all events in the entry set values that have this point of invocation
            for (Map<StackTraceElement, List<NDArrayEvent>> stackTraceElementListMap : values()) {
                for (List<NDArrayEvent> ndArrayEvent : stackTraceElementListMap.values()) {
                    for (NDArrayEvent event : ndArrayEvent) {
                        if (event.getPointOfInvocation().equals(pointOfOrigin)) {
                            ret.add(event);
                        }
                    }
                }
            }


            List<NDArrayEvent> ret2 = new ArrayList<>(ret);
            Collections.sort(ret2, Comparator.comparingLong(NDArrayEvent::getEventId));

            return ret2;

        }

    }

    public List<NDArrayEvent> eventsForInvocation(StackTraceElement pointOfInvocation, NDArrayEventType eventType) {
        if (organizeByPointOfInvocation) {
            List<NDArrayEvent> ret = new ArrayList<>();
            if (containsKey(pointOfInvocation)) {
                for (List<NDArrayEvent> ndArrayEvent : get(pointOfInvocation).values()) {
                    ret.addAll(ndArrayEvent);
                }

                Collections.sort(ret, Comparator.comparingLong(NDArrayEvent::getEventId));

            }

            return ret;

        } else {

            Set<NDArrayEvent> ret = new LinkedHashSet<>();
            //return all events in the entry set values that have this point of invocation
            for (Map<StackTraceElement, List<NDArrayEvent>> stackTraceElementListMap : values()) {
                for (List<NDArrayEvent> ndArrayEvent : stackTraceElementListMap.values()) {
                    for (NDArrayEvent event : ndArrayEvent) {
                        if (event.getPointOfInvocation().equals(pointOfInvocation)) {
                            ret.addAll(ndArrayEvent.stream().filter(e -> e.getNdArrayEventType() == eventType).collect(Collectors.toList()));
                        }
                    }
                }
            }


            List<NDArrayEvent> ret2 = new ArrayList<>(ret);
            Collections.sort(ret2, Comparator.comparingLong(NDArrayEvent::getEventId));

            return ret2;

        }


    }




    /**
     * Add an event to the dictionary
     * organized by {@link #organizeByPointOfInvocation}
     * by default the dictionary is organized by point of origin
     * @param event
     */

    public void addEvent(NDArrayEvent event) {
        StackTraceElement rootKey = organizeByPointOfInvocation ? event.getPointOfInvocation() : event.getPointOfOrigin();
        if(!containsKey(rootKey)) {
            put(rootKey,new ConcurrentHashMap<>());
        }

        Map<StackTraceElement, List<NDArrayEvent>> stackTraceElementListMap = get(rootKey);
        StackTraceElement subKey = organizeByPointOfInvocation ? event.getPointOfOrigin() : event.getPointOfInvocation();
        if(!stackTraceElementListMap.containsKey(subKey)) {
            stackTraceElementListMap.put(event.getPointOfInvocation(),new ArrayList<>());
        }

        stackTraceElementListMap.get(subKey).add(event);


    }


}
