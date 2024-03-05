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

import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceLookupKey;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQueryFilters;
import org.nd4j.shade.guava.collect.Table;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * A breakdown of {@link NDArrayEvent}
 * by stack trace element.
 * This is used for comparing
 * the breakdown of events by stack trace element
 * and comparing them.
 */
public class NDArrayEventMultiMethodStackTraceBreakdown extends ConcurrentHashMap<String,NDArrayEventStackTraceBreakDown> {



    public Map<String,Set<NDArrayEvent>> eventsWithParentInvocation(StackTraceQuery stackTraceQuery,StackTraceQuery targetOrigin) {
        Map<String,Set<NDArrayEvent>> ret = new HashMap<>();
        for(Map.Entry<String,NDArrayEventStackTraceBreakDown> breakdown : entrySet()) {
            Set<NDArrayEvent> events = new LinkedHashSet<>();
            for(Entry<StackTraceElement, Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>>> table : breakdown.getValue().entrySet()) {
                for(List<NDArrayEvent> entry : table.getValue().values()) {
                    for(NDArrayEvent event : entry) {
                        for(StackTraceElement element : event.getParentPointOfInvocation()) {
                            if(stackTraceQuery.filter(element)) {
                                if(targetOrigin != null && targetOrigin.filter(event.getPointOfOrigin())) {
                                    events.add(event);
                                } else {
                                    events.add(event);

                                }
                            }
                        }
                    }
                }
            }

            ret.put(breakdown.getKey(),events);
        }

        return ret;
    }


    public Map<String,Set<StackTraceElement>> possibleParentPointsOfInvocation()  {
        Map<String,Set<StackTraceElement>> ret = new HashMap<>();
        for(Map.Entry<String,NDArrayEventStackTraceBreakDown> breakdown : entrySet()) {
            Set<StackTraceElement> pointsOfInvocation = new HashSet<>();
            breakdown.getValue().values().forEach(table -> {
                for(List<NDArrayEvent> entry : table.values()) {
                    for(NDArrayEvent event : entry) {
                        for(StackTraceElement element : event.getParentPointOfInvocation()) {
                            pointsOfInvocation.add(element);
                        }
                    }
                }
            });
            ret.put(breakdown.getKey(),pointsOfInvocation);
        }

        return ret;
    }
    public Map<String,Set<StackTraceElement>> possiblePointsOfOrigin() {
        Map<String,Set<StackTraceElement>> ret = new HashMap<>();
        for(Map.Entry<String,NDArrayEventStackTraceBreakDown> breakdown : entrySet()) {
            Set<StackTraceElement> pointsOfOrigin = new HashSet<>();
            breakdown.getValue().values().forEach(table -> {
                for(List<NDArrayEvent> entry : table.values()) {
                    for(NDArrayEvent event : entry) {
                        pointsOfOrigin.add(event.getPointOfOrigin());
                    }
                }
            });
            ret.put(breakdown.getKey(),pointsOfOrigin);
        }

        return ret;
    }

    /**
     * Get the possible points of invocation for each method
     * @return
     */
    public Map<String,Set<StackTraceElement>> possiblePointsOfInvocation() {
        Map<String,Set<StackTraceElement>> ret = new HashMap<>();
        for(Map.Entry<String,NDArrayEventStackTraceBreakDown> breakdown : entrySet()) {
            Set<StackTraceElement> pointsOfInvocation = new HashSet<>();
            breakdown.getValue().values().forEach(table -> {
                for(List<NDArrayEvent> entry : table.values()) {
                    for(NDArrayEvent event : entry) {
                        pointsOfInvocation.add(event.getPointOfInvocation());}
                }
            });
            ret.put(breakdown.getKey(),pointsOfInvocation);
        }

        return ret;
    }


    /**
     * Get all the breakdowns mapped by
     * method name
     * @return the breakdowns mapped by method name
     */
    public Map<String,Set<BreakDownComparison>> allBreakDowns() {
        return allBreakDowns(MultiMethodFilter.builder().build());
    }

    /**
     * Get the {@link BreakDownComparison} for a stack frame
     * @param className the class name to get the comparison for
     * @param methodName the method name to get the comparison for
     * @param lineNumber the line number to get the comparison for
     * @param pointOfOriginFilters the point of origin filters to apply
     * @param eventFilters the event filters to apply
     * @return the comparison for the given stack frame
     */
    public Map<String,Set<BreakDownComparison>> comparisonsForStackFrame(String className,
                                                                         String[] methodName,
                                                                         int lineNumber,
                                                                         StackTraceQueryFilters pointOfOriginFilters,
                                                                         StackTraceQueryFilters eventFilters) {

        if(className == null || methodName == null) {
            return new HashMap<>();
        }


        Map<String,Set<BreakDownComparison>> ret = new HashMap<>();
        for(String method : methodName) {
            if(method == null || method.isEmpty()) {
                continue;
            }

            StackTraceElement stackTraceElement = StackTraceLookupKey.stackTraceElementOf(StackTraceLookupKey.of(className, method, lineNumber));
            Map<String, Set<BreakDownComparison>> stringSetMap = allBreakDowns();
            Set<Entry<String, Set<BreakDownComparison>>> entries = stringSetMap.entrySet();

            Map<String,Set<BreakDownComparison>> ret2  = entries.stream()
                    .collect(Collectors.toConcurrentMap(input -> input.getKey(), input -> input.getValue()
                            .stream()
                            .filter(input2 ->
                                    input2.pointOfInvocation()
                                            .equals(stackTraceElement))
                            .filter( input3 -> !StackTraceQueryFilters.shouldFilter(
                                    new StackTraceElement[]{input3.pointsOfOrigin().getFirst()
                                            ,input3.pointsOfOrigin().getSecond()},pointOfOriginFilters))
                            .map(input5 -> BreakDownComparison.filterEvents(input5, eventFilters))
                            .filter(input6 -> !input6.anyEmpty())
                            .collect(Collectors.toSet())));
            ret.putAll(ret2);
        }


        return ret;
    }




    public Map<String,Set<BreakDownComparison>> allBreakDowns(MultiMethodFilter filter) {
        Map<String, Set<BreakDownComparison>> ret = new ConcurrentHashMap<>();
        Map<String, Set<StackTraceElement>> possiblePointsOfOrigin = possiblePointsOfOrigin();
        Map<String, Set<StackTraceElement>> possiblePointsOfInvocation = possiblePointsOfInvocation();
        Map<String, Set<StackTraceElement>> possibleParentPointsOfInvocation = possibleParentPointsOfInvocation();
        for(String s : keySet()) {
            Set<StackTraceElement> possiblePointsOfOriginForMethod = possiblePointsOfOrigin.get(s);
            Set<StackTraceElement> possiblePointsOfInvocationForMethod = possiblePointsOfInvocation.get(s);
            Set<StackTraceElement> possibleParentPointsOfInvocationForMethod = possibleParentPointsOfInvocation.get(s);
            possiblePointsOfOriginForMethod.stream().forEach(origin -> {
                possiblePointsOfOriginForMethod.stream().forEach(compPointOfOrigin -> {
                    possiblePointsOfInvocationForMethod.stream().forEach(invocation -> {
                        possibleParentPointsOfInvocationForMethod.stream().forEach(parentInvocation -> {
                            //check for filters where appropriate to make results easier to work with
                            if(!MultiMethodFilter.isEmpty(filter)) {
                                if (filter.getPointOfOriginFilters() != null && !filter.getPointOfOriginFilters().isEmpty()) {
                                    if(filter.isInclusionFilter()) {
                                        if (StackTraceQuery.stackTraceElementMatchesCriteria(filter.getPointOfOriginFilters(), origin, -1)) {
                                            return;
                                        }
                                    } else {
                                        if (!StackTraceQuery.stackTraceElementMatchesCriteria(filter.getPointOfOriginFilters(), origin, -1)) {
                                            return;
                                        }
                                    }

                                }

                                if (filter.getPointOfInvocationFilters() != null && !filter.getPointOfInvocationFilters().isEmpty()) {
                                    if(filter.isInclusionFilter()) {
                                        if (StackTraceQuery.stackTraceElementMatchesCriteria(filter.getPointOfInvocationFilters(), invocation, -1)) {
                                            return;
                                        }
                                    } else {
                                        if (!StackTraceQuery.stackTraceElementMatchesCriteria(filter.getPointOfInvocationFilters(), invocation, -1)) {
                                            return;
                                        }
                                    }

                                }

                                if (filter.getParentPointOfInvocationFilters() != null && !filter.getParentPointOfInvocationFilters().isEmpty()) {
                                    if(filter.isInclusionFilter()) {
                                        if(StackTraceQuery.stackTraceElementMatchesCriteria(filter.getParentPointOfInvocationFilters(), parentInvocation, -1)) {
                                            return;
                                        }


                                    } else {
                                        if (!StackTraceQuery.stackTraceElementMatchesCriteria(filter.getParentPointOfInvocationFilters(), parentInvocation, -1)) {
                                            return;
                                        }
                                    }

                                }
                            }

                            BreakdownArgs breakdownArgs = BreakdownArgs.builder()
                                    .commonParentOfInvocation(StackTraceLookupKey.of(parentInvocation))
                                    .compPointOfOrigin(StackTraceLookupKey.of(compPointOfOrigin))
                                    .pointOfOrigin(StackTraceLookupKey.of(origin))
                                    .commonPointOfInvocation(StackTraceLookupKey.of(invocation))
                                    .build();
                            BreakDownComparison breakDownComparison = get(s).compareBreakDown(breakdownArgs);
                            //avoid extra noise with empty results
                            if(breakDownComparison.anyEmpty()) {
                                return;
                            }
                            //don't add things that are only the same
                            if(filter.isOnlyIncludeDifferences() && breakDownComparison.firstIndexDifference() < 0) {
                                return;
                            }

                            if(!ret.containsKey(s)) {
                                ret.put(s,new LinkedHashSet<>());
                            }

                            ret.get(s).add(breakDownComparison);
                        });
                    });
                });
            });

        }

        return ret;

    }




}
