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

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.array.event.NDArrayMetaData;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQueryFilters;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

@Data
@NoArgsConstructor
@Builder
public class BreakDownComparison implements Serializable {

    private List<NDArrayEvent> first;
    private Map<NDArrayEventType,List<NDArrayEvent>> firstEventsSegmented;
    private List<NDArrayEvent> second;
    private Map<NDArrayEventType,List<NDArrayEvent>> secondEventsSegmented;
    private Set<StackTraceElement> parentPointsOfInvocation;

    public BreakDownComparison(List<NDArrayEvent> first,
                               Map<NDArrayEventType,List<NDArrayEvent>> firstEventsSegmented,
                               List<NDArrayEvent> second,
                               Map<NDArrayEventType,List<NDArrayEvent>>secondEventsSegmented,
                               Set<StackTraceElement> parentPointsOfInvocation) {
        this.first = first;
        this.firstEventsSegmented = executionScopes(first);
        this.second = second;
        this.secondEventsSegmented = executionScopes(second);
        this.parentPointsOfInvocation = parentPointsOfInvocation();
    }


    /**
     * Returns an {@link EventDifference} based on the
     * differences between the two lists
     * @return
     */
    public EventDifference calculateDifference() {
        Pair<NDArrayEvent,NDArrayEvent> diff = firstDifference();
        NDArrayMetaData[] parentDataAtEvent = diff.getFirst().getParentDataAtEvent();
        NDArrayMetaData[] compParents = diff.getSecond().getParentDataAtEvent();
       List<List<Pair<NDArrayEvent, NDArrayEvent>>> ret = new ArrayList<>();
       List<List<BreakDownComparison>> comparisonBreakDowns = new ArrayList<>();
        if(parentDataAtEvent != null && compParents != null) {
            if(parentDataAtEvent.length != compParents.length) {
                return null;
            }

            List<Pair<NDArrayEvent,NDArrayEvent>> differences = new ArrayList<>();
            List<BreakDownComparison> comparisons = new ArrayList<>();
            for(int i = 0; i < parentDataAtEvent.length; i++) {
                NDArrayMetaData firstParent = parentDataAtEvent[i];
                NDArrayMetaData secondParent = compParents[i];
                Nd4jEventLog nd4jEventLog = Nd4j.getExecutioner().getNd4jEventLog();
                BreakDownComparison breakDownComparison = nd4jEventLog.compareEventsFor(firstParent.getId(), secondParent.getId());
                differences.add(breakDownComparison.firstDifference());
                comparisons.add(breakDownComparison);
            }

            ret.add(differences);
            comparisonBreakDowns.add(comparisons);
        }

        return EventDifference.builder().differences(ret)
                .comparisonBreakDowns(comparisonBreakDowns)
                .build();

    }

    /**
     * Returns true if any of the lists are empty
     * @return true if any of the lists are empty
     */
    public boolean anyEmpty() {
        return first == null || first.isEmpty() || second == null || second.isEmpty();
    }

    /**
     * Returns the first event type
     * @param i the index to get the event type for
     * @return the event type at the given index
     */
    public Pair<StackTraceElement,StackTraceElement> stackTracesAt(int i) {
        return Pair.of(first.get(i).getStackTrace()[0], second.get(i).getStackTrace()[0]);
    }

    /**
     * Returns the first event type
     * @param i the index to get the event type for
     * @return the event type at the given index
     */
    public Pair<NDArrayEventType,NDArrayEventType> eventTypesAt(int i) {
        return Pair.of(first.get(i).getNdArrayEventType(), second.get(i).getNdArrayEventType());
    }

    /**
     * Returns the events at the given index
     * @param i the index to get the events for
     * @return the events at the given index
     */

    public Pair<NDArrayEvent,NDArrayEvent> eventsAt(int i) {
        return Pair.of(first.get(i), second.get(i));
    }


    /**
     * Display the first difference according to
     * {@link #firstDifference()}
     * @return the first difference as a pair
     */
    public Pair<String,String> displayFirstDifference() {
        Pair<NDArrayEvent, NDArrayEvent> diff = firstDifference();
        if(diff != null) {
            return Pair.of(diff.getFirst().getDataAtEvent().getData().toString(), diff.getSecond().getDataAtEvent().getData().toString());
        }
        return null;
    }

    /**
     * Returns the first difference between the two lists
     * @return the first difference between the two lists
     */
    public Pair<NDArrayEvent, NDArrayEvent> firstDifference() {
        for(int i = 0; i < first.size(); i++) {
            if(!first.get(i).getDataAtEvent().getData().equals(second.get(i).getDataAtEvent().getData())
            || !first.get(i).getDataAtEvent().getDataBuffer().equals(second.get(i).getDataAtEvent().getDataBuffer())) {
                return Pair.of(first.get(i), second.get(i));
            }
        }
        return null;
    }


    /**
     * Returns the parent points of invocation
     * for the given events accordingv to the definition of
     * {@link NDArrayEvent#parentOfInvocation(StackTraceElement[], StackTraceElement, StackTraceElement)}
     * @return
     */
    public Set<StackTraceElement> parentPointsOfInvocation() {
        if(parentPointsOfInvocation != null) {
            return parentPointsOfInvocation;
        }

        //collect points of invocation from both
        Set<StackTraceElement> ret = new HashSet<>();
        if(first != null) {
            for(NDArrayEvent ndArrayEvent :  first) {
                for(StackTraceElement stackTraceElement: ndArrayEvent.getParentPointOfInvocation()) {
                    ret.add(stackTraceElement);
                }
            }
        }

        if(second != null) {
            for(NDArrayEvent ndArrayEvent :  second) {
                for(StackTraceElement stackTraceElement: ndArrayEvent.getParentPointOfInvocation()) {
                    ret.add(stackTraceElement);
                }
            }
        }



        return ret;
    }


    /**
     * Returns a list of execution scopes
     * for the given events
     * @param events the events to get the execution scopes for
     * @return
     */
    public static Map<NDArrayEventType,List<NDArrayEvent>> executionScopes(List<NDArrayEvent> events) {
        if(events == null)
            throw new IllegalArgumentException("Events must not be null");
        return events.stream().collect(Collectors.groupingBy(NDArrayEvent::getNdArrayEventType));
    }

    /**
     * Returns the index of the first difference between the two lists
     * @return
     */

    public int firstIndexDifference() {
        int ret = -1;
        for(int i = 0; i < first.size(); i++) {
            if(!first.get(i).getDataAtEvent().getData().equals(second.get(i)
                    .getDataAtEvent().getData())) {
                ret = i;
                break;
            }
        }
        return ret;
    }

    /**
     * Filters the events based on the given stack trace query filters
     * @param breakDownComparison the breakdown comparison to filter
     * @param stackTraceQueryFilters the filters to apply
     * @return the filtered breakdown comparison
     */

    public static BreakDownComparison filterEvents(BreakDownComparison breakDownComparison,
                                                   StackTraceQueryFilters stackTraceQueryFilters) {
        if(breakDownComparison.anyEmpty()) {
            return BreakDownComparison.empty();
        }

        List<NDArrayEvent> retFirst = breakDownComparison.getFirst().stream()
                .filter(event ->
                        !StackTraceQueryFilters.shouldFilter(event.getStackTrace(),stackTraceQueryFilters)

                )
                .collect(Collectors.toList());

        List<NDArrayEvent> retSecond = breakDownComparison.getSecond().stream()
                .filter(event ->
                        !StackTraceQueryFilters.shouldFilter(event.getStackTrace(),stackTraceQueryFilters)

                )
                .collect(Collectors.toList());


        BreakDownComparison ret = BreakDownComparison.builder()
                .first(retFirst)
                .second(retSecond)
                .build();
        return ret;
    }

    private static boolean shouldFilter(StackTraceQueryFilters stackTraceQueryFilters, NDArrayEvent event) {
        return !StackTraceQueryFilters.shouldFilter(event.getStackTrace(), stackTraceQueryFilters)
                && !StackTraceQueryFilters.shouldFilter(event.getParentPointOfInvocation().toArray(new StackTraceElement[0]),
                stackTraceQueryFilters);
    }


    /**
     * Returns the first point of origin
     * @return
     */
    public Pair<StackTraceElement,StackTraceElement> pointsOfOrigin() {
        if(first == null || first.isEmpty())
            return null;
        if(second == null || second.isEmpty())
            return null;

        return Pair.of(first.get(0).getPointOfOrigin(), second.get(0).getPointOfOrigin());
    }

    /**
     * Returns the first point of origin
     * @return
     */
    public StackTraceElement pointOfOrigin() {
        if(first == null || first.isEmpty())
            return null;
        if(first == null || first.isEmpty())
            return null;
        if(second == null || second.isEmpty())
            return null;
        if(!first.get(0).getPointOfOrigin().equals(second.get(0).getPointOfOrigin())) {
            return null;
        }
        return first.get(0).getPointOfOrigin();
    }


    /**
     * Returns the first point of invocation
     * @return
     */
    public Pair<StackTraceElement,StackTraceElement> pointsOfInvocation() {
        if(first == null || first.isEmpty())
            return null;
        if(second == null || second.isEmpty())
            return null;

        return Pair.of(first.get(0).getPointOfInvocation(), second.get(0).getPointOfInvocation());
    }


    /**
     * Returns true if any point of origin equals the given stack trace element
     * @param stackTraceElement the stack trace element to check
     * @return true if any point of origin equals the given stack trace element
     */
    public boolean anyPointOfOriginEquals(StackTraceElement stackTraceElement) {
        return first.get(0).getPointOfOrigin().equals(stackTraceElement) || second.get(0).getPointOfOrigin().equals(stackTraceElement);
    }

    /**
     * Returns true if any point of invocation equals the given stack trace element
     * @param stackTraceElement the stack trace element to check
     * @return true if any point of invocation equals the given stack trace element
     */
    public boolean anyPointOfInvocationEquals(StackTraceElement stackTraceElement) {
        return first.get(0).getPointOfInvocation().equals(stackTraceElement) || second.get(0).getPointOfInvocation().equals(stackTraceElement);
    }

    public StackTraceElement pointOfInvocation() {
        if(first == null || first.isEmpty())
            return null;
        if(second == null || second.isEmpty())
            return null;
        if(!first.get(0).getPointOfInvocation().equals(second.get(0).getPointOfInvocation())) {
            return null;
        }
        return first.get(0).getPointOfInvocation();
    }

    public static BreakDownComparison empty() {
        return BreakDownComparison.builder()
                .first(new ArrayList<>())
                .second(new ArrayList<>())
                .build();
    }

}
