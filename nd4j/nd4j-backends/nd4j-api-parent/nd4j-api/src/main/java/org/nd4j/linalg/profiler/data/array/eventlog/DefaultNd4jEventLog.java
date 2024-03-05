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
package org.nd4j.linalg.profiler.data.array.eventlog;

import org.nd4j.common.collection.NamedTables;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.dict.BreakDownComparison;
import org.nd4j.linalg.profiler.data.array.event.dict.NDArrayEventDictionary;
import org.nd4j.linalg.profiler.data.array.registry.ArrayRegistry;
import org.nd4j.linalg.profiler.data.array.registry.DefaultArrayRegistry;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.groupingBy;

/**
 * An NDArrayEventLog is a log of {@link NDArrayEvent}
 * instances. Logs are held by {@link org.nd4j.linalg.api.ndarray.INDArray#getId()}
 * <p>
 *     Events cover different types of operations on ndarrays.
 *     Mainly writes and reads. This log is enabled with
 *     {@link org.nd4j.linalg.factory.Environment#isLogNDArrayEvents()}
 *     <p>
 *         This log should not be used in production. This log should only be used
 *         in very limited circumstances to understand difficult to track down
 *         bugs around view creation and other operations.
 *
 *             <p>
 *                 This log is not persisted.
 *
 */
public class DefaultNd4jEventLog implements Nd4jEventLog {
    private Map<Long,List<NDArrayEvent>> events;
    private Map<Long,List<WorkspaceUseMetaData>> workspaceEvents;

    private ArrayRegistry arrayRegistry;

    private NamedTables<String,Integer,StackTraceElement> stackTracePointOfEvent;
    public DefaultNd4jEventLog() {
        events = new ConcurrentHashMap<>();
        workspaceEvents = new ConcurrentHashMap<>();
        arrayRegistry = new DefaultArrayRegistry();
        stackTracePointOfEvent = new NamedTables<>();
    }



    @Override
    public BreakDownComparison compareEventsFor(long arrId, long arrIdComp) {
        List<NDArrayEvent> testBasicTraversal = ndArrayEventsForId(arrId);
        List<NDArrayEvent> testBasicTraversal2 = ndArrayEventsForId(arrIdComp);
        return BreakDownComparison.builder().first(testBasicTraversal).second(testBasicTraversal2).build();
    }

    @Override
    public NDArrayEventDictionary arrayEventsByMethod(String className, String methodName, boolean organizeByInvocation) {
        NDArrayEventDictionary ndArrayEventDictionary = new NDArrayEventDictionary(organizeByInvocation);
        List<NDArrayEvent> testBasicTraversal = Nd4j.getExecutioner().getNd4jEventLog().arrayEventsForClassAndMethod(className, methodName);
        testBasicTraversal.stream().forEach(event -> {
            ndArrayEventDictionary.addEvent(event);
        });

        return ndArrayEventDictionary;
    }

    @Override
    public List<NDArrayEvent> arrayEventsForClassAndMethod(String className, String methodName) {
        if(!this.stackTracePointOfEvent.containsKey(className))
            return new ArrayList<>();
        Table<String, Integer, StackTraceElement> stringIntegerStackTraceElementTable = this.stackTracePointOfEvent.get(className);
        return stringIntegerStackTraceElementTable.values()
                .stream()
                .filter(input -> input != null)
                .map(input -> lookupPointOfEvent(className, methodName, input.getLineNumber()))
                .filter(input -> input != null)
                .map(stackTraceElement
                        -> arrayEventsForStackTracePoint(stackTraceElement.getClassName(),
                        stackTraceElement.getMethodName(),stackTraceElement.getLineNumber()))
                .flatMap(Collection::stream).collect(Collectors.toList());
    }


    @Override
    public List<NDArrayEvent> arrayEventsForStackTracePoint(String className, String methodName, int lineNumber) {
        StackTraceElement stackTraceElement = lookupPointOfEvent(className,methodName,lineNumber);
        if(stackTraceElement == null)
            return new ArrayList<>();
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getPointOfInvocation() != null &&
                input.getPointOfInvocation().equals(stackTraceElement)).collect(Collectors.toList());
    }

    @Override
    public StackTraceElement lookupPointOfEvent(String className, String methodName, int lineNumber) {
        if(!stackTracePointOfEvent.containsKey(className))
            return null;
        if(!stackTracePointOfEvent.get(className).contains(methodName,lineNumber))
            return null;
        return stackTracePointOfEvent.get(className).get(methodName,lineNumber);
    }

    @Override
    public void addStackTracePointOfEvent(StackTraceElement stackTraceElement) {
        if(!stackTracePointOfEvent.containsKey(stackTraceElement.getClassName())) {
            stackTracePointOfEvent.put(stackTraceElement.getClassName(), HashBasedTable.create());
        }

        if(!stackTracePointOfEvent.get(stackTraceElement.getClassName()).contains(stackTraceElement.getMethodName(),stackTraceElement.getLineNumber())) {
            stackTracePointOfEvent.get(stackTraceElement.getClassName()).put(stackTraceElement.getMethodName(),stackTraceElement.getLineNumber(),stackTraceElement);
        }
    }


    @Override
    public List<WorkspaceUseMetaData> workspacesWhere(WorkspaceUseMetaData.EventTypes eventType) {
        return workspaceEvents.values()
                .stream().flatMap(Collection::stream).filter(input -> input.getEventType() == eventType).collect(Collectors.toList());
    }


    private boolean anyEqual(Enum workspaceType,WorkspaceUseMetaData[] metaData) {
        for(WorkspaceUseMetaData workspaceUseMetaData : metaData) {
            if(workspaceUseMetaData.getAssociatedEnum() == workspaceType)
                return true;
        }
        return false;
    }


    @Override
    public List<WorkspaceUseMetaData> workspaceByTypeWithEventType(Enum type, WorkspaceUseMetaData.EventTypes eventType) {
        return workspaceEvents.values().stream().flatMap(Collection::stream).filter(input -> input.getAssociatedEnum() == type && input.getEventType() == eventType).collect(Collectors.toList());
    }

    @Override
    public List<WorkspaceUseMetaData> workspacesByType(Enum type) {
        return workspaceEvents.values().stream().flatMap(Collection::stream).filter(input -> input.getAssociatedEnum() == type)
                .collect(Collectors.toList());
    }

    /**
     * Returns the events for a given {@link MemoryWorkspace#getUniqueId()}
     * @param id the id to get the events for
     * @return
     */
    @Override
    public List<WorkspaceUseMetaData> eventsForWorkspaceUniqueId(long id) {
        return workspaceEvents.get(id);
    }

    @Override
    public void recordWorkspaceEvent(WorkspaceUseMetaData workspaceUseMetaData) {
        if (!workspaceEvents.containsKey(workspaceUseMetaData.getUniqueId()))
            workspaceEvents.put(workspaceUseMetaData.getUniqueId(), new ArrayList<>());
        workspaceEvents.get(workspaceUseMetaData.getUniqueId()).add(workspaceUseMetaData);
    }

    @Override
    public List<Long> parentArraysForArrayId(long id) {
        if(!events.containsKey(id))
            return new ArrayList<>();
        Set<Long> ret = new HashSet<>();
        for(NDArrayEvent event : events.get(id)) {
            ret.addAll(Arrays.stream(event.getParentDataAtEvent()).map(input -> input.getId()).collect(Collectors.toList()));
        }
        return new ArrayList<>(ret);
    }

    @Override
    public List<Long> childArraysForArrayId(long id) {
        if(!events.containsKey(id))
            return new ArrayList<>();
        Set<Long> ret = new HashSet<>();
        for(NDArrayEvent event : events.get(id)) {
            ret.add(event.getDataAtEvent().getId());
        }
        return new ArrayList<>(ret);
    }


    @Override
    public Map<Long,List<WorkspaceUseMetaData>> workspaceEvents() {
        return workspaceEvents;
    }

    @Override
    public Map<Long, List<NDArrayEvent>> ndarrayEvents() {
        return events;
    }

    @Override
    public ArrayRegistry registry() {
        return arrayRegistry;
    }

}
