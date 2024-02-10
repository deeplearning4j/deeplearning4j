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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.*;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.dict.NDArrayEventDictionary;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.array.summary.SummaryOfArrayEvents;
import org.nd4j.linalg.profiler.data.array.watch.WatchCriteria;
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
    private List<WatchCriteria> watchCriteria;

    private ArrayRegistry arrayRegistry;
    private Set<Long> watched;
    private Map<Long, List<ArrayDataRevisionSnapshot>> arrayDataRevisionSnapshotMap;
    private Map<Long,INDArray> snapshotLatestRevision;

    private NamedTables<String,Integer,StackTraceElement> stackTracePointOfEvent;
    public DefaultNd4jEventLog() {
        events = new ConcurrentHashMap<>();
        workspaceEvents = new ConcurrentHashMap<>();
        watched = new HashSet<>();
        arrayRegistry = new DefaultArrayRegistry();
        watchCriteria = new ArrayList<>();
        arrayDataRevisionSnapshotMap = new ConcurrentHashMap<>();
        snapshotLatestRevision = new ConcurrentHashMap<>();
        stackTracePointOfEvent = new NamedTables<>();
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
    public Map<Long,List<ArrayDataRevisionSnapshot>> snapshotData() {
        return arrayDataRevisionSnapshotMap;
    }

    @Override
    public Set<Long> watched() {
        return watched;
    }


    @Override
    public Map<Long, SummaryOfArrayEvents> arrayEventsSummaryForWatched() {
        if(watched == null || watched.isEmpty())
            return new HashMap<>();
        Map<Long,SummaryOfArrayEvents> ret = new HashMap<>();
        for(Long id : watched) {
            ret.put(id,eventsForArrayId(id));
        }

        return ret;
    }

    @Override
    public List<SummaryOfArrayEvents> eventsForIds(List<Long> ids) {
        List<SummaryOfArrayEvents> ret = new ArrayList<>();
        for(Long id : ids) {
            ret.add(eventsForArrayId(id));
        }
        return ret;
    }
    @Override
    public SummaryOfArrayEvents eventsForArrayId(long id) {
        return SummaryOfArrayEvents.builder()
                .arrayId(id)
                .ndArrayEvents(this.ndArrayEventsFor(id))
                .workspaceUseMetaData(workspaceEvents.get(id))
                .arrayDataRevisionSnapshots(arrayDataRevisionSnapshotsForId(id))
                .build();
    }

    @Override
    public List<ArrayDataRevisionSnapshot> arrayDataRevisionSnapshotsForId(long id) {
        if(!arrayDataRevisionSnapshotMap.containsKey(id))
            return new ArrayList<>();
        return arrayDataRevisionSnapshotMap.get(id);
    }

    @Override
    public List<WatchCriteria> watchCriteria() {
        return watchCriteria;
    }

    @Override
    public void stopWatching(WatchCriteria... watchCriteria) {
        for(WatchCriteria criteria : watchCriteria) {
            this.watchCriteria.remove(criteria);
        }
    }

    @Override
    public void stopWatching(long id) {
        watched.remove(id);
    }


    @Override
    public void watchWithCriteria(WatchCriteria... watchCriteria) {
        for(WatchCriteria criteria : watchCriteria) {
            this.watchCriteria.add(criteria);
        }

    }

    /**
     *  Watch an ndarray for changes.
     *  Automatically adds events to the log
     *  reflecting changes over time to the given array.
     * @param watch the ndarray to watch
     *
     */
    @Override
    public void watchNDArrayWithId(INDArray watch) {
        //whenever an event is logged check for when an array has been changed
        //outside of events logged. Track this based on the value at a timestamp.
        watched.add(watch.getId());
        arrayRegistry.register(watch);
    }

    @Override
    public void watchNDArrayWithId(long id) {
        watched.add(id);

    }

    @Override
    public List<WorkspaceUseMetaData> workspacesWhere(WorkspaceUseMetaData.EventTypes eventType) {
        return workspaceEvents.values()
                .stream().flatMap(Collection::stream).filter(input -> input.getEventType() == eventType).collect(Collectors.toList());
    }

    @Override
    public List<NDArrayEvent> eventsWithClosedChildWorkspacesOrArrays() {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getNdArrayEventType() ==
                        NDArrayEventType.CLOSE ||
                        input.getChildWorkspaceUseMetaData() != null && input.getChildWorkspaceUseMetaData().getEventType() == WorkspaceUseMetaData.EventTypes.CLOSE)
                .collect(Collectors.toList());
    }

    @Override
    public List<NDArrayEvent> eventsWithClosedParentWorkspacesOrArrays() {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getNdArrayEventType() ==
                        NDArrayEventType.CLOSE ||
                        input.getParentWorkspace() != null && input.getParentWorkspace().getEventType() == WorkspaceUseMetaData.EventTypes.CLOSE)
                .collect(Collectors.toList());
    }


    @Override
    public List<NDArrayEvent> eventsWithParentWorkspaceEventType(WorkspaceUseMetaData.EventTypes eventType) {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getParentWorkspace() != null && input.getParentWorkspace().getEventType() == eventType).collect(Collectors.toList());
    }
    @Override
    public List<NDArrayEvent> eventsWithChildWorkspaceEventType(WorkspaceUseMetaData.EventTypes eventType) {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getChildWorkspaceUseMetaData() != null && input.getChildWorkspaceUseMetaData().getEventType() == eventType).collect(Collectors.toList());
    }

    @Override
    public List<NDArrayEvent> eventsWithChildWorkspace(Enum workspaceType) {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getChildWorkspaceUseMetaData() != null &&
                input.getChildWorkspaceUseMetaData().getAssociatedEnum() == workspaceType).collect(Collectors.toList());
    }

    @Override
    public List<NDArrayEvent> eventsWithParentWorkspace(Enum workspaceType) {
        return events.values().stream().flatMap(Collection::stream).filter(input -> input.getParentWorkspace() != null &&
                input.getParentWorkspace().getAssociatedEnum() == workspaceType).collect(Collectors.toList());
    }

    @Override
    public List<WorkspaceUseMetaData> workspaceByTypeWithEventType(Enum type, WorkspaceUseMetaData.EventTypes eventType) {
        return workspaceEvents.values().stream().flatMap(Collection::stream).filter(input -> input.getAssociatedEnum() == type && input.getEventType() == eventType).collect(Collectors.toList());
    }

    @Override
    public List<WorkspaceUseMetaData> workspacesByType(Enum type) {
        return workspaceEvents.values().stream().flatMap(Collection::stream).filter(input -> input.getAssociatedEnum() == type).collect(Collectors.toList());
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
        registerDataUpdatesAsNeeded(workspaceUseMetaData);
    }

    @Override
    public void registerDataUpdatesAsNeeded(NDArrayEvent event) {
        registerDataUpdatesAsNeeded(null,event);
    }
    @Override
    public void registerDataUpdatesAsNeeded(WorkspaceUseMetaData workspaceUseMetaData) {
        registerDataUpdatesAsNeeded(workspaceUseMetaData,null);
    }

    @Override
    public void registerDataUpdatesAsNeeded(WorkspaceUseMetaData workspaceUseMetaData, NDArrayEvent event) {
        for (Long arrayId : watched) {
            if (arrayRegistry.contains(arrayId)) {
                INDArray array = arrayRegistry.lookup(arrayId);
                if (array != null) {
                    List<ArrayDataRevisionSnapshot> arrayDataRevisionSnapshotList = arrayDataRevisionSnapshotMap.get(arrayId);

                    if(arrayDataRevisionSnapshotList == null) {
                        String data = array.toStringFull();
                        arrayDataRevisionSnapshotList = new ArrayList<>();
                        arrayDataRevisionSnapshotMap.put(arrayId,arrayDataRevisionSnapshotList);
                        ArrayDataRevisionSnapshot arrayDataRevisionSnapshot1 = ArrayDataRevisionSnapshot.builder()
                                .arrayId(arrayId)
                                .data(data)
                                .timeStamp(System.currentTimeMillis())
                                .lastEvent(event)
                                .workspaceUseMetaData(workspaceUseMetaData)
                                .build();
                        arrayDataRevisionSnapshotList.add(arrayDataRevisionSnapshot1);
                    } else {
                        ArrayDataRevisionSnapshot arrayDataRevisionSnapshot = arrayDataRevisionSnapshotList.get(arrayDataRevisionSnapshotList.size() - 1);
                        INDArray previousSnapshot = snapshotLatestRevision.get(arrayId);
                        if(!array.equals(previousSnapshot)) {
                            ArrayDataRevisionSnapshot arrayDataRevisionSnapshot1 = ArrayDataRevisionSnapshot.builder()
                                    .arrayId(arrayId)
                                    .data(array.toStringFull())
                                    .timeStamp(System.currentTimeMillis())
                                    .lastEvent(event)
                                    .workspaceUseMetaData(workspaceUseMetaData)
                                    .build();
                            arrayDataRevisionSnapshotList.add(arrayDataRevisionSnapshot1);
                        }
                    }

                }
            }

        }
    }

    @Override
    public List<Long> parentArraysForArrayId(long id) {
        if(!events.containsKey(id))
            return new ArrayList<>();
        Set<Long> ret = new HashSet<>();
        for(NDArrayEvent event : events.get(id)) {
            ret.add(event.getParentArrayId());
        }
        return new ArrayList<>(ret);
    }

    @Override
    public List<Long> childArraysForArrayId(long id) {
        if(!events.containsKey(id))
            return new ArrayList<>();
        Set<Long> ret = new HashSet<>();
        for(NDArrayEvent event : events.get(id)) {
            ret.add(event.getChildArrayId());
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
    public List<NDArrayEvent> arrayEventsForParentId(long id) {
        if(events.containsKey(id))
            return new ArrayList<>(new HashSet<>(events.get(id)).stream().filter(input -> input.getParentArrayId() == id)
                    .collect(Collectors.toList()));
        return new ArrayList<>();
    }

    @Override
    public List<NDArrayEvent> eventsForArrayChildId(long id) {
        if(events.containsKey(id))
            return new ArrayList<>(new HashSet<>(events.get(id)).stream().filter(input -> input.getChildArrayId() == id)
                    .collect(Collectors.toList()));
        return new ArrayList<>();
    }

    @Override
    public ArrayRegistry registry() {
        return arrayRegistry;
    }

}
