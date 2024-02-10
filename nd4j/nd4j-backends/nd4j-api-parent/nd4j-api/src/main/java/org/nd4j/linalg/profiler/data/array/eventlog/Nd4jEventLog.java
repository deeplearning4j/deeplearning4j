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

import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.profiler.data.array.*;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.dict.NDArrayEventDictionary;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.array.summary.SummaryOfArrayEvents;
import org.nd4j.linalg.profiler.data.array.watch.WatchCriteria;
import org.nd4j.linalg.profiler.data.array.registry.ArrayRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An NDArrayEventLog is a log of {@link NDArrayEvent}
 * instances. Logs are held by {@link INDArray#getId()}
 *
 * Events cover different types of operations on ndarrays.
 * Mainly writes and reads. This log is enabled with
 * {@link Environment#isLogNDArrayEvents()}
 *
 * This log should not be used in production. This log should only be used
 * in very limited circumstances to understand difficult to track down
 * bugs around view creation and other operations.
 *
 * @author Adam Gibson
 */
public interface Nd4jEventLog {

    /**
     * Returns the {@link NDArrayEvent}
     * grouped by the line number
     * , session name and session index.
     *
     * @param className            the class name to get the event for
     * @param methodName           the method name to get the event for
     * @param organizeByInvocation
     * @return
     */
    NDArrayEventDictionary arrayEventsByMethod(String className, String methodName, boolean organizeByInvocation);

    /**
     * Returns the {@link NDArrayEvent}
     * grouped by the line number
     * @param className the class name to get the event for
     * @param methodName   the method name to get the event for
     * @return
     */
    List<NDArrayEvent> arrayEventsForClassAndMethod(String className, String methodName);

    /**
     * Returns the related {@link NDArrayEvent}
     * based on the point of invocation found in
     * {@link NDArrayEvent#getPointOfInvocation()}
     * @param className the class name to get the event for
     * @param methodName the method name to get the event for
     * @param lineNumber the line number to get the event for
     * @return
     */
    List<NDArrayEvent> arrayEventsForStackTracePoint(String className,String methodName,int lineNumber);

    default List<ArrayDataRevisionSnapshot> arrayDataRevisionSnapshotsFor(INDArray arr) {
        return arrayDataRevisionSnapshotsForId(arr.getId());
    }

    StackTraceElement lookupPointOfEvent(String className, String methodName, int lineNumber);

    void addStackTracePointOfEvent(StackTraceElement stackTraceElement);

    Map<Long,List<ArrayDataRevisionSnapshot>> snapshotData();

    Set<Long> watched();

    Map<Long, SummaryOfArrayEvents> arrayEventsSummaryForWatched();

    List<SummaryOfArrayEvents> eventsForIds(List<Long> ids);

    SummaryOfArrayEvents eventsForArrayId(long id);

    List<ArrayDataRevisionSnapshot> arrayDataRevisionSnapshotsForId(long id);

    List<WatchCriteria> watchCriteria();

    void stopWatching(WatchCriteria... watchCriteria);

    /**
     * Stop Watch an ndarray
     * based on the {@link INDArray#getId()}
     *
     * @param watch
     */
    default void stopWatching(INDArray watch) {
        stopWatching(watch.getId());
    }

    /**
     * Stop watching an ndarray
     * based on the {@link INDArray#getId()}
     *
     * @param id the id of the array to stop watching
     */
    void stopWatching(long id);



    /**
     * Watch ndarrays that fulfill a set of criteria.
     * based on teh {@link WatchCriteria}
     * events logged will monitor array ids
     * coming through that fulfill the specified criteria.
     *
     * Criteria are accumulated.
     * @param watchCriteria
     */
    void watchWithCriteria(WatchCriteria... watchCriteria);

    /**
     * Returns all events for a given id.
     *
     * @param watch  the id to get the events for
     */
    default void watchNDArrayWithId(INDArray watch) {
        watchNDArrayWithId(watch.getId());
    }

    /**
     *  Watch an ndarray based on the {@link INDArray#getId()}
     * @param id
     */
    void watchNDArrayWithId(long id);

    /**
     * Return workspaces with a particular {@link org.nd4j.linalg.api.memory.WorkspaceUseMetaData.EventTypes}
     *
     * @param eventType the event type to filter by
     * @return the list of workspaces for the given event type
     */
    List<WorkspaceUseMetaData> workspacesWhere(WorkspaceUseMetaData.EventTypes eventType);

    /**
     * Returns all events with a closed "child" array
     * which in this case will be the array itself
     * where a workspace was closed or an array was closed
     * when the array was used.
     * @return
     */
    List<NDArrayEvent> eventsWithClosedChildWorkspacesOrArrays();

    /**
     * Returns all events with a closed "parent" array
     * which in this case will be an array where a view was created
     * and had descendenant arrays created from it.
     * where a workspace was closed or an array was closed
     * when the array was used.
     * @return
     */
    List<NDArrayEvent> eventsWithClosedParentWorkspacesOrArrays();

    /**
     * Returns all events with a given workspace event type
     * for the parent workspace.
     * @param eventType the event type to filter by
     * @return the list of events for the given workspace event type
     */
    List<NDArrayEvent> eventsWithParentWorkspaceEventType(WorkspaceUseMetaData.EventTypes eventType);

    /**
     * Returns all events with a given workspace event type
     * for the child workspace.
     * @param eventType the event type to filter by
     * @return the list of events for the given workspace event type
     */
    List<NDArrayEvent> eventsWithChildWorkspaceEventType(WorkspaceUseMetaData.EventTypes eventType);

    /**
     * Returns all events with a given workspace type
     * for the child workspace.
     * @param workspaceType the workspace type to filter by
     * @return the list of events for the given workspace type
     */
    List<NDArrayEvent> eventsWithChildWorkspace(Enum workspaceType);

    /**
     * Returns all events with a given workspace type
     * for the parent workspace.
     * @param workspaceType the workspace type to filter by
     * @return the list of events for the given workspace type
     */
    List<NDArrayEvent> eventsWithParentWorkspace(Enum workspaceType);

    /**
     * Returns all events with a given workspace type
     * @param type the type to get the events for
     * @param eventType the event type to get the events for
     * @return the list of events for the given workspace type
     */
    List<WorkspaceUseMetaData> workspaceByTypeWithEventType(Enum type, WorkspaceUseMetaData.EventTypes eventType);

    /**
     * Returns all events with a given array id
     * @param type the id to get the events for
     * @return the list of events for the given array id
     */
    List<WorkspaceUseMetaData> workspacesByType(Enum type);

    /**
     * Returns the events for a given {@link org.nd4j.linalg.api.memory.MemoryWorkspace#getUniqueId()}
     * @param id the id to get the events for
     * @return
     */

    List<WorkspaceUseMetaData> eventsForWorkspaceUniqueId(long id);

    /**
     * Record a workspace event
     * @param workspaceUseMetaData the meta data to record
     * @param <T> the type of the meta data
     */
    <T> void  recordWorkspaceEvent(WorkspaceUseMetaData workspaceUseMetaData);

    /**
     * Record a workspace event
     *
     * @param event the event to record
     */
    void registerDataUpdatesAsNeeded(NDArrayEvent event);

    void registerDataUpdatesAsNeeded(WorkspaceUseMetaData workspaceUseMetaData);

    /**
     * Register data updates as needed
     * based on the {@link WorkspaceUseMetaData}
     * and {@link NDArrayEvent}

     * @param workspaceUseMetaData the meta data to register
     * @param event the event to register
     */
    void registerDataUpdatesAsNeeded(WorkspaceUseMetaData workspaceUseMetaData, NDArrayEvent event);

    /**
     * Returns the parents for a given id
     * based on the {@link INDArray#getId()}
     * @param id the id to get the parents for
     * @return the parents for the given id
     */
    List<Long> parentArraysForArrayId(long id);

    /**
     * Returns the children for a given id
     * based on the {@link INDArray#getId()}
     * @param id the id to get the children for
     * @return the children for the given id
     */
    List<Long> childArraysForArrayId(long id);

    /**
     * Returns all of the events {@link WorkspaceUseMetaData} mapped by workspace id.
     * @return
     */
    Map<Long,List<WorkspaceUseMetaData>> workspaceEvents();

    /**
     * Return a reference to the underlying
     * events map.
     * @return
     */
    Map<Long, List<NDArrayEvent>> ndarrayEvents();


    /**
     * Returns all events with this array as a parent id.
     * A parent id is an id of an array that was used to create
     * a view. The field used to search for this is {@link NDArrayEvent#getParentArrayId()}
     * @param id the id of the parent array
     * @return
     */
    List<NDArrayEvent> arrayEventsForParentId(long id );

    /**
     * Returns all events with this array as a child id.
     * A child id is an id of an array that was created from a view.
     * The field used to search for this is {@link NDArrayEvent#getChildArrayId()}
     * @param id the id of the child array
     * @return the list of events for the given child id
     */
    List<NDArrayEvent> eventsForArrayChildId(long id);


    ArrayRegistry registry();

    /**
     * Returns all events for a given id.
     * @param id
     * @return
     */
    default List<NDArrayEvent> ndArrayEventsFor(long id) {
        return ndarrayEvents().get(id);
    }

    /**
     * Filter events by type.
     * @param id
     * @param type
     * @return
     */
    default List<NDArrayEvent> ndArrayEventsForType(long id, NDArrayEventType type) {
        List<NDArrayEvent> events = ndArrayEventsFor(id);
        events.removeIf(e -> e.getNdArrayEventType() != type);
        return events;
    }

    /**
     * Add an {@link NDArrayEvent}
     * to the log
     * @param id the id of the array to add
     * @param event the event to add
     */
    default void addToNDArrayLog(long id, NDArrayEvent event) {
        if(!ndarrayEvents().containsKey(id))
            ndarrayEvents().put(id, new ArrayList<>());

        ndarrayEvents().get(id).add(event);
        //register the point of invocation
        //to further analyze where arrays are created.
        if(event.getPointOfInvocation() != null) {
            addStackTracePointOfEvent(event.getPointOfInvocation());
        }

        if(watchCriteria() != null && !watchCriteria().isEmpty()) {
            for(WatchCriteria watchCriteria : watchCriteria()) {
                if(watchCriteria.fulfillsCriteria(event)) {
                    if(event.getChildArrayId() >= 0) {
                        watchNDArrayWithId(event.getChildArrayId());
                    }
                    if(event.getParentArrayId() >= 0) {
                        watchNDArrayWithId(event.getParentArrayId());
                    }

                    registerDataUpdatesAsNeeded(event);

                }
            }
        }

    }

}
