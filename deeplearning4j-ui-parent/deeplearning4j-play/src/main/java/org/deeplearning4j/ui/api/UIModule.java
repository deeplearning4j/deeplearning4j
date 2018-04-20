package org.deeplearning4j.ui.api;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.i18n.I18NResource;

import java.util.Collection;
import java.util.List;

/**
 * UIModule encapsulates the user interface functionality for a page or group of pages that rely on data coming
 * from a {@link StatsStorage} instance.<br>
 * When a {@link StatsStorage} object is attached to a {@link UIServer}, the UI server will
 * start receiving {@link StatsStorageEvent} instances; some of these (only the appropriate ones based on the specified
 * TypeIDs from the {@link #getCallbackTypeIDs()} method) will be routed to the UIModule, via {@link #reportStorageEvents(Collection)}.
 * Each UIModule will generally handle one (or at most a few) different types of data (Type IDs); note however that events
 * for a single Type ID can be routed to multiple UI modules, if required.
 * <p>
 * The UIModule also encapsulates the relevant routing information: i.e., what GET/PUT (etc) methods are available for this
 * module, and how those methods should be handled.
 *
 * @author Alex Black
 */
public interface UIModule {

    /**
     * Get the list of Type IDs that should be collected from the registered {@link StatsStorage} instances, and
     * passed on to the {@link #reportStorageEvents(Collection)} method.
     *
     * @return List of relevant Type IDs
     */
    List<String> getCallbackTypeIDs();

    /**
     * Get a list of {@link Route} objects, that specify GET/SET etc methods, and how these should be handled.
     *
     * @return List of routes
     */
    List<Route> getRoutes();

    /**
     * Whenever the {@link UIServer} receives some {@link StatsStorageEvent}s from one of the registered {@link StatsStorage}
     * instances, it will filter these and pass on to the UI module those ones that match one of the Type IDs from
     * {@link #getCallbackTypeIDs()}.<br>
     * Typically, these will be batched together at least somewhat, rather than being reported individually.
     *
     * @param events       List of relevant events (type IDs match one of those from {@link #getCallbackTypeIDs()}
     */
    void reportStorageEvents(Collection<StatsStorageEvent> events);

    /**
     * Notify the UI module that the given {@link StatsStorage} instance has been attached to the UI
     *
     * @param statsStorage    Stats storage that has been attached
     */
    void onAttach(StatsStorage statsStorage);

    /**
     * Notify the UI module that the given {@link StatsStorage} instance has been detached from the UI
     *
     * @param statsStorage    Stats storage that has been detached
     */
    void onDetach(StatsStorage statsStorage);


    List<I18NResource> getInternationalizationResources();
}
