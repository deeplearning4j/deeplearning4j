package org.deeplearning4j.spark.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StatsStorageRouterProvider;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.listeners.VanillaStatsStorageRouterProvider;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/** Created by huitseeker on 2/15/17. */
@Slf4j
public class SparkListenable {

    protected TrainingMaster trainingMaster;
    private List<TrainingListener> listeners = new ArrayList<>();

    /**
     * This method allows you to specify trainingListeners for this model.
     *
     * @param listeners Iteration listeners
     */
    public void setListeners(@NonNull Collection<TrainingListener> listeners) {
        this.listeners.clear();
        this.listeners.addAll(listeners);
        if (trainingMaster != null)
            trainingMaster.setListeners(this.listeners);
    }

    /**
     * This method allows you to specify trainingListeners for this model. Note that for listeners
     * like StatsListener (that have state that will be sent somewhere), consider instead using {@link
     * #setListeners(StatsStorageRouter, Collection)}
     *
     * @param listeners Listeners to set
     */
    public void setListeners(@NonNull TrainingListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the
     * case of any listeners that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, TrainingListener... listeners) {
        setListeners(statsStorage, Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the
     * case of any listeners that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, Collection<? extends TrainingListener> listeners) {
        //Check if we have any RoutingIterationListener instances that need a StatsStorage implementation...
        StatsStorageRouterProvider routerProvider = null;
        if (listeners != null) {
            for (TrainingListener l : listeners) {
                if (l instanceof RoutingIterationListener) {
                    RoutingIterationListener rl = (RoutingIterationListener) l;
                    if (statsStorage == null && rl.getStorageRouter() == null) {
                        log.warn("RoutingIterationListener provided without providing any StatsStorage instance. Iterator may not function without one. Listener: {}",
                                        l);
                    } else if (rl.getStorageRouter() != null && !(rl.getStorageRouter() instanceof Serializable)) {
                        //Spark would throw a (probably cryptic) serialization exception later anyway...
                        throw new IllegalStateException(
                                        "RoutingIterationListener provided with non-serializable storage router "
                                                        + "\nRoutingIterationListener class: " + rl.getClass().getName()
                                                        + "\nStatsStorageRouter class: "
                                                        + rl.getStorageRouter().getClass().getName());
                    }

                    //Need to give workers a router provider...
                    if (routerProvider == null) {
                        routerProvider = new VanillaStatsStorageRouterProvider();
                    }
                }
            }
        }
        this.listeners.clear();
        if (listeners != null) {
            this.listeners.addAll(listeners);
            if (trainingMaster != null)
                trainingMaster.setListeners(statsStorage, this.listeners);
        }
    }
}
