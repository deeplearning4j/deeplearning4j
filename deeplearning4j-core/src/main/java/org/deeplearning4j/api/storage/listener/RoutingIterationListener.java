package org.deeplearning4j.api.storage.listener;

import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.io.Serializable;

/**
 * An extension of the {@link TrainingListener} interface for those listeners that pass data off to a
 * {@link org.deeplearning4j.api.storage.StatsStorageRouter} instance.
 * The most common use case here is in distributed training scenarios: each worker has a set of listeners, that have
 * to be serialized and transferred across the network, to some storage mechanism.<br>
 * The StatsStorageRouter implementations themselves may not be serializable, or should be shared between multiple workers,
 * so instead, we use a {@link org.deeplearning4j.api.storage.StatsStorageRouterProvider}
 *
 * @author Alex Black
 */
public interface RoutingIterationListener extends TrainingListener, Cloneable, Serializable {

    void setStorageRouter(StatsStorageRouter router);

    StatsStorageRouter getStorageRouter();

    void setWorkerID(String workerID);

    String getWorkerID();

    void setSessionID(String sessionID);

    String getSessionID();

    RoutingIterationListener clone();

}
