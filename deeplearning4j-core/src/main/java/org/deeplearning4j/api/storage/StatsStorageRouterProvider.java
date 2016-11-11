package org.deeplearning4j.api.storage;

import java.io.Serializable;

/**
 * Simple interface to provide a StatsStorageRouter. Typically used for distributed training such as Spark.
 *
 * @author Alex Black
 */
public interface StatsStorageRouterProvider extends Serializable {

    StatsStorageRouter getRouter();

}
