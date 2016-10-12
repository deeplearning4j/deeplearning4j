package org.deeplearning4j.api.storage;

import java.io.Serializable;

/**
 * Created by Alex on 11/10/2016.
 */
public interface StatsStorageRouterProvider extends Serializable {

    StatsStorageRouter getRouter();

}
