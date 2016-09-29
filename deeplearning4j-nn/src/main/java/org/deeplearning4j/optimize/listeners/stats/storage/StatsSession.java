package org.deeplearning4j.optimize.listeners.stats.storage;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Created by Alex on 29/09/2016.
 */
@AllArgsConstructor @Data
public class StatsSession implements Serializable {

    private final String sessionID;
    private final long initTime;

}
