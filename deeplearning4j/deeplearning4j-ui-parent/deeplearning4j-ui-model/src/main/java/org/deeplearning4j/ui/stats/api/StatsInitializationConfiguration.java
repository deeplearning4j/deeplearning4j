package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.ui.stats.StatsListener;

import java.io.Serializable;

/**
 * Configuration interface for static (unchanging) information, to be reported by {@link StatsListener}.
 * This interface allows for software/hardware/model information to be collected (or, not)
 *
 * @author Alex Black
 */
public interface StatsInitializationConfiguration extends Serializable {

    /**
     * Should software configuration information be collected? For example, OS, JVM, and ND4J backend details
     *
     * @return true if software information should be collected; false if not
     */
    boolean collectSoftwareInfo();

    /**
     * Should hardware configuration information be collected? JVM available processors, number of devices, total memory for each device
     *
     * @return true if hardware information should be collected
     */
    boolean collectHardwareInfo();

    /**
     * Should model information be collected? Model class, configuration (JSON), number of layers, number of parameters, etc.
     *
     * @return true if model information should be collected
     */
    boolean collectModelInfo();

}
