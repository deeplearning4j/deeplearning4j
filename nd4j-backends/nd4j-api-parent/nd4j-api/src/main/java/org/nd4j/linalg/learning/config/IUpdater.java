package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * IUpdater interface: used for configuration and instantiation of updaters - both built-in and custom.<br>
 * Note that the actual implementations for updaters are in {@link GradientUpdater}
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
                setterVisibility = JsonAutoDetect.Visibility.NONE)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IUpdater extends Serializable, Cloneable {

    /**
     * Determine the updater state size for the given number of parameters. Usually a integer multiple (0,1 or 2)
     * times the number of parameters in a layer.
     *
     * @param numParams Number of parameters
     * @return Updater state size for the given number of parameters
     */
    long stateSize(long numParams);

    /**
     * Apply the new learning rate and any other schedules
     *
     * @param iteration       Current iteration count
     * @param newLearningRate new learning rate to set for the updater
     */
    void applySchedules(int iteration, double newLearningRate);

    /**
     * Create a new gradient updater
     *
     * @param viewArray           The updater state size view away
     * @param initializeViewArray If true: initialise the updater state
     * @return
     */
    GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray);

    boolean equals(Object updater);

    /**
     * Clone the updater
     */
    IUpdater clone();

}
