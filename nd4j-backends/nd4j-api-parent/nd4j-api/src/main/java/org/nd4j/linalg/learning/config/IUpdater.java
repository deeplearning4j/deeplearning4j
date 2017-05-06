package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Created by Alex on 06/05/2017.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
        setterVisibility = JsonAutoDetect.Visibility.NONE)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IUpdater extends Serializable, Cloneable {

    long stateSize(long numParams);

    void applySchedules(int iteration, double newLearningRate);

    GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray);

    boolean equals(Object updater);

    IUpdater clone();

}
