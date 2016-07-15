package org.deeplearning4j.gym.space;

/**
 * @author rubenfiszel on 7/13/16.
 *
 * Generalize every ObservationSpace having for type parameter {@link LowDimensional} Observation
 *
 * @param <O> type of {@link LowDimensional}  Observation
 */
public interface LowDimensionalSpace<O extends LowDimensional> extends ObservationSpace<O> {
}