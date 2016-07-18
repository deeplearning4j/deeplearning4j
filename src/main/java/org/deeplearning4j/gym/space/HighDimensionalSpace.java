package org.deeplearning4j.gym.space;


/**
 * @author rubenfiszel on 7/13/16.
 *
 * Generalize every ObservationSpace having for type parameter {@link HighDimensional} Observation
 *
 * @param <O> type of {@link HighDimensional}  Observation
 */
public interface HighDimensionalSpace<O extends HighDimensional> extends ObservationSpace<O>  {
}
