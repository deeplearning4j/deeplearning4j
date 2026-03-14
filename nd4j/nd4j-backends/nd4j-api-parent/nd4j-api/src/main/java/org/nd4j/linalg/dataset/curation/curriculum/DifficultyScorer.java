package org.nd4j.linalg.dataset.curation.curriculum;

/**
 * Interface for scoring the difficulty of a training example.
 * Higher scores indicate harder examples.
 *
 * @param <T> the type of training example
 */
public interface DifficultyScorer<T> {

    /**
     * Score the difficulty of an example. Higher values = harder.
     *
     * @param example the training example
     * @return difficulty score (higher = harder)
     */
    double score(T example);
}
