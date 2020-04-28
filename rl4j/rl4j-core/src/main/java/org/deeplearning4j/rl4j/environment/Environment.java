package org.deeplearning4j.rl4j.environment;

import java.util.Map;

public interface Environment<ACTION> {
    Schema<ACTION> getSchema();
    Map<String, Object> reset();
    StepResult step(ACTION action);
    boolean isEpisodeFinished();
    void close();
}
