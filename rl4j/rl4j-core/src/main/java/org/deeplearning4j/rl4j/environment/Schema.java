package org.deeplearning4j.rl4j.environment;

import lombok.Value;

@Value
public class Schema<ACTION> {
    private ActionSchema<ACTION> actionSchema;
}
