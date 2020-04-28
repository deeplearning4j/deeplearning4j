package org.deeplearning4j.rl4j.environment;

import lombok.Value;

@Value
public class ActionSchema<ACTION> {
    private ACTION noOp;
    //FIXME ACTION randomAction();
}
