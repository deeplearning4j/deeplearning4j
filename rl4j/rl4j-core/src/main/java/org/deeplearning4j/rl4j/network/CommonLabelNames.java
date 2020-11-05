package org.deeplearning4j.rl4j.network;

public abstract class CommonLabelNames {
    public static final String QValues = "Q";

    public static abstract class ActorCritic {
        public static final String Value = "value"; // critic
        public static final String Policy = "policy"; // actor
    }
}
