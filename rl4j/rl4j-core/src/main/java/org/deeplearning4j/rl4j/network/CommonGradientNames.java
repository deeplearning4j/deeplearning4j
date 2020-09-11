package org.deeplearning4j.rl4j.network;

public abstract class CommonGradientNames {
    public static final String QValues = "Q";

    public static abstract class ActorCritic {
        public static final String Value = "value"; // critic
        public static final String Policy = "policy"; // actor
        public static final String Combined = "combined"; // combined actor-critic gradients
    }

}
