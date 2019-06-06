package org.nd4j.graph;

public final class UIEventSubtype {
    private UIEventSubtype() { }
    public static final byte NONE = 0;
    public static final byte EVALUATION = 1;
    public static final byte LOSS = 2;
    public static final byte LEARNING_RATE = 3;
    public static final byte TUNING_METRIC = 4;
    public static final byte PERFORMANCE = 5;
    public static final byte PROFILING = 6;
    public static final byte FEATURE_LABEL = 7;
    public static final byte PREDICTION = 8;
    public static final byte USER_CUSTOM = 9;

    public static final String[] names = { "NONE", "EVALUATION", "LOSS", "LEARNING_RATE", "TUNING_METRIC", "PERFORMANCE", "PROFILING", "FEATURE_LABEL", "PREDICTION", "USER_CUSTOM", };

    public static String name(int e) { return names[e]; }
}

