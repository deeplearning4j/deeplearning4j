package org.nd4j.graph;

public final class UIEventSubtype {
    private UIEventSubtype() { }
    public static final byte NONE = 0;
    public static final byte EVALUATION = 1;
    public static final byte LOSS = 2;
    public static final byte TUNING_METRIC = 3;
    public static final byte PERFORMANCE = 4;
    public static final byte PROFILING = 5;
    public static final byte FEATURE_LABEL = 6;
    public static final byte PREDICTION = 7;
    public static final byte USER_CUSTOM = 8;

    public static final String[] names = { "NONE", "EVALUATION", "LOSS", "TUNING_METRIC", "PERFORMANCE", "PROFILING", "FEATURE_LABEL", "PREDICTION", "USER_CUSTOM", };

    public static String name(int e) { return names[e]; }
}

