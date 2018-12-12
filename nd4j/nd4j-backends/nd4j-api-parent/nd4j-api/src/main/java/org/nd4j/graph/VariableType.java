package org.nd4j.graph;

public final class VariableType {
    private VariableType() { }
    public static final byte VARIABLE = 0;
    public static final byte CONSTANT = 1;
    public static final byte ARRAY = 2;
    public static final byte PLACEHOLDER = 3;

    public static final String[] names = { "VARIABLE", "CONSTANT", "ARRAY", "PLACEHOLDER", };

    public static String name(int e) { return names[e]; }
}
