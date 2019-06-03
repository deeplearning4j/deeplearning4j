package org.deeplearning4j.models.fasttext;

public enum FTLossFunctions {
    HS("hs"),
    NS("ns"),
    SOFTMAX("softmax");

    private final String name;

    FTLossFunctions(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return this.name;
    }
}
