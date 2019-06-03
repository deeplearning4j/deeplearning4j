package org.deeplearning4j.models.fasttext;

public enum FTOptions {
    INPUT_FILE("-input"),
    OUTPUT_FILE("-output");

    private final String name;

    FTOptions(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return this.name;
    }
}
