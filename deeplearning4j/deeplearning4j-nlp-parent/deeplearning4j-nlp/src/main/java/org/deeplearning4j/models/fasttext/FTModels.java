package org.deeplearning4j.models.fasttext;

public enum FTModels {
    CBOW("cbow"),
    SG("sg"),
    SUP("sup");

    private final String name;

    FTModels(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return this.name;
    }
}
