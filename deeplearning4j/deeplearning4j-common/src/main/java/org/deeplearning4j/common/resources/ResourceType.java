package org.deeplearning4j.common.resources;

public enum ResourceType {

    DATASET,
    ZOO_MODEL;

    public String resourceName(){
        switch (this){
            case DATASET:
                return "data";
            case ZOO_MODEL:
                return "models";
            default:
                return this.toString();
        }
    }

}
