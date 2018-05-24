package org.deeplearning4j.common.resources;

public enum ResourceType {

    DATASET,
    ZOO_MODEL,
    RESOURCE;

    public String resourceName(){
        switch (this){
            case DATASET:
                return "data";
            case ZOO_MODEL:
                return "models";
            case RESOURCE:
                return "resources";
            default:
                return this.toString();
        }
    }

}
