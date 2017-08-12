package org.nd4j.linalg.primitives;

import org.nd4j.shade.jackson.annotation.JsonProperty;

public class AtomicDouble extends com.google.common.util.concurrent.AtomicDouble {

    public AtomicDouble(){
        this(0.0);
    }

    public AtomicDouble(@JsonProperty("value") double value){
        super(value);
    }

}
