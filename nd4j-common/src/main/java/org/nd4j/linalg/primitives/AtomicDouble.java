package org.nd4j.linalg.primitives;

import org.nd4j.linalg.primitives.serde.JsonDeserializerAtomicDouble;
import org.nd4j.linalg.primitives.serde.JsonSerializerAtomicDouble;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

@JsonSerialize(using = JsonSerializerAtomicDouble.class)
@JsonDeserialize(using = JsonDeserializerAtomicDouble.class)
public class AtomicDouble extends com.google.common.util.concurrent.AtomicDouble {

    public AtomicDouble(){
        this(0.0);
    }

    public AtomicDouble(@JsonProperty("value") double value){
        super(value);
    }

    public AtomicDouble(float value){
        this((double)value);
    }

}
