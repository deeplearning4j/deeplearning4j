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

    @Override
    public boolean equals(Object o){
        //NOTE: com.google.common.util.concurrent.AtomicDouble extends Number, hence this class extends number
        if(o instanceof Number){
            return get() == ((Number)o).doubleValue();
        }
        return false;
    }

    @Override
    public int hashCode(){
        //return Double.hashCode(get());    //Java 8+
        return Double.valueOf(get()).hashCode();
    }
}
