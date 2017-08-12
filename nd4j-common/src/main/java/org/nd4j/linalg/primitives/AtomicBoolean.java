package org.nd4j.linalg.primitives;

public class AtomicBoolean extends java.util.concurrent.atomic.AtomicBoolean {

    public AtomicBoolean(boolean initialValue){
        super(initialValue);
    }

    public AtomicBoolean(){
        this(false);
    }

    @Override
    public boolean equals(Object o){
        if(o instanceof AtomicBoolean){
            return get() == ((AtomicBoolean)o).get();
        } else if(o instanceof Boolean){
            return get() == ((Boolean)o);
        }
        return false;
    }

    @Override
    public int hashCode(){
        return get() ? 1 : 0;
    }

}
