package org.deeplearning4j.nn.api.activations;

import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.concurrent.atomic.AtomicLong;

public class ActivationsFactory {

    private static final int DEFAULT_CACHE_SIZE = 256;
    private static final ActivationsFactory INSTANCE = new ActivationsFactory();
    public static ActivationsFactory getInstance(){
        return INSTANCE;
    }


    private ActivationsSingle[] singles;
    private ActivationsPair[] pairs;
    private ActivationsTriple[] triples;
    private Activations[] tuples;

    private AtomicLong singlesStart = new AtomicLong(-1);
    private AtomicLong singlesEnd = new AtomicLong(-1);
    private AtomicLong pairsStart = new AtomicLong(-1);
    private AtomicLong pairsEnd = new AtomicLong(-1);
    private AtomicLong triplesStart = new AtomicLong(-1);
    private AtomicLong triplesEnd = new AtomicLong(-1);

    private ActivationsFactory(){
        this(DEFAULT_CACHE_SIZE);
    }

    private ActivationsFactory(int cacheSize){
        singles = new ActivationsSingle[cacheSize];
        pairs = new ActivationsPair[cacheSize];
        triples = new ActivationsTriple[cacheSize];
        tuples = new Activations[cacheSize];
    }

    public void release(Activations activations){
        if(activations == null) return;
        activations.clear();

        if(activations instanceof ActivationsSingle){
            set(activations, singles, singlesStart, singlesEnd);
        } else if(activations instanceof ActivationsPair){
            set(activations, pairs, pairsStart, pairsEnd);
        } else if(activations instanceof ActivationsTriple){
            set(activations, triples, triplesStart, triplesEnd);
        }

        //TODO: use an IdentityHashMap (Set) to avoid multiple releases of the same activations value
        //This will require a hashcode() implementation that allows for O(1) comparisons/checks - like a simple
        // counter...
    }

    public void release(Collection<? extends Activations> activations){
        //TODO duplicates check
        for(Activations a : activations){
            release(a);
        }
    }

    public Activations create(int size){
        switch (size){
            case 1:
                return new ActivationsSingle(null, null, null);
            case 2:
                return new ActivationsPair(null, null, null, null, null, null);
            case 3:
                return new ActivationsTriple(null, null, null,
                        null, null, null,
                        null, null, null);
            default:
                return new ActivationsTuple(new INDArray[size], null, null);
        }
    }

    public Activations create(INDArray activations){
        return create(activations, null, null);
    }

    public Activations create(INDArray activations, INDArray mask){
        return create(activations, mask, (mask == null ? null : MaskState.Active ));
    }

    public Activations create(INDArray activations, INDArray mask, MaskState maskState) {
        //First: determine if any cached value is available
        ActivationsSingle single = get(singles, singlesStart, singlesEnd);
        if(single != null){
            return setValues(single, 0, activations, mask, maskState);
        }
        return new ActivationsSingle(activations, mask, maskState);
    }

    public Activations createPair(INDArray activations1, INDArray activations2, INDArray mask1, INDArray mask2,
                                  MaskState maskState1, MaskState maskState2) {
        //First: determine if any cached value is available
        ActivationsPair pair = get(pairs, pairsStart, pairsEnd);
        if(pair != null){
            setValues(pair, 0, activations1, mask1, maskState1);
            return setValues(pair, 1, activations2, mask2, maskState2);
        }
        return new ActivationsPair(activations1, activations2, mask1, mask2, maskState1, maskState2);
    }

    public Activations createTriple(INDArray activations1, INDArray activations2, INDArray activations3,
                                    INDArray mask1, INDArray mask2, INDArray mask3,
                                    MaskState maskState1, MaskState maskState2, MaskState maskState3) {
        //First: determine if any cached value is available
        ActivationsTriple triple = get(triples, triplesStart, triplesEnd);
        if(triple != null){
            setValues(triple, 0, activations1, mask1, maskState1);
            setValues(triple, 1, activations2, mask2, maskState2);
            setValues(triple, 2, activations3, mask3, maskState3);
        }
        return new ActivationsTriple(activations1, activations2, activations3,
                mask1, mask2, mask3,
                maskState1, maskState2, maskState3);
    }

    public Activations create(INDArray[] activations, INDArray[] maskArrays, MaskState[] maskStates){
        return new ActivationsTuple(activations, maskArrays, maskStates);
    }

    private Activations setValues(Activations to, int idx, INDArray a, INDArray m, MaskState ms){
        to.set(idx, a);
        to.setMask(idx, m);
        to.setMaskState(idx, ms);
        return to;
    }

    public static <T> T get(T[] from, AtomicLong start, AtomicLong end){
        long s = start.get();
        long e = end.get();
        if(e < 0 || e < s){
            //No available cached values
            return null;
        }
        //There *may* be cached values available to use
        while(e <= s){
            T temp = from[(int)(e%from.length)];
            boolean got = end.compareAndSet(e, e+1);
            if(got){
                return temp;
            }
            s = start.get();
            e = end.get();
        }
        return null;    //Failed (other thread got all available values first)
    }

    public static <T> void set(T value, T[] to, AtomicLong start, AtomicLong end){
        long s = start.get();
        long e = end.get();
        if(s-e >= to.length-1){
            //No space left, no-op
            return;
        }
        //There *may* be space available to put this value for reuse
        while(s-e <= to.length-1){
            boolean got = start.compareAndSet(s, s+1);
            if(got){
                to[(int)((s+1)%to.length)] = value;
                return;
            }
            s = start.get();
            e = end.get();
        }
        //Failed (other thread used available slots first)
    }

}
