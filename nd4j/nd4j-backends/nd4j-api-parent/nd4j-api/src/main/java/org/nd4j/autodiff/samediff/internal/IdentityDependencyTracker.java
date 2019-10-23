package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * Object dependency tracker, using object identity (not object equality) for the Ys (of type T)<br>
 * See {@link AbstractDependencyTracker} for more details
 *
 * @author Alex Black
 */
@Slf4j
public class IdentityDependencyTracker<T, D> extends AbstractDependencyTracker<T,D> {

    @Override
    protected Map<T, ?> newTMap() {
        return new IdentityHashMap<>();
    }

    @Override
    protected Set<T> newTSet() {
        return Collections.newSetFromMap(new IdentityHashMap<T, Boolean>());
    }

    @Override
    protected String toStringT(T t) {
        if(t instanceof INDArray){
            INDArray i = (INDArray)t;
            return System.identityHashCode(t) + " - id=" + i.getId() + ", " + i.shapeInfoToString();
        } else {
            return System.identityHashCode(t) + " - " + t.toString();
        }
    }

    @Override
    protected String toStringD(D d) {
        return d.toString();
    }
}
