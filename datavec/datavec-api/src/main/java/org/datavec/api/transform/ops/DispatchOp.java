package org.datavec.api.transform.ops;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * This class takes many {@link IAggregableReduceOp}, each acting on one field, and each returning several
 * {@link Writable} elements, in the form of a list of {@link Writable}. It produces a reduce operation that
 * distributes a list of {@link Writable} elements to these operations, one per operation.
 *
 *
 * Created by huitseeker on 5/14/17.
 */
@AllArgsConstructor
public class DispatchOp<T, U> implements IAggregableReduceOp<List<T>, List<U>> {


    @Getter
    @NonNull
    private List<IAggregableReduceOp<T, List<U>>> operations;

    @Override
    public <W extends IAggregableReduceOp<List<T>, List<U>>> void combine(W accu) {
        if (accu instanceof DispatchOp) {
            List<IAggregableReduceOp<T, List<U>>> otherOps = ((DispatchOp<T, U>) accu).getOperations();
            if (operations.size() != otherOps.size())
                throw new IllegalArgumentException(
                                "Tried to combine() incompatible " + this.getClass().getName() + " operators: received "
                                                + otherOps.size() + " operations, expected " + operations.size());
            for (int i = 0; i < Math.min(operations.size(), otherOps.size()); i++) {
                operations.get(i).combine(otherOps.get(i));
            }
        } else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(List<T> ts) {
        for (int i = 0; i < Math.min(operations.size(), ts.size()); i++) {
            operations.get(i).accept(ts.get(i));
        }
    }

    @Override
    public List<U> get() {
        List<U> res = new ArrayList<>();
        for (int i = 0; i < operations.size(); i++) {
            res.addAll(operations.get(i).get());
        }
        return res;
    }
}
