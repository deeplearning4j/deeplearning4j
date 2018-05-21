package org.datavec.api.transform.ops;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.datavec.api.writable.Writable;

/**
 * This class converts an {@link IAggregableReduceOp} operating on a Integer to one operating
 * on {@link Writable} instances. It's expected this will only work if that {@link Writable}
 * supports a conversion to Integer.
 *
 * Created by huitseeker on 5/14/17.
 */
@AllArgsConstructor
@Data
public class IntWritableOp<T> implements IAggregableReduceOp<Writable, T> {

    @Getter
    private IAggregableReduceOp<Integer, T> operation;

    @Override
    public <W extends IAggregableReduceOp<Writable, T>> void combine(W accu) {
        if (accu instanceof IntWritableOp)
            operation.combine(((IntWritableOp) accu).getOperation());
        else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(Writable writable) {
        operation.accept(writable.toInt());
    }

    @Override
    public T get() {
        return operation.get();
    }
}
