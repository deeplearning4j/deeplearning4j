package org.datavec.api.transform.ops;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.datavec.api.writable.Writable;

/**
 * This class converts an {@link IAggregableReduceOp} operating on a Double to one operating
 * on {@link Writable} instances. It's expected this will only work well if that {@link Writable}
 * supports a conversion to Double.
 *
 * Created by huitseeker on 5/14/17.
 */
@AllArgsConstructor
@Data
public class DoubleWritableOp<T> implements IAggregableReduceOp<Writable, T> {

    @Getter
    private IAggregableReduceOp<Double, T> operation;

    @Override
    public <W extends IAggregableReduceOp<Writable, T>> void combine(W accu) {
        if (accu instanceof DoubleWritableOp)
            operation.combine(((DoubleWritableOp) accu).getOperation());
        else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(Writable writable) {
        operation.accept(writable.toDouble());
    }

    @Override
    public T get() {
        return operation.get();
    }
}
