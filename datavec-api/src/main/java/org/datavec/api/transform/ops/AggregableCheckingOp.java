package org.datavec.api.transform.ops;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.Writable;

/**
 * A variant of {@link IAggregableReduceOp} exercised on a {@link Writable} that takes schema metadata
 * in its constructor, and checks the input {@link Writable} against the schema before accepting it.
 *
 * Created by huitseeker on 5/8/17.
 */
@AllArgsConstructor
@Data
public class AggregableCheckingOp<T> implements IAggregableReduceOp<Writable, T> {

    @Getter
    private IAggregableReduceOp<Writable, T> operation;
    @Getter
    private ColumnMetaData metaData;

    @Override
    public <W extends IAggregableReduceOp<Writable, T>> void combine(W accu) {
        if (accu instanceof AggregableCheckingOp) {
            AggregableCheckingOp<T> accumulator = (AggregableCheckingOp) accu;
            if (metaData.getColumnType() != accumulator.getMetaData().getColumnType())
                throw new IllegalArgumentException(
                                "Invalid merge with operation on " + accumulator.getMetaData().getName() + " of type "
                                                + accumulator.getMetaData().getColumnType() + " expected "
                                                + metaData.getName() + " of type " + metaData.getColumnType());
            else
                operation.combine(accumulator);
        } else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(Writable writable) {
        if (metaData.isValid(writable))
            operation.accept(writable);
    }

    @Override
    public T get() {
        return operation.get();
    }
}
