package org.datavec.api.transform.ops;

import org.nd4j.linalg.function.Consumer;
import org.nd4j.linalg.function.Supplier;

import java.io.Serializable;

/**
 * Created by huitseeker on 4/28/17.
 */
public interface IAggregableReduceOp<T, V> extends Consumer<T>, Supplier<V>, Serializable {

    <W extends IAggregableReduceOp<T, V>> void combine(W accu);

}
