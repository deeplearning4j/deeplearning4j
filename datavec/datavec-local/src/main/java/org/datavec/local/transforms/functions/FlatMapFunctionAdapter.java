package org.datavec.local.transforms.functions;

import java.io.Serializable;
import java.util.List;

/**
 *
 * A function that returns zero or more output records from each input record.
 *
 * Adapter for function interface in order to
 * freeze interface changes
 */
public interface FlatMapFunctionAdapter<T, R> extends Serializable {
    List<R> call(T t) throws Exception;
}
