package org.deeplearning4j.berkeley;

/**
 * Filters are boolean cooccurrences which accept or reject items.
 * @author Dan Klein
 */
public interface Filter<T> {
  boolean accept(T t);
}
