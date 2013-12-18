package com.ccc.sendalyzeit.deeplearning.berkeley;

/**
 * Filters are boolean functions which accept or reject items.
 * @author Dan Klein
 */
public interface Filter<T> {
  boolean accept(T t);
}
