package org.deeplearning4j.util;

//@FunctionalInterface
public interface Consumer<T> {
    void accept(T t);
}