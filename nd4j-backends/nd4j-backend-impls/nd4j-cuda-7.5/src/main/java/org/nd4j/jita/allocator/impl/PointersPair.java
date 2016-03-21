package org.nd4j.jita.allocator.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PointersPair {
    /**
     * this field can be 0, on system without any special devices
     */
    private Pointer devicePointer;

    /**
     * this should always contain long pointer to host memory
     */
    private Pointer hostPointer;

    public PointersPair(long devicePointer, long hostPointer) {
        // TODO: conversion from longs to javacpp pointers
    }
}
