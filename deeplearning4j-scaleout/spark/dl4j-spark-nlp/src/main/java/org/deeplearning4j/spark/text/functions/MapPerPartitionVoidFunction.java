package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.VoidFunction;

import java.util.Iterator;

/**
 * @author jeffreytang
 */
public class MapPerPartitionVoidFunction implements VoidFunction<Iterator<?>> {

    @Override
    public void call(Iterator<?> integerIterator) throws Exception {}
}

