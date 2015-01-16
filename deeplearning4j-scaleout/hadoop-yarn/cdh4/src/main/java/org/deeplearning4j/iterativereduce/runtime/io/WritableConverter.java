package org.deeplearning4j.iterativereduce.runtime.io;


import org.apache.hadoop.io.Writable;

/**
 * Convert a writable to another writable (used for say: transitioning dates or categorical to numbers)
 *
 * @author Adam Gibson
 */
public interface WritableConverter {


    /**
     * Convert a writable to another kind of writable
     * @param writable the writable to convert
     * @return the converted writable
     */
    Writable convert(Writable writable);

}
