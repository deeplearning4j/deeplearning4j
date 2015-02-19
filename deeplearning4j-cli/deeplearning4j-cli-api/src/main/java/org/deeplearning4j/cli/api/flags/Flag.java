package org.deeplearning4j.cli.api.flags;

/**
 * @author sonali
 */
public interface Flag {
    /**
     * Return a value based on a string
     * @param <E>
     * @param value the value to instantiate from
     * @return the value
     */
    <E> E value(String value) throws Exception;

}
