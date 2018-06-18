package org.deeplearning4j.nn.layers;

import java.util.Map;

public interface LayerHelper {

    /**
     * Return the currently allocated memory for the helper.<br>
     * (a) Excludes: any shared memory used by multiple helpers/layers<br>
     * (b) Excludes any temporary memory
     * (c) Includes all memory that persists for longer than the helper method<br>
     * This is mainly used for debugging and reporting purposes. Returns a map:<br>
     * Key: The name of the type of memory<br>
     * Value: The amount of memory<br>
     *
     * @return Map of memory, may be null if none is used.
     */
    Map<String,Long> helperMemoryUse();

}
