package org.deeplearning4j.spark.stats;

import java.io.Serializable;

/**
 * Created by Alex on 26/06/2016.
 */
public interface EventStats extends Serializable {

    String getMachineID();

    String getJvmID();

    long getThreadID();

    long getStartTime();

    long getDurationMs();

    /**
     * Get a String representation of the EventStats. This should be a single line delimited representation, suitable
     * for exporting (such as CSV). Should not contain a new-line character at the end of each line
     *
     * @param delimiter Delimiter to use for the data
     * @return String representation of the EventStats object
     */
    String asString(String delimiter);

    /**
     * Get a header line for exporting the EventStats object, for use with {@link #asString(String)}
     *
     * @param delimiter Delimiter to use for the header
     * @return Header line
     */
    String getStringHeader(String delimiter);
}
