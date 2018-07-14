package org.nd4j.api.loader;

import java.io.IOException;
import java.io.Serializable;

/**
 * A simple interface for loading objects from a {@link Source} object
 *
 * @param <T> Type of loaded object
 * @author Alex Black
 */
public interface Loader<T> extends Serializable {

    T load(Source source) throws IOException;
}
