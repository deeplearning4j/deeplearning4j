package org.nd4j.api.loader;

import java.io.Serializable;

/**
 * A factory interface for getting {@link Source} objects given a String path
 * @author Alex Black
 */
public interface SourceFactory extends Serializable {
    Source getSource(String path);
}
