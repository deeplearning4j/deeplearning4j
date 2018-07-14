package org.nd4j.api.loader;

import java.io.IOException;
import java.io.InputStream;

/**
 * Used with {@link Loader} to represent the source of an object. The source is a path, usually a URI that can
 * be used to generate an {@link InputStream}
 *
 * @author Alex Black
 */
public interface Source {
    InputStream getInputStream() throws IOException;

    String getPath();
}
