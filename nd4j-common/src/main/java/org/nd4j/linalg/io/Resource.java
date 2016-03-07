package org.nd4j.linalg.io;


import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;

/**
 * Resource
 */
public interface Resource extends InputStreamSource {
    /**
     * Whether the resource exists on the classpath
     * @return
     */
    boolean exists();

    /**
     *
     * @return
     */
    boolean isReadable();

    /**
     *
     * @return
     */
    boolean isOpen();

    /**
     *
     * @return
     * @throws IOException
     */
    URL getURL() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    URI getURI() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    File getFile() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    long contentLength() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    long lastModified() throws IOException;

    /**
     *
     * @param var1
     * @return
     * @throws IOException
     */
    Resource createRelative(String var1) throws IOException;

    /**
     *
     * @return
     */
    String getFilename();

    /**
     *
     * @return
     */
    String getDescription();
}
