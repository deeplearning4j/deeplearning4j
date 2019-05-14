package org.nd4j.resources;

import java.io.File;
import java.io.InputStream;

public interface Resolver {

    /**
     * Priority of this resolver. 0 is highest priority (check first), larger values are lower priority (check last)
     */
    int priority();

    boolean exists(String resourcePath);

    File asFile(String resourcePath);

    InputStream asStream(String resourcePath);

    boolean hasLocalCache();

    File localCacheRoot();

    //TODO maybe a method to list configuration options or properties?

}
