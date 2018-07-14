package org.nd4j.api.loader;

import java.io.InputStream;

public interface Source {
    InputStream getInputStream();

    String getPath();
}
