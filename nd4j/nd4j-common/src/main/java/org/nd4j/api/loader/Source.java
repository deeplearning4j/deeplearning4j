package org.nd4j.api.loader;

import java.io.IOException;
import java.io.InputStream;

public interface Source {
    InputStream getInputStream() throws IOException;

    String getPath();
}
