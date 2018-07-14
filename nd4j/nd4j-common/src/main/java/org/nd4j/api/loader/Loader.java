package org.nd4j.api.loader;

import java.io.IOException;

public interface Loader<T> {

    T load(Source source) throws IOException;
}
