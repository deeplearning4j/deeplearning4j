package org.nd4j.api.loader;

import java.io.IOException;
import java.io.Serializable;

public interface Loader<T> extends Serializable {

    T load(Source source) throws IOException;
}
