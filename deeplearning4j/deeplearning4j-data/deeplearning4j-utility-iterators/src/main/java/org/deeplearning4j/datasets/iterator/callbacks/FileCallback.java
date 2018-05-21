package org.deeplearning4j.datasets.iterator.callbacks;

import java.io.File;

/**
 * @author raver119@gmail.com
 */
public interface FileCallback {

    <T> T call(File file);
}
