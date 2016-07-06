package org.canova.api.io.labels;

import org.canova.api.writable.Writable;

import java.net.URI;

/**
 * PathLabelGenerator: interface to infer the label of a file directly from the path of a file<br>
 * Example: /negative/file17.csv -> class "0"; /positive/file116.csv -> class "1" etc.<br>
 * Though note that the output is a writable, hence it need not be numerical.
 * @author Alex Black
 */
public interface PathLabelGenerator {

    Writable getLabelForPath(String path);

    Writable getLabelForPath(URI uri);

}
