package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;

/**
 * Helper class that returns label descriptions for YOLO models trained with Pascal VOC.
 *
 * @author saudet
 */
public class VOCLabels extends BaseLabels {

    public VOCLabels() throws IOException {
        super("voc.names");
    }
}
