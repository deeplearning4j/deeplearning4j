package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;

/**
 * Helper class that returns label descriptions for YOLO models trained with <a href="http://cocodataset.org/">COCO</a>.
 *
 * @author saudet
 */
public class COCOLabels extends BaseLabels {

    public COCOLabels() throws IOException {
        super("coco.names");
    }
}
