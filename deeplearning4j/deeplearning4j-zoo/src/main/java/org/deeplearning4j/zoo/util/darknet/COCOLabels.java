package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Helper class that returns label descriptions for YOLO models trained with <a href="http://cocodataset.org/">COCO</a>.
 *
 * @author saudet
 */
public class COCOLabels extends BaseLabels {

    public COCOLabels() throws IOException {
        super("coco.names");
    }

    @Override
    protected URL getURL() {
        try {
            return DL4JResources.getURL("resources/darknet/coco.names");
        } catch (MalformedURLException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    protected String resourceName() {
        return "darknet";
    }

    @Override
    protected String resourceMD5() {
        return "4caf6834300c8b2ff19964b36e54d637";
    }
}
