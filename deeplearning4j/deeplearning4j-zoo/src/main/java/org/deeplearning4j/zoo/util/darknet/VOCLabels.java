package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Helper class that returns label descriptions for YOLO models trained with Pascal VOC.
 *
 * @author saudet
 */
public class VOCLabels extends BaseLabels {

    public VOCLabels() throws IOException {
        super("voc.names");
    }

    @Override
    protected URL getURL() {
        try {
            return DL4JResources.getURL("resources/darknet/voc.names");
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
        return "";
    }
}
