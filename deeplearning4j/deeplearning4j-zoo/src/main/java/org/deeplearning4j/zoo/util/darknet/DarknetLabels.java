package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper class that returns label descriptions for Darknet models trained with ImageNet.
 *
 * @author saudet
 */
public class DarknetLabels extends BaseLabels {

    private boolean shortNames;
    private int numClasses;

    /** Calls {@code this(true)}.
     * Defaults to 1000 clasess
     */
    public DarknetLabels() throws IOException {
        this(true);
    }

    /**
     * @param numClasses Number of classes (usually 1000 or 9000, depending on the model)
     */
    public DarknetLabels(int numClasses) throws IOException {
        this(true, numClasses);
    }

    @Override
    protected URL getURL() {
        try{
            if (shortNames) {
                return DL4JResources.getURL("resources/darknet/imagenet.shortnames.list");
            } else {
                return DL4JResources.getURL("resources/darknet/imagenet.labels.list");
            }
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param shortnames if true, uses "imagenet.shortnames.list", otherwise "imagenet.labels.list".
     */
    public DarknetLabels(boolean shortnames) throws IOException {
        this(shortnames, 1000);
    }

    /**
     * @param shortnames if true, uses "imagenet.shortnames.list", otherwise "imagenet.labels.list".
     * @param numClasses Number of classes (usually 1000 or 9000, depending on the model)
     * @throws IOException
     */
    public DarknetLabels(boolean shortnames, int numClasses) throws IOException {
        this.shortNames = shortnames;
        this.numClasses = numClasses;
        List<String> labels = getLabels(shortnames ? "imagenet.shortnames.list" : "imagenet.labels.list");
        this.labels = new ArrayList<>();
        for( int i=0; i<numClasses; i++ ){
            this.labels.add(labels.get(i));
        }
    }

    @Override
    protected String resourceName() {
        return "darknet";
    }

    @Override
    protected String resourceMD5() {
        if(shortNames){
            return "23d2a102a2de03d1b169c748b7141a20";
        } else {
            return "23ab429a707492324fef60a933551941";
        }
    }
}
