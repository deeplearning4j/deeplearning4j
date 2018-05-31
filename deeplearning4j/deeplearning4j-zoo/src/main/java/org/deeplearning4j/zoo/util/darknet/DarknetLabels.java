package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Helper class that returns label descriptions for Darknet models trained with ImageNet.
 *
 * @author saudet
 */
public class DarknetLabels extends BaseLabels {

    private boolean shortNames;

    /** Calls {@code this(true)}. */
    public DarknetLabels() throws IOException {
        this(true);
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
        this.shortNames = shortnames;
        this.labels = getLabels(shortnames ? "imagenet.shortnames.list" : "imagenet.labels.list");
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
