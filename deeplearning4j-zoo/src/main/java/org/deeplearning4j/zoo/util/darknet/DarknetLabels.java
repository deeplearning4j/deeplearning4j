package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;

/**
 * Helper class that returns label descriptions for Darknet models trained with ImageNet.
 *
 * @author saudet
 */
public class DarknetLabels extends BaseLabels {

    /** Calls {@code this(true)}. */
    public DarknetLabels() throws IOException {
        this(true);
    }

    /**
     * @param shortnames if true, uses "imagenet.shortnames.list", otherwise "imagenet.labels.list".
     */
    public DarknetLabels(boolean shortnames) throws IOException {
        super(shortnames ? "imagenet.shortnames.list" : "imagenet.labels.list");
    }
}
