package org.deeplearning4j.models.glove.count;

/**
 * Created by raver on 24.12.2015.
 */
public interface Merger {
    /*
        Storage->Memory merging part
     */
    boolean hasMoreObjects();


    Object nextObject();
    /*
        Memory -> Storage part
     */
    void writeObject();
}
