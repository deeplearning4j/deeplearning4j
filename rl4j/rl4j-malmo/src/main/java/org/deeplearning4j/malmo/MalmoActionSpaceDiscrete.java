package org.deeplearning4j.malmo;

/**
 * Action space that allows for a fixed set of specific Malmo actions
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoActionSpaceDiscrete extends MalmoActionSpace {
    /**
     * Construct an actions space from an array of action strings
     * @param actions Array of action strings
     */
    public MalmoActionSpaceDiscrete(String... actions) {
        super(actions.length);
        this.actions = actions;
    }
}
