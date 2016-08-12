package org.deeplearning4j.rl4j.learning.sync;

import java.util.ArrayList;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 */
public interface IExpReplay<A> {

    public ArrayList<Transition<A>> getBatch();

    public void store(Transition<A> transition);

}
