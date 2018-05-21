package org.nd4j.linalg.memory.stash;

/**
 * This interface describes factory/holder for manipulating Stash objects
 *
 * @author raver119@gmail.com
 */
public interface StashManager {

    <T extends Object> boolean checkIfStashExists(T stashId);

    <T extends Object> Stash<T> getStash(T stashId);

    <T extends Object> Stash<T> createStashIfNotExists(T stashId);
}
