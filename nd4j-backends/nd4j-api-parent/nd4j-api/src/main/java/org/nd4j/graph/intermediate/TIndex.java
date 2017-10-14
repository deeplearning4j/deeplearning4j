package org.nd4j.graph.intermediate;

import org.nd4j.linalg.primitives.ImmutablePair;

/**
 * This class is used as index for TNodes
 *
 * @author raver119@gmail.com
 */
public class TIndex {
    protected ImmutablePair<Integer, Integer> pair;

    protected TIndex() {

    }

    protected TIndex(int node, int index) {
        pair = ImmutablePair.makePair(node, index);
    }

    public static TIndex makeOf(int node, int index) {
        return new TIndex(node, index);
    }

    public static TIndex makeOf(int node) {
        return makeOf(node, 0);
    }

    public int getNode(){
        return pair.getFirst();
    }

    public int getIndex() {
        return pair.getSecond();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TIndex tIndex = (TIndex) o;

        return pair.equals(tIndex.pair);
    }

    @Override
    public int hashCode() {
        return pair.hashCode();
    }

    @Override
    public String toString() {
        return "TIndex{" + pair.getFirst() + ":" + pair.getSecond() + '}';
    }
}
