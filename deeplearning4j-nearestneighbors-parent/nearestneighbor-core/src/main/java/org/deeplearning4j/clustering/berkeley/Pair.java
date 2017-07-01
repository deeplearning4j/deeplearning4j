/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.berkeley;

import java.io.Serializable;
import java.util.Comparator;


/**
 * A generic-typed pair of objects.
 * @author Dan Klein
 */
public class Pair<F, S> implements Serializable, Comparable<Pair<F, S>> {
    static final long serialVersionUID = 42;

    F first;
    S second;

    public F getFirst() {
        return first;
    }

    public S getSecond() {
        return second;
    }

    public void setFirst(F pFirst) {
        first = pFirst;
    }

    public void setSecond(S pSecond) {
        second = pSecond;
    }

    public Pair<S, F> reverse() {
        return new Pair<>(second, first);
    }

    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Pair))
            return false;

        final Pair pair = (Pair) o;

        return !(first != null ? !first.equals(pair.first) : pair.first != null)
                        && !(second != null ? !second.equals(pair.second) : pair.second != null);

    }

    public int hashCode() {
        int result;
        result = (first != null ? first.hashCode() : 0);
        result = 29 * result + (second != null ? second.hashCode() : 0);
        return result;
    }

    public String toString() {
        return "(" + getFirst() + ", " + getSecond() + ")";
    }

    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    /**
     * Compares this object with the specified object for order.  Returns a
     * negative integer, zero, or a positive integer as this object is less
     * than, equal to, or greater than the specified object.
     * <p/>
     * <p>The implementor must ensure <tt>sgn(x.compareTo(y)) ==
     * -sgn(y.compareTo(x))</tt> for all <tt>x</tt> and <tt>y</tt>.  (This
     * implies that <tt>x.compareTo(y)</tt> must throw an exception iff
     * <tt>y.compareTo(x)</tt> throws an exception.)
     * <p/>
     * <p>The implementor must also ensure that the relation is transitive:
     * <tt>(x.compareTo(y)&gt;0 &amp;&amp; y.compareTo(z)&gt;0)</tt> implies
     * <tt>x.compareTo(z)&gt;0</tt>.
     * <p/>
     * <p>Finally, the implementor must ensure that <tt>x.compareTo(y)==0</tt>
     * implies that <tt>sgn(x.compareTo(z)) == sgn(y.compareTo(z))</tt>, for
     * all <tt>z</tt>.
     * <p/>
     * <p>It is strongly recommended, but <i>not</i> strictly required that
     * <tt>(x.compareTo(y)==0) == (x.equals(y))</tt>.  Generally speaking, any
     * class that implements the <tt>Comparable</tt> interface and violates
     * this condition should clearly indicate this fact.  The recommended
     * language is "Note: this class has a natural ordering that is
     * inconsistent with equals."
     * <p/>
     * <p>In the foregoing description, the notation
     * <tt>sgn(</tt><i>expression</i><tt>)</tt> designates the mathematical
     * <i>signum</i> function, which is defined to return one of <tt>-1</tt>,
     * <tt>0</tt>, or <tt>1</tt> according to whether the value of
     * <i>expression</i> is negative, zero or positive.
     *
     * @param o the object to be compared.
     * @return a negative integer, zero, or a positive integer as this object
     * is less than, equal to, or greater than the specified object.
     * @throws NullPointerException if the specified object is null
     * @throws ClassCastException   if the specified object's type prevents it
     *                              from being compared to this object.
     */
    @Override
    public int compareTo(Pair<F, S> o) {
        return new DefaultLexicographicPairComparator().compare(this, o);
    }

    // Compares only first values
    public static class FirstComparator<S extends Comparable<? super S>, T> implements Comparator<Pair<S, T>> {
        public int compare(Pair<S, T> p1, Pair<S, T> p2) {
            return p1.getFirst().compareTo(p2.getFirst());
        }
    }

    public static class ReverseFirstComparator<S extends Comparable<? super S>, T> implements Comparator<Pair<S, T>> {
        public int compare(Pair<S, T> p1, Pair<S, T> p2) {
            return p2.getFirst().compareTo(p1.getFirst());
        }
    }

    // Compares only second values
    public static class SecondComparator<S, T extends Comparable<? super T>> implements Comparator<Pair<S, T>> {
        public int compare(Pair<S, T> p1, Pair<S, T> p2) {
            return p1.getSecond().compareTo(p2.getSecond());
        }
    }

    public static class ReverseSecondComparator<S, T extends Comparable<? super T>> implements Comparator<Pair<S, T>> {
        public int compare(Pair<S, T> p1, Pair<S, T> p2) {
            return p2.getSecond().compareTo(p1.getSecond());
        }
    }

    public static <S, T> Pair<S, T> newPair(S first, T second) {
        return new Pair<>(first, second);
    }

    // Duplicate method to faccilitate backwards compatibility
    // - aria42
    public static <S, T> Pair<S, T> makePair(S first, T second) {
        return new Pair<>(first, second);
    }

    public static class LexicographicPairComparator<F, S> implements Comparator<Pair<F, S>> {
        Comparator<F> firstComparator;
        Comparator<S> secondComparator;

        public int compare(Pair<F, S> pair1, Pair<F, S> pair2) {
            int firstCompare = firstComparator.compare(pair1.getFirst(), pair2.getFirst());
            if (firstCompare != 0)
                return firstCompare;
            return secondComparator.compare(pair1.getSecond(), pair2.getSecond());
        }

        public LexicographicPairComparator(Comparator<F> firstComparator, Comparator<S> secondComparator) {
            this.firstComparator = firstComparator;
            this.secondComparator = secondComparator;
        }
    }

    public static class DefaultLexicographicPairComparator<F extends Comparable<F>, S extends Comparable<S>>
                    implements Comparator<Pair<F, S>> {

        public int compare(Pair<F, S> o1, Pair<F, S> o2) {
            int firstCompare = o1.getFirst().compareTo(o2.getFirst());
            if (firstCompare != 0) {
                return firstCompare;
            }
            return o1.getSecond().compareTo(o2.getSecond());
        }

    }


}
