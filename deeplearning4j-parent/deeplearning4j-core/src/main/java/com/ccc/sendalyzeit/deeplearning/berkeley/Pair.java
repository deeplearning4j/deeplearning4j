package com.ccc.sendalyzeit.deeplearning.berkeley;

import java.io.*;
import java.util.*;


/**
 * A generic-typed pair of objects.
 * @author Dan Klein
 */
public class Pair<F, S> implements Serializable {
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
		return new Pair<S, F>(second, first);
	}

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof Pair))
			return false;

		final Pair pair = (Pair) o;

		if (first != null ? !first.equals(pair.first) : pair.first != null)
			return false;
		if (second != null ? !second.equals(pair.second) : pair.second != null)
			return false;

		return true;
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

	// Compares only first values
	public static class FirstComparator<S extends Comparable<? super S>, T>
	implements Comparator<Pair<S, T>> {
		public int compare(Pair<S, T> p1, Pair<S, T> p2) {
			return p1.getFirst().compareTo(p2.getFirst());
		}
	}

	public static class ReverseFirstComparator<S extends Comparable<? super S>, T>
	implements Comparator<Pair<S, T>> {
		public int compare(Pair<S, T> p1, Pair<S, T> p2) {
			return p2.getFirst().compareTo(p1.getFirst());
		}
	}

	// Compares only second values
	public static class SecondComparator<S, T extends Comparable<? super T>>
	implements Comparator<Pair<S, T>> {
		public int compare(Pair<S, T> p1, Pair<S, T> p2) {
			return p1.getSecond().compareTo(p2.getSecond());
		}
	}

	public static class ReverseSecondComparator<S, T extends Comparable<? super T>>
	implements Comparator<Pair<S, T>> {
		public int compare(Pair<S, T> p1, Pair<S, T> p2) {
			return p2.getSecond().compareTo(p1.getSecond());
		}
	}

	public static <S, T> Pair<S, T> newPair(S first, T second) {
		return new Pair<S, T>(first, second);
	}
	// Duplicate method to faccilitate backwards compatibility
	// - aria42
	public static <S, T> Pair<S, T> makePair(S first, T second) {
		return new Pair<S, T>(first, second);
	}

	public static class LexicographicPairComparator<F,S>  implements Comparator<Pair<F,S>> {
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

	public static class DefaultLexicographicPairComparator<F extends Comparable<F>,S extends Comparable<S>>  
	implements Comparator<Pair<F,S>> {

		public int compare(Pair<F, S> o1, Pair<F, S> o2) {
			int firstCompare = o1.getFirst().compareTo(o2.getFirst());
			if (firstCompare != 0) {
				return firstCompare;
			}
			return o2.getSecond().compareTo(o2.getSecond());
		}

	}


}
