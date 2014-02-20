package org.deeplearning4j.rng;

import java.io.Serializable;

import org.apache.commons.math3.random.RandomGenerator;

/**
 * Any {@link RandomGenerator} implementation can be thread-safe if it
 * is used through an instance of this class.
 * This is achieved by enclosing calls to the methods of the actual
 * generator inside the overridden {@code synchronized} methods of this
 * class.
 *
 * @since 3.1
 * @version $Id: SynchronizedRandomGenerator.java 1416643 2012-12-03 19:37:14Z tn $
 */
public class SynchronizedRandomGenerator implements RandomGenerator,Serializable {
	/** Object to which all calls will be delegated. */
	private final RandomGenerator wrapped;

	/**
	 * Creates a synchronized wrapper for the given {@code RandomGenerator}
	 * instance.
	 *
	 * @param rng Generator whose methods will be called through
	 * their corresponding overridden synchronized version.
	 * To ensure thread-safety, the wrapped generator <em>must</em>
	 * not be used directly.
	 */
	public SynchronizedRandomGenerator(RandomGenerator rng) {
		wrapped = rng;
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized void setSeed(int seed) {
		wrapped.setSeed(seed);
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized void setSeed(int[] seed) {
		wrapped.setSeed(seed);
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized void setSeed(long seed) {
		wrapped.setSeed(seed);
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized void nextBytes(byte[] bytes) {
		wrapped.nextBytes(bytes);
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized int nextInt() {
		return wrapped.nextInt();
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized int nextInt(int n) {
		return wrapped.nextInt(n);
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized long nextLong() {
		return wrapped.nextLong();
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized boolean nextBoolean() {
		return wrapped.nextBoolean();
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized float nextFloat() {
		return wrapped.nextFloat();
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized double nextDouble() {
		return wrapped.nextDouble();
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized double nextGaussian() {
		return wrapped.nextGaussian();
	}
}
