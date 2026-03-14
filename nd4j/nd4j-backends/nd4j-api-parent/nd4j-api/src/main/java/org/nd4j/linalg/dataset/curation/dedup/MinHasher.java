package org.nd4j.linalg.dataset.curation.dedup;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * MinHash implementation for estimating Jaccard similarity between text documents.
 * Uses random hash functions to create compact signatures.
 */
public class MinHasher implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int shingleSize;
    private final int numHashes;
    private final long[] hashA;
    private final long[] hashB;
    private static final long LARGE_PRIME = 4294967311L;

    public MinHasher(int shingleSize, int numHashes, long seed) {
        this.shingleSize = shingleSize;
        this.numHashes = numHashes;
        this.hashA = new long[numHashes];
        this.hashB = new long[numHashes];

        Random rng = new Random(seed);
        for (int i = 0; i < numHashes; i++) {
            hashA[i] = rng.nextLong() & 0x7FFFFFFFL;
            hashB[i] = rng.nextLong() & 0x7FFFFFFFL;
        }
    }

    public MinHasher(int shingleSize, int numHashes) {
        this(shingleSize, numHashes, 42L);
    }

    /**
     * Compute MinHash signature for a text.
     */
    public int[] signature(String text) {
        Set<String> shingles = shingle(text);
        int[] sig = new int[numHashes];

        for (int i = 0; i < numHashes; i++) {
            long minHash = Long.MAX_VALUE;
            for (String shingle : shingles) {
                long h = ((hashA[i] * (shingle.hashCode() & 0xFFFFFFFFL) + hashB[i]) % LARGE_PRIME) & 0x7FFFFFFFL;
                if (h < minHash) {
                    minHash = h;
                }
            }
            sig[i] = (int) minHash;
        }

        return sig;
    }

    /**
     * Estimate Jaccard similarity from two signatures.
     */
    public double estimatedJaccard(int[] sig1, int[] sig2) {
        if (sig1.length != sig2.length) {
            throw new IllegalArgumentException("Signatures must have equal length");
        }
        int matches = 0;
        for (int i = 0; i < sig1.length; i++) {
            if (sig1[i] == sig2[i]) matches++;
        }
        return (double) matches / sig1.length;
    }

    /**
     * Create character-level shingles from text.
     */
    public Set<String> shingle(String text) {
        Set<String> shingles = new HashSet<>();
        String normalized = text.toLowerCase().replaceAll("\\s+", " ").trim();
        for (int i = 0; i <= normalized.length() - shingleSize; i++) {
            shingles.add(normalized.substring(i, i + shingleSize));
        }
        if (shingles.isEmpty() && !normalized.isEmpty()) {
            shingles.add(normalized);
        }
        return shingles;
    }

    public int getShingleSize() { return shingleSize; }
    public int getNumHashes() { return numHashes; }
}
