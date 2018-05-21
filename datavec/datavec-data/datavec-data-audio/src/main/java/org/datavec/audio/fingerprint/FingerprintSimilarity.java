/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */
package org.datavec.audio.fingerprint;


import org.datavec.audio.properties.FingerprintProperties;

/**
 * A class for fingerprint's similarity
 * 
 * @author jacquet
 *
 */
public class FingerprintSimilarity {

    private FingerprintProperties fingerprintProperties = FingerprintProperties.getInstance();
    private int mostSimilarFramePosition;
    private float score;
    private float similarity;

    /**
     * Constructor
     */
    public FingerprintSimilarity() {
        mostSimilarFramePosition = Integer.MIN_VALUE;
        score = -1;
        similarity = -1;
    }

    /**
     * Get the most similar position in terms of frame number
     * 
     * @return most similar frame position
     */
    public int getMostSimilarFramePosition() {
        return mostSimilarFramePosition;
    }

    /**
     * Set the most similar position in terms of frame number
     * 
     * @param mostSimilarFramePosition
     */
    public void setMostSimilarFramePosition(int mostSimilarFramePosition) {
        this.mostSimilarFramePosition = mostSimilarFramePosition;
    }

    /**
     * Get the similarity of the fingerprints
     * similarity from 0~1, which 0 means no similar feature is found and 1 means in average there is at least one match in every frame
     * 
     * @return fingerprints similarity
     */
    public float getSimilarity() {
        return similarity;
    }

    /**
     * Set the similarity of the fingerprints
     * 
     * @param similarity similarity
     */
    public void setSimilarity(float similarity) {
        this.similarity = similarity;
    }

    /**
     * Get the similarity score of the fingerprints
     * Number of features found in the fingerprints per frame 
     * 
     * @return fingerprints similarity score
     */
    public float getScore() {
        return score;
    }

    /**
     * Set the similarity score of the fingerprints
     * 
     * @param score
     */
    public void setScore(float score) {
        this.score = score;
    }

    /**
     * Get the most similar position in terms of time in second
     * 
     * @return most similar starting time
     */
    public float getsetMostSimilarTimePosition() {
        return (float) mostSimilarFramePosition / fingerprintProperties.getNumRobustPointsPerFrame()
                        / fingerprintProperties.getFps();
    }
}
