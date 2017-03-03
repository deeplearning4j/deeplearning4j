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

package org.datavec.audio.properties;

public class FingerprintProperties {

    protected static FingerprintProperties instance = null;

    private int numRobustPointsPerFrame = 4; // number of points in each frame, i.e. top 4 intensities in fingerprint
    private int sampleSizePerFrame = 2048; // number of audio samples in a frame, it is suggested to be the FFT Size
    private int overlapFactor = 4; // 8 means each move 1/8 nSample length. 1 means no overlap, better 1,2,4,8 ...	32
    private int numFilterBanks = 4;

    private int upperBoundedFrequency = 1500; // low pass
    private int lowerBoundedFrequency = 400; // high pass
    private int fps = 5; // in order to have 5fps with 2048 sampleSizePerFrame, wave's sample rate need to be 10240 (sampleSizePerFrame*fps)
    private int sampleRate = sampleSizePerFrame * fps; // the audio's sample rate needed to resample to this in order to fit the sampleSizePerFrame and fps
    private int numFramesInOneSecond = overlapFactor * fps; // since the overlap factor affects the actual number of fps, so this value is used to evaluate how many frames in one second eventually  

    private int refMaxActivePairs = 1; // max. active pairs per anchor point for reference songs
    private int sampleMaxActivePairs = 10; // max. active pairs per anchor point for sample clip
    private int numAnchorPointsPerInterval = 10;
    private int anchorPointsIntervalLength = 4; // in frames (5fps,4 overlap per second)
    private int maxTargetZoneDistance = 4; // in frame (5fps,4 overlap per second)

    private int numFrequencyUnits = (upperBoundedFrequency - lowerBoundedFrequency + 1) / fps + 1; // num frequency units

    public static FingerprintProperties getInstance() {
        if (instance == null) {
            synchronized (FingerprintProperties.class) {
                if (instance == null) {
                    instance = new FingerprintProperties();
                }
            }
        }
        return instance;
    }

    public int getNumRobustPointsPerFrame() {
        return numRobustPointsPerFrame;
    }

    public int getSampleSizePerFrame() {
        return sampleSizePerFrame;
    }

    public int getOverlapFactor() {
        return overlapFactor;
    }

    public int getNumFilterBanks() {
        return numFilterBanks;
    }

    public int getUpperBoundedFrequency() {
        return upperBoundedFrequency;
    }

    public int getLowerBoundedFrequency() {
        return lowerBoundedFrequency;
    }

    public int getFps() {
        return fps;
    }

    public int getRefMaxActivePairs() {
        return refMaxActivePairs;
    }

    public int getSampleMaxActivePairs() {
        return sampleMaxActivePairs;
    }

    public int getNumAnchorPointsPerInterval() {
        return numAnchorPointsPerInterval;
    }

    public int getAnchorPointsIntervalLength() {
        return anchorPointsIntervalLength;
    }

    public int getMaxTargetZoneDistance() {
        return maxTargetZoneDistance;
    }

    public int getNumFrequencyUnits() {
        return numFrequencyUnits;
    }

    public int getMaxPossiblePairHashcode() {
        return maxTargetZoneDistance * numFrequencyUnits * numFrequencyUnits + numFrequencyUnits * numFrequencyUnits
                        + numFrequencyUnits;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public int getNumFramesInOneSecond() {
        return numFramesInOneSecond;
    }
}
