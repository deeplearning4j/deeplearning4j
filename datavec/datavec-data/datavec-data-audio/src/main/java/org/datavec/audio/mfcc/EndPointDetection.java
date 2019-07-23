package org.datavec.audio.mfcc;


/**
 * https://www.academia.edu/4253791/A_New_Silence_Removal_and_Endpoint_Detection_Algorithm_for_Speech_and_Speaker_Recognition_Applications
 * @author wangfeng
 */
public class EndPointDetection {

    private double[] originalSignal;
    private int samplingRate;

    public EndPointDetection(double[] originalSignal, int samplingRate) {
        this.originalSignal = originalSignal;
        this.samplingRate = samplingRate;
    }

    public double[] doEndPointDetection() {
        int originalVoiceSize = originalSignal.length;
        int[] voice = new int[originalVoiceSize];
        double sum = 0;
        double sd = 0.0;
        double u = 0.0;
       // Step 1: Calculate the mean and standard deviation of thefirst 1600 samples of the given utterance
        int firstSamples = originalVoiceSize > 1600 ? 1600: originalVoiceSize;
        for (int i = 0; i < firstSamples; i++) {
            sum += originalSignal[i];
        }
        //calculation the mean
        u = sum / firstSamples;
        sum = 0;
        // calculation the Standard Deviation
        for (int i = 0; i < firstSamples; i++) {
            sum += Math.pow((originalSignal[i] - u), 2);
        }
        sd = Math.sqrt(sum / firstSamples);

        //Step 2:Go from 1st sample to the last sample of thespeech recording.
        // In each sample check whether one-dimensional Mahalanobis distance function i.e. |x-µ|/sd greater than 3 or not
        int thresholdVal = 2;
        //Step 3:Mark the voiced sample as 1 and unvoiced sample as 0.
        //Divide the whole speech signal into 10 msnon-overlapping windows. Now the complete speech isrepresented by only zeros and ones
        //Note that the threshold reject the samples upto 99.7% as per given by equation no. 4 in a Gaussian Distributionthus accepting only the voiced samples.
        for (int i = 0; i < originalVoiceSize; i++) {
            if ((Math.abs(originalSignal[i] - u) / sd) > thresholdVal) {//the sample is to be treated as voiced sample otherwise it isan silence/unvoiced
                voice[i] = 1;
            } else {
                voice[i] = 0;
            }
        }

        //Step 4:Consider there are M no. of zeros and number of ones in a window. If M ≥ N then convert each of onesto zeros and vice versa.
        // This method adopted here keepingin mind that a speech production system consisting of vocal chord, tongue, vocal tract etc.
        // cannot changeabruptly in a short period of time window taken here as 10ms
        int sampleCountPerWindow= 1;// samplingRate / 1000 * 10;
        int windowCount = originalVoiceSize / sampleCountPerWindow;//based on formula 80 samples(10ms windows has 80 samples)
        int usefulVoiceCount = 0;
        for (int i = 0; i < windowCount ; i ++) {
            int voiceZeroCount = 0;
            int voiceOneCount = 0;
            for (int j = i; j < i + sampleCountPerWindow; j ++) {
                if (voice[j] != 0) {
                    voiceOneCount ++;
                } else {
                    voiceZeroCount ++;
                }
            }
            if (voiceZeroCount >= voiceOneCount ) {
                for (int j = i; j < i + sampleCountPerWindow; j ++) {
                    voice[j] = 0;
                }
            } else {
                usefulVoiceCount += sampleCountPerWindow;
            }
        }
        //Step 5: Collect the voiced part only according to thelabeled ‘1’ samples from the windowed array and dump itin a new array.
        // Retrieve the voiced part of the originalspeech signal from labeled 1 samples
        double[] usefulVoice = new double[usefulVoiceCount];
        int index = 0;
        for (int i = 0; i < originalVoiceSize; i++) {
            boolean usefulVoiceBol = false;
            for (int j = i; j < i + sampleCountPerWindow; j ++) {
                if (voice[j] != 0) {
                    usefulVoiceBol = true;
                }
            }
            for (int j = i; j < i + sampleCountPerWindow && usefulVoiceBol; j ++) {
                usefulVoice[index++] = originalSignal[j];
            }
        }
        // end
        return usefulVoice;
    }
}
