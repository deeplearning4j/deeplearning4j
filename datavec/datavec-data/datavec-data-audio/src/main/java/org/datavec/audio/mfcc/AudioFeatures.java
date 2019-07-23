package org.datavec.audio.mfcc;


import org.datavec.audio.dsp.FastFourierTransform;
import org.datavec.audio.mfcc.EndPointDetection;

/**
 * https://arxiv.org/ftp/arxiv/papers/1003/1003.4083.pdf
 * https://pdfs.semanticscholar.org/80b6/d95980872807e6fe776a277d54ba637bfbe3.pdf
 * @author wangfeng
 */
public class AudioFeatures {
    private double sampleRate = 0;
    private double framerate = 0;
    private double samplePercentageFrame = 0;

    //0: endpoint-detection
    public double[] endPointDetection(double[] amplitudes, int samplingRate) {
        EndPointDetection epd = new EndPointDetection(amplitudes, samplingRate);
        amplitudes = epd.doEndPointDetection();
        return amplitudes;
    }

    //1: pre-emphasis
    //This step processes the passing of signal through a filter which emphasizes higher frequencies.
    // This process will increase the energy of signal at higher frequency
    //signal[i]=signal[i+1]-0.95*signal[i]
    public double[] preEmphasis(double[] amplitudes) {
        double preEmphasis = 0.95;
        double[] resultAmplitudes = new double[amplitudes.length];
        for (int i = 1; i < amplitudes.length; i++) {
            resultAmplitudes[i] = amplitudes[i] - preEmphasis * amplitudes[i - 1];
        }
        return resultAmplitudes;
    }

    //2: Framing
    //The process of segmenting the speech samples obtained from analog to digital conversion (ADC) into
    // a small frame with the length within the range of 20 to 40 msec. The voice signal is divided into frames  of N samples.
    // Adjacent frames are being separated by M (M<N).  Typical values used are M = 100 and N= 256
    //frameMove must less than samplePerFrame
    public double[][] splitFrame(double[] amplitudes, int samplingRate) {
        int samplePerFrame = samplingRate / 1000 * 40; //N
        int adjacentSamplePerFrame = samplingRate / 1000 * 20; //M

        int framesCount = (int) (amplitudes.length - samplePerFrame) / adjacentSamplePerFrame + 1;
        int pad_signal_length = framesCount * adjacentSamplePerFrame + samplePerFrame;
        int balanceVal = pad_signal_length - amplitudes.length;
        double[] balanceVals = new double[balanceVal];
        double[] newAmplitudes = new double[pad_signal_length];
        System.arraycopy(amplitudes, 0, newAmplitudes, 0, amplitudes.length);
        System.arraycopy(balanceVals, 0, newAmplitudes, amplitudes.length, balanceVals.length);
        double[][] splitFrames = new double[framesCount][samplePerFrame];
        for (int i = 0; i < framesCount; i++) {
            int startIndex = i * adjacentSamplePerFrame;
            for (int j = 0; j < samplePerFrame; j++) {
                double val = newAmplitudes[startIndex + j];
                splitFrames[i][j] = val;
            }
        }
        return splitFrames;
    }

    //3:Hamming windowing
    //Hamming window is used as window shape by considering the next
    //block in feature extraction processing chain and integrates all the closest frequency lines
    public double[][] doWindows(double[][] splitFrames) {
        // get number of frame
        int FramesCount = splitFrames.length;
        // get number of samples in each frame
        int samplePerFrame = splitFrames[0].length;
        //get hamming Windows
        double[] hammingWindow = new double[samplePerFrame];
        double R = 2 * Math.PI;
        for (int i = 0; i < samplePerFrame; i++) {
            hammingWindow[i] = 0.54 - 0.46 * Math.cos(R * i / (samplePerFrame - 1));
        }
        //let frame signal is multiplied by the Hamming window to smooth the signal,to reduce the intensity of the side lobes after Fourier transform to achieve a better frequency spectrum
        for (int i = 0; i < FramesCount; i++) {
            for (int j = 0; j < samplePerFrame; j++) {
                splitFrames[i][j] = splitFrames[i][j] * hammingWindow[j];
            }
        }
        return splitFrames;
    }

    //4:Fast Fourier Transform
    //To convert each frame of N samples from time domain into frequency domain
    //The framing data is still a mixture of many high and low frequency sounds. The data at this time is the time domain. The Fourier transform can be converted into the frequency domain to divide the complex sound waves into sound waves of various frequencies
    public double[][] fastFourierTransform(double[][] splitFrames) {
        FastFourierTransform fft = new FastFourierTransform();
        for (int m = 0; m < splitFrames.length; m++) {
            double[] frameData = splitFrames[m];
            splitFrames[m] = fft.getMagnitudes(frameData);
        }
        return splitFrames;
    }

    //5:Mel Filter Bank Processing
    //The frequencies range in FFT spectrum is very wide and voice signal does not follow the linear scale
    //When the frequency is small, mel changes faster with Hz; when the frequency is large, the rise of mel is slow and the slope of the curve is small.
    // This shows that the human ear is more sensitive to low-frequency tones, and the human ear is very slow at high frequencies.
    // The Meyer scale filter set is inspired by this.
    public double[][] melScaleFilterBankProcessor(double[][] fftSignal, int melFiltersCount, int fftSize, double generalFreqLow, double generalFreqHigh, int samplePerFrame, int samplingRate) {
        double[] mels = new double[melFiltersCount + 2];
        double[] hzs = new double[melFiltersCount + 2];
        //get lowest and highest Mel Frequency (general frequency to mel frequency);convert Hz to Mel
        double melFreqLow = 2595 * Math.log10(1 + generalFreqLow / 700);
        double melFreqHigh = 2595 * Math.log10(1 + generalFreqHigh / 700);
        //Sampling interval frequency
        double deltaMelFreq = (melFreqHigh - melFreqLow) / (melFiltersCount + 2);

        mels[0] = melFreqLow;
        mels[mels.length - 1] = melFreqHigh;
        //get mel Filters value, X VALUE
        for (int i = 1; i <= melFiltersCount; i++) {
            mels[i] = melFreqLow + (i + 1) * deltaMelFreq;
        }
        //reverse
        for (int i = 0; i < mels.length; i++) {
            hzs[i] = 700 * (Math.pow(10, mels[i] / 2595) - 1);
        }
        //computer Y VALUE
        int[] fftBin = new int[melFiltersCount + 2];
        for (int i = 0; i < fftBin.length; i++) {
            fftBin[i] = (int) Math.round((fftSize + 1) * hzs[i] / samplingRate * 2) - 1;
        }
        //filter data
        double[][] fbanks = new double[fftSignal.length][melFiltersCount];
        for (int m = 0; m < fftSignal.length; m++) {
            double[] fftData = fftSignal[m];
            double[] melfilters = new double[melFiltersCount];
            for (int i = 1; i <= melFiltersCount; i++) {
                double leftVal = 0;
                double rightVal = 0;
                int left = fftBin[i - 1];
                int center = fftBin[i];
                int right = fftBin[i + 1];
                for (int k = left; k <= center; k++) {
                    leftVal += ((k - fftBin[i - 1]) / (fftBin[i] - fftBin[i - 1])) * fftData[k];
                }
                for (int k = center + 1; k < right; k++) {
                    rightVal += (1 - (fftBin[i + 1] - k) / (fftBin[i + 1] - fftBin[i])) * fftData[k];
                }
                //sum the energy in each filter
                melfilters[i - 1] = leftVal + rightVal;
                //take the logarithm of all filterbank energies.
                melfilters[i - 1] = Math.log(melfilters[i - 1]);
            }
            fbanks[m] = melfilters;
        }
        return fbanks;
    }

    //mean normalization
    //If the Mel-scaled filter banks were the desired features then we can skip to mean normalization.
    public double[][] meanNormalization(double[][] mfccFeature) {
        double sum;
        double mean;
        for (int i = 0; i < mfccFeature.length; i++) {
            // calculate mean
            sum = 0.0;
            for (int j = 0; j < mfccFeature[i].length; j++) {
                sum += mfccFeature[i][j];// ith coeff of all frame
            }
            mean = sum / mfccFeature[i].length;
            // subtract
            for (int j = 0; j < mfccFeature[i].length; j++) {
                mfccFeature[i][j] = mfccFeature[i][j] - mean;
            }
        }
        return mfccFeature;
    }

    //6:Discrete Cosine Transform (DCT)
    //numCeps:cepstral coefficients 2-13
    public double[][] dct(double[][] mfccFeature, int numCeps) {
        for (int k = 0; k < mfccFeature.length; k ++) {
            double cepc[] = new double[numCeps];
            int N = mfccFeature[k].length;
            for (int i = 1; i <= numCeps; i ++) {
                for (int j = 1; j <= N; j ++) {
                    cepc[i - 1] += mfccFeature[k][j - 1] * Math.cos(Math.PI * (i - 1) / N * (j - 0.5));
                }
                if(i == 1) {
                    cepc[i - 1] = cepc[i - 1] * Math.sqrt(2/N)* Math.sqrt(1/N);
                } else {
                    cepc[i - 1] = cepc[i - 1] * (2 * 1.0/N);
                }
            }
            mfccFeature[k] = cepc;
        }
        return mfccFeature;
    }


    //7:Delta Energy and Delta Spectrum
    //The voice signal and the frames changes, such as the slope of a formant at its transitions.
    //numCeps:cepstral coefficients 2-13
    public double[][] doCepstralMeanNormalization(double[][] mfccFeature, int numCeps) {
        double sum;
        double mean;
        int numOfFrames = mfccFeature.length;
        double[][] mCeps = new double[numOfFrames][numCeps - 1];// same size
        // 1.loop through each mfcc coeff
        for (int i = 0; i < numCeps - 1; i++) {
            // calculate mean
            sum = 0.0;
            for (int j = 0; j < numOfFrames; j++) {
                sum += mfccFeature[j][i];
            }
            mean = sum / numOfFrames;
            for (int j = 0; j < numOfFrames; j++) {
                mCeps[j][i] = mfccFeature[j][i] - mean;
            }
        }
        return mCeps;
    }

    //energy of given PCM frame
    public double[] calcEnergy(int samplingRate, double[][] framedSignal) {
        try {
            double[] energyValue = new double[framedSignal.length];
            for (int i = 0; i < framedSignal.length; i++) {
                float sum = 0;
                for (int j = 0; j < samplingRate / 1000 * 40; j++) {
                    sum += Math.pow(framedSignal[i][j], 2);
                }
                energyValue[i] = Math.log(sum);
            }
            return energyValue;
        } catch (Exception e) {
            return null;
        }
    }

    //
    // Also known as differential and acceleration coefficients.
    // The MFCC feature vector describes only the power spectral envelope of a single frame,
    // but it seems like speech would also have information in the dynamics i.e. what are the trajectories of the MFCC coefficients over time.
    // It turns out that calculating the MFCC trajectories and appending them to the original feature vector increases ASR performance by quite a bit
    public double[] diffDelta1D(double[] data, int wid) {
        int dataSize = data.length;

        double sqSum = 0;
        for (int i = 1; i <= wid; i++) {
            sqSum +=  2.0 * Math.pow(i, 2);
        }
        double[] delta1D = new double[dataSize];
        for (int j = 1; j < dataSize - wid * 2; j++) {
            double sumData = 0;
            for (int k = 1; k <= wid; k ++) {
                sumData += 1.0 * k * (data[j + k + 1] - data[j + k - 1]);
            }
            delta1D[j-1] = sumData / sqSum;
        }
        return delta1D;
    }
    public double[][] diffDelta2D(double[][] data, int wid) {

        for (int i = 0; i < data.length; i ++) {
            diffDelta1D(data[i], wid);
        }
        return data;
    }
}
