/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.dataset;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 7/15/16.
 */
@Slf4j
@RunWith(Parameterized.class)
public class PreProcessor3D4DTest extends BaseNd4jTest {

    public PreProcessor3D4DTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBruteForce3d() {

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();

        int timeSteps = 15;
        int samples = 100;
        //multiplier for the features
        INDArray featureScaleA = Nd4j.create(new double[] {1, -2, 3}).reshape(3, 1);
        INDArray featureScaleB = Nd4j.create(new double[] {2, 2, 3}).reshape(3, 1);

        Construct3dDataSet caseA = new Construct3dDataSet(featureScaleA, timeSteps, samples, 1);
        Construct3dDataSet caseB = new Construct3dDataSet(featureScaleB, timeSteps, samples, 1);

        myNormalizer.fit(caseA.sampleDataSet);
        assertEquals(caseA.expectedMean, myNormalizer.getMean());
        assertTrue(Transforms.abs(myNormalizer.getStd().div(caseA.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        myMinMaxScaler.fit(caseB.sampleDataSet);
        assertEquals(caseB.expectedMin, myMinMaxScaler.getMin());
        assertEquals(caseB.expectedMax, myMinMaxScaler.getMax());

        //Same Test with an Iterator, values should be close for std, exact for everything else
        DataSetIterator sampleIterA = new TestDataSetIterator(caseA.sampleDataSet, 5);
        DataSetIterator sampleIterB = new TestDataSetIterator(caseB.sampleDataSet, 5);

        myNormalizer.fit(sampleIterA);
        assertEquals(myNormalizer.getMean(), caseA.expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().div(caseA.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        myMinMaxScaler.fit(sampleIterB);
        assertEquals(myMinMaxScaler.getMin(), caseB.expectedMin);
        assertEquals(myMinMaxScaler.getMax(), caseB.expectedMax);

    }

    @Test
    public void testBruteForce3dMaskLabels() {

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fitLabel(true);
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fitLabel(true);

        //generating a dataset with consecutive numbers as feature values. Dataset also has masks
        int samples = 100;
        INDArray featureScale = Nd4j.create(new float[] {1, 2, 10}).reshape(3, 1);
        int timeStepsU = 5;
        Construct3dDataSet sampleU = new Construct3dDataSet(featureScale, timeStepsU, samples, 1);
        int timeStepsV = 3;
        Construct3dDataSet sampleV = new Construct3dDataSet(featureScale, timeStepsV, samples, sampleU.newOrigin);
        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(sampleU.sampleDataSet);
        dataSetList.add(sampleV.sampleDataSet);

        DataSet fullDataSetA = DataSet.merge(dataSetList);
        DataSet fullDataSetAA = fullDataSetA.copy();
        //This should be the same datasets as above without a mask
        Construct3dDataSet fullDataSetNoMask =
                        new Construct3dDataSet(featureScale, timeStepsU + timeStepsV, samples, 1);

        //preprocessors - label and feature values are the same
        myNormalizer.fit(fullDataSetA);
        assertEquals(myNormalizer.getMean(), fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getStd(), fullDataSetNoMask.expectedStd);
        assertEquals(myNormalizer.getLabelMean(), fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getLabelStd(), fullDataSetNoMask.expectedStd);

        myMinMaxScaler.fit(fullDataSetAA);
        assertEquals(myMinMaxScaler.getMin(), fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getMax(), fullDataSetNoMask.expectedMax);
        assertEquals(myMinMaxScaler.getLabelMin(), fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getLabelMax(), fullDataSetNoMask.expectedMax);


        //Same Test with an Iterator, values should be close for std, exact for everything else
        DataSetIterator sampleIterA = new TestDataSetIterator(fullDataSetA, 5);
        DataSetIterator sampleIterB = new TestDataSetIterator(fullDataSetAA, 5);

        myNormalizer.fit(sampleIterA);
        assertEquals(myNormalizer.getMean(), fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getLabelMean(), fullDataSetNoMask.expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().div(fullDataSetNoMask.expectedStd).sub(1)).maxNumber()
                        .floatValue() < 0.01);
        assertTrue(Transforms.abs(myNormalizer.getLabelStd().div(fullDataSetNoMask.expectedStd).sub(1)).maxNumber()
                        .floatValue() < 0.01);

        myMinMaxScaler.fit(sampleIterB);
        assertEquals(myMinMaxScaler.getMin(), fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getMax(), fullDataSetNoMask.expectedMax);
        assertEquals(myMinMaxScaler.getLabelMin(), fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getLabelMax(), fullDataSetNoMask.expectedMax);
    }

    @Test
    public void testStdX() throws Exception {
        INDArray array = Nd4j.create(new double[] {11.10, 22.20, 33.30, 44.40, 55.50, 66.60, 77.70, 88.80, 99.90,
                        111.00, 122.10, 133.20, 144.30, 155.40, 166.50, 177.60, 188.70, 199.80, 210.90, 222.00, 233.10,
                        244.20, 255.30, 266.40, 277.50, 288.60, 299.70, 310.80, 321.90, 333.00, 344.10, 355.20, 366.30,
                        377.40, 388.50, 399.60, 410.70, 421.80, 432.90, 444.00, 455.10, 466.20, 477.30, 488.40, 499.50,
                        510.60, 521.70, 532.80, 543.90, 555.00, 566.10, 577.20, 588.30, 599.40, 610.50, 621.60, 632.70,
                        643.80, 654.90, 666.00, 677.10, 688.20, 699.30, 710.40, 721.50, 732.60, 743.70, 754.80, 765.90,
                        777.00, 788.10, 799.20, 810.30, 821.40, 832.50, 843.60, 854.70, 865.80, 876.90, 888.00, 899.10,
                        910.20, 921.30, 932.40, 943.50, 954.60, 965.70, 976.80, 987.90, 999.00, 1, 010.10, 1, 021.20, 1,
                        032.30, 1, 043.40, 1, 054.50, 1, 065.60, 1, 076.70, 1, 087.80, 1, 098.90, 1, 110.00, 1, 121.10,
                        1, 132.20, 1, 143.30, 1, 154.40, 1, 165.50, 1, 176.60, 1, 187.70, 1, 198.80, 1, 209.90, 1,
                        221.00, 1, 232.10, 1, 243.20, 1, 254.30, 1, 265.40, 1, 276.50, 1, 287.60, 1, 298.70, 1, 309.80,
                        1, 320.90, 1, 332.00, 1, 343.10, 1, 354.20, 1, 365.30, 1, 376.40, 1, 387.50, 1, 398.60, 1,
                        409.70, 1, 420.80, 1, 431.90, 1, 443.00, 1, 454.10, 1, 465.20, 1, 476.30, 1, 487.40, 1, 498.50,
                        1, 509.60, 1, 520.70, 1, 531.80, 1, 542.90, 1, 554.00, 1, 565.10, 1, 576.20, 1, 587.30, 1,
                        598.40, 1, 609.50, 1, 620.60, 1, 631.70, 1, 642.80, 1, 653.90, 1, 665.00, 2.10, 4.20, 6.30,
                        8.40, 10.50, 12.60, 14.70, 16.80, 18.90, 21.00, 23.10, 25.20, 27.30, 29.40, 31.50, 33.60, 35.70,
                        37.80, 39.90, 42.00, 44.10, 46.20, 48.30, 50.40, 52.50, 54.60, 56.70, 58.80, 60.90, 63.00,
                        65.10, 67.20, 69.30, 71.40, 73.50, 75.60, 77.70, 79.80, 81.90, 84.00, 86.10, 88.20, 90.30,
                        92.40, 94.50, 96.60, 98.70, 100.80, 102.90, 105.00, 107.10, 109.20, 111.30, 113.40, 115.50,
                        117.60, 119.70, 121.80, 123.90, 126.00, 128.10, 130.20, 132.30, 134.40, 136.50, 138.60, 140.70,
                        142.80, 144.90, 147.00, 149.10, 151.20, 153.30, 155.40, 157.50, 159.60, 161.70, 163.80, 165.90,
                        168.00, 170.10, 172.20, 174.30, 176.40, 178.50, 180.60, 182.70, 184.80, 186.90, 189.00, 191.10,
                        193.20, 195.30, 197.40, 199.50, 201.60, 203.70, 205.80, 207.90, 210.00, 212.10, 214.20, 216.30,
                        218.40, 220.50, 222.60, 224.70, 226.80, 228.90, 231.00, 233.10, 235.20, 237.30, 239.40, 241.50,
                        243.60, 245.70, 247.80, 249.90, 252.00, 254.10, 256.20, 258.30, 260.40, 262.50, 264.60, 266.70,
                        268.80, 270.90, 273.00, 275.10, 277.20, 279.30, 281.40, 283.50, 285.60, 287.70, 289.80, 291.90,
                        294.00, 296.10, 298.20, 300.30, 302.40, 304.50, 306.60, 308.70, 310.80, 312.90, 315.00, 10.00,
                        20.00, 30.00, 40.00, 50.00, 60.00, 70.00, 80.00, 90.00, 100.00, 110.00, 120.00, 130.00, 140.00,
                        150.00, 160.00, 170.00, 180.00, 190.00, 200.00, 210.00, 220.00, 230.00, 240.00, 250.00, 260.00,
                        270.00, 280.00, 290.00, 300.00, 310.00, 320.00, 330.00, 340.00, 350.00, 360.00, 370.00, 380.00,
                        390.00, 400.00, 410.00, 420.00, 430.00, 440.00, 450.00, 460.00, 470.00, 480.00, 490.00, 500.00,
                        510.00, 520.00, 530.00, 540.00, 550.00, 560.00, 570.00, 580.00, 590.00, 600.00, 610.00, 620.00,
                        630.00, 640.00, 650.00, 660.00, 670.00, 680.00, 690.00, 700.00, 710.00, 720.00, 730.00, 740.00,
                        750.00, 760.00, 770.00, 780.00, 790.00, 800.00, 810.00, 820.00, 830.00, 840.00, 850.00, 860.00,
                        870.00, 880.00, 890.00, 900.00, 910.00, 920.00, 930.00, 940.00, 950.00, 960.00, 970.00, 980.00,
                        990.00, 1, 000.00, 1, 010.00, 1, 020.00, 1, 030.00, 1, 040.00, 1, 050.00, 1, 060.00, 1, 070.00,
                        1, 080.00, 1, 090.00, 1, 100.00, 1, 110.00, 1, 120.00, 1, 130.00, 1, 140.00, 1, 150.00, 1,
                        160.00, 1, 170.00, 1, 180.00, 1, 190.00, 1, 200.00, 1, 210.00, 1, 220.00, 1, 230.00, 1, 240.00,
                        1, 250.00, 1, 260.00, 1, 270.00, 1, 280.00, 1, 290.00, 1, 300.00, 1, 310.00, 1, 320.00, 1,
                        330.00, 1, 340.00, 1, 350.00, 1, 360.00, 1, 370.00, 1, 380.00, 1, 390.00, 1, 400.00, 1, 410.00,
                        1, 420.00, 1, 430.00, 1, 440.00, 1, 450.00, 1, 460.00, 1, 470.00, 1, 480.00, 1, 490.00, 1,
                        500.00, 99.00, 198.00, 297.00, 396.00, 495.00, 594.00, 693.00, 792.00, 891.00, 990.00, 1,
                        089.00, 1, 188.00, 1, 287.00, 1, 386.00, 1, 485.00, 1, 584.00, 1, 683.00, 1, 782.00, 1, 881.00,
                        1, 980.00, 2, 079.00, 2, 178.00, 2, 277.00, 2, 376.00, 2, 475.00, 2, 574.00, 2, 673.00, 2,
                        772.00, 2, 871.00, 2, 970.00, 3, 069.00, 3, 168.00, 3, 267.00, 3, 366.00, 3, 465.00, 3, 564.00,
                        3, 663.00, 3, 762.00, 3, 861.00, 3, 960.00, 4, 059.00, 4, 158.00, 4, 257.00, 4, 356.00, 4,
                        455.00, 4, 554.00, 4, 653.00, 4, 752.00, 4, 851.00, 4, 950.00, 5, 049.00, 5, 148.00, 5, 247.00,
                        5, 346.00, 5, 445.00, 5, 544.00, 5, 643.00, 5, 742.00, 5, 841.00, 5, 940.00, 6, 039.00, 6,
                        138.00, 6, 237.00, 6, 336.00, 6, 435.00, 6, 534.00, 6, 633.00, 6, 732.00, 6, 831.00, 6, 930.00,
                        7, 029.00, 7, 128.00, 7, 227.00, 7, 326.00, 7, 425.00, 7, 524.00, 7, 623.00, 7, 722.00, 7,
                        821.00, 7, 920.00, 8, 019.00, 8, 118.00, 8, 217.00, 8, 316.00, 8, 415.00, 8, 514.00, 8, 613.00,
                        8, 712.00, 8, 811.00, 8, 910.00, 9, 009.00, 9, 108.00, 9, 207.00, 9, 306.00, 9, 405.00, 9,
                        504.00, 9, 603.00, 9, 702.00, 9, 801.00, 9, 900.00, 9, 999.00, 10, 098.00, 10, 197.00, 10,
                        296.00, 10, 395.00, 10, 494.00, 10, 593.00, 10, 692.00, 10, 791.00, 10, 890.00, 10, 989.00, 11,
                        088.00, 11, 187.00, 11, 286.00, 11, 385.00, 11, 484.00, 11, 583.00, 11, 682.00, 11, 781.00, 11,
                        880.00, 11, 979.00, 12, 078.00, 12, 177.00, 12, 276.00, 12, 375.00, 12, 474.00, 12, 573.00, 12,
                        672.00, 12, 771.00, 12, 870.00, 12, 969.00, 13, 068.00, 13, 167.00, 13, 266.00, 13, 365.00, 13,
                        464.00, 13, 563.00, 13, 662.00, 13, 761.00, 13, 860.00, 13, 959.00, 14, 058.00, 14, 157.00, 14,
                        256.00, 14, 355.00, 14, 454.00, 14, 553.00, 14, 652.00, 14, 751.00, 14, 850.00, 7.16, 14.31,
                        21.47, 28.62, 35.78, 42.94, 50.09, 57.25, 64.40, 71.56, 78.72, 85.87, 93.03, 100.18, 107.34,
                        114.50, 121.65, 128.81, 135.96, 143.12, 150.28, 157.43, 164.59, 171.74, 178.90, 186.06, 193.21,
                        200.37, 207.52, 214.68, 221.84, 228.99, 236.15, 243.30, 250.46, 257.62, 264.77, 271.93, 279.08,
                        286.24, 293.40, 300.55, 307.71, 314.86, 322.02, 329.18, 336.33, 343.49, 350.64, 357.80, 364.96,
                        372.11, 379.27, 386.42, 393.58, 400.74, 407.89, 415.05, 422.20, 429.36, 436.52, 443.67, 450.83,
                        457.98, 465.14, 472.30, 479.45, 486.61, 493.76, 500.92, 508.08, 515.23, 522.39, 529.54, 536.70,
                        543.86, 551.01, 558.17, 565.32, 572.48, 579.64, 586.79, 593.95, 601.10, 608.26, 615.42, 622.57,
                        629.73, 636.88, 644.04, 651.20, 658.35, 665.51, 672.66, 679.82, 686.98, 694.13, 701.29, 708.44,
                        715.60, 722.76, 729.91, 737.07, 744.22, 751.38, 758.54, 765.69, 772.85, 780.00, 787.16, 794.32,
                        801.47, 808.63, 815.78, 822.94, 830.10, 837.25, 844.41, 851.56, 858.72, 865.88, 873.03, 880.19,
                        887.34, 894.50, 901.66, 908.81, 915.97, 923.12, 930.28, 937.44, 944.59, 951.75, 958.90, 966.06,
                        973.22, 980.37, 987.53, 994.68, 1, 001.84, 1, 009.00, 1, 016.15, 1, 023.31, 1, 030.46, 1,
                        037.62, 1, 044.78, 1, 051.93, 1, 059.09, 1, 066.24, 1, 073.40, 9.00, 18.00, 27.00, 36.00, 45.00,
                        54.00, 63.00, 72.00, 81.00, 90.00, 99.00, 108.00, 117.00, 126.00, 135.00, 144.00, 153.00,
                        162.00, 171.00, 180.00, 189.00, 198.00, 207.00, 216.00, 225.00, 234.00, 243.00, 252.00, 261.00,
                        270.00, 279.00, 288.00, 297.00, 306.00, 315.00, 324.00, 333.00, 342.00, 351.00, 360.00, 369.00,
                        378.00, 387.00, 396.00, 405.00, 414.00, 423.00, 432.00, 441.00, 450.00, 459.00, 468.00, 477.00,
                        486.00, 495.00, 504.00, 513.00, 522.00, 531.00, 540.00, 549.00, 558.00, 567.00, 576.00, 585.00,
                        594.00, 603.00, 612.00, 621.00, 630.00, 639.00, 648.00, 657.00, 666.00, 675.00, 684.00, 693.00,
                        702.00, 711.00, 720.00, 729.00, 738.00, 747.00, 756.00, 765.00, 774.00, 783.00, 792.00, 801.00,
                        810.00, 819.00, 828.00, 837.00, 846.00, 855.00, 864.00, 873.00, 882.00, 891.00, 900.00, 909.00,
                        918.00, 927.00, 936.00, 945.00, 954.00, 963.00, 972.00, 981.00, 990.00, 999.00, 1, 008.00, 1,
                        017.00, 1, 026.00, 1, 035.00, 1, 044.00, 1, 053.00, 1, 062.00, 1, 071.00, 1, 080.00, 1, 089.00,
                        1, 098.00, 1, 107.00, 1, 116.00, 1, 125.00, 1, 134.00, 1, 143.00, 1, 152.00, 1, 161.00, 1,
                        170.00, 1, 179.00, 1, 188.00, 1, 197.00, 1, 206.00, 1, 215.00, 1, 224.00, 1, 233.00, 1, 242.00,
                        1, 251.00, 1, 260.00, 1, 269.00, 1, 278.00, 1, 287.00, 1, 296.00, 1, 305.00, 1, 314.00, 1,
                        323.00, 1, 332.00, 1, 341.00, 1, 350.00});

        float templateStd = array.std(1).getFloat(0, 0);

        assertEquals(301.22601, templateStd, 0.01);
    }

    @Test
    public void testBruteForce4d() {
        Construct4dDataSet imageDataSet = new Construct4dDataSet(10, 5, 10, 15);

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fit(imageDataSet.sampleDataSet);
        assertEquals(imageDataSet.expectedMean, myNormalizer.getMean());

        float aat = Transforms.abs(myNormalizer.getStd().div(imageDataSet.expectedStd).sub(1)).maxNumber().floatValue();
        float abt = myNormalizer.getStd().maxNumber().floatValue();
        float act = imageDataSet.expectedStd.maxNumber().floatValue();
        System.out.println("ValA: " + aat);
        System.out.println("ValB: " + abt);
        System.out.println("ValC: " + act);
        assertTrue(aat < 0.05);

        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fit(imageDataSet.sampleDataSet);
        assertEquals(imageDataSet.expectedMin, myMinMaxScaler.getMin());
        assertEquals(imageDataSet.expectedMax, myMinMaxScaler.getMax());

        DataSet copyDataSet = imageDataSet.sampleDataSet.copy();
        myNormalizer.transform(copyDataSet);
    }

    @Test
    public void test3dRevertStandardize() {
        test3dRevert(new NormalizerStandardize());
    }

    @Test
    public void test3dRevertNormalize() {
        test3dRevert(new NormalizerMinMaxScaler());
    }

    private void test3dRevert(DataNormalization SUT) {
        INDArray features = Nd4j.rand(new int[] {5, 2, 10}, 12345).muli(2).addi(1);
        DataSet data = new DataSet(features, Nd4j.zeros(5, 1, 10));
        DataSet dataCopy = data.copy();

        SUT.fit(data);

        SUT.preProcess(data);
        assertNotEquals(data, dataCopy);

        SUT.revert(data);
        assertEquals(dataCopy.getFeatures(), data.getFeatures());
        assertEquals(dataCopy.getLabels(), data.getLabels());
    }

    @Test
    public void test3dNinMaxScaling() {
        INDArray values = Nd4j.linspace(-10, 10, 100).reshape(5, 2, 10);
        DataSet data = new DataSet(values, values);

        NormalizerMinMaxScaler SUT = new NormalizerMinMaxScaler();
        SUT.fit(data);
        SUT.preProcess(data);

        // Data should now be in a 0-1 range
        float min = data.getFeatures().minNumber().floatValue();
        float max = data.getFeatures().maxNumber().floatValue();

        assertEquals(0, min, Nd4j.EPS_THRESHOLD);
        assertEquals(1, max, Nd4j.EPS_THRESHOLD);
    }

    public class Construct3dDataSet {

        /*
           This will return a dataset where the features are consecutive numbers scaled by featureScaler (a column vector)
           If more than one sample is specified it will continue the series from the last sample
           If origin is not 1, the series will start from the value given
            */
        DataSet sampleDataSet;
        INDArray featureScale;
        int numFeatures, maxN, timeSteps, samples, origin, newOrigin;
        INDArray expectedMean, expectedStd, expectedMin, expectedMax;

        public Construct3dDataSet(INDArray featureScale, int timeSteps, int samples, int origin) {
            this.featureScale = featureScale;
            this.timeSteps = timeSteps;
            this.samples = samples;
            this.origin = origin;

            // FIXME: int cast
            numFeatures = (int) featureScale.size(0);
            maxN = samples * timeSteps;
            INDArray template = Nd4j.linspace(origin, origin + timeSteps - 1, timeSteps);
            template = Nd4j.concat(0, Nd4j.linspace(origin, origin + timeSteps - 1, timeSteps), template);
            template = Nd4j.concat(0, Nd4j.linspace(origin, origin + timeSteps - 1, timeSteps), template);
            template.muliColumnVector(featureScale);
            template = template.reshape(1, numFeatures, timeSteps);
            INDArray featureMatrix = template.dup();

            int newStart = origin + timeSteps;
            int newEnd;
            for (int i = 1; i < samples; i++) {
                newEnd = newStart + timeSteps - 1;
                template = Nd4j.linspace(newStart, newEnd, timeSteps);
                template = Nd4j.concat(0, Nd4j.linspace(newStart, newEnd, timeSteps), template);
                template = Nd4j.concat(0, Nd4j.linspace(newStart, newEnd, timeSteps), template);
                template.muliColumnVector(featureScale);
                template = template.reshape(1, numFeatures, timeSteps);
                newStart = newEnd + 1;
                featureMatrix = Nd4j.concat(0, featureMatrix, template);
            }
            INDArray labelSet = featureMatrix.dup();
            this.newOrigin = newStart;
            sampleDataSet = new DataSet(featureMatrix, labelSet);

            //calculating stats
            // The theoretical mean should be the mean of 1,..samples*timesteps
            float theoreticalMean = origin - 1 + (samples * timeSteps + 1) / 2.0f;
            expectedMean = Nd4j.create(new double[] {theoreticalMean, theoreticalMean, theoreticalMean}).reshape(3, 1);
            expectedMean.muliColumnVector(featureScale);

            float stdNaturalNums = (float) Math.sqrt((samples * samples * timeSteps * timeSteps - 1) / 12);
            expectedStd = Nd4j.create(new float[] {stdNaturalNums, stdNaturalNums, stdNaturalNums}).reshape(3, 1);
            expectedStd.muliColumnVector(Transforms.abs(featureScale, true));
            //preprocessors use the population std so divides by n not (n-1)
            expectedStd = expectedStd.dup().muli(Math.sqrt(maxN)).divi(Math.sqrt(maxN)).transpose();

            //min max assumes all scaling values are +ve
            expectedMin = Nd4j.ones(3, 1).muliColumnVector(featureScale);
            expectedMax = Nd4j.ones(3, 1).muli(samples * timeSteps).muliColumnVector(featureScale);
        }

    }

    public class Construct4dDataSet {

        DataSet sampleDataSet;
        INDArray expectedMean, expectedStd, expectedMin, expectedMax;
        INDArray expectedLabelMean, expectedLabelStd, expectedLabelMin, expectedLabelMax;

        public Construct4dDataSet(int nExamples, int nChannels, int height, int width) {
            Nd4j.getRandom().setSeed(12345);

            INDArray allImages = Nd4j.rand(new int[] {nExamples, nChannels, height, width});
            allImages.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).muli(100)
                            .addi(200);
            allImages.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()).muli(0.01)
                            .subi(10);

            INDArray labels = Nd4j.linspace(1, nChannels, nChannels).reshape('c', nChannels, 1);
            sampleDataSet = new DataSet(allImages, labels);

            expectedMean = allImages.mean(0, 2, 3);
            expectedStd = allImages.std(0, 2, 3);

            expectedLabelMean = labels.mean(0);
            expectedLabelStd = labels.std(0);

            expectedMin = allImages.min(0, 2, 3);
            expectedMax = allImages.max(0, 2, 3);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
