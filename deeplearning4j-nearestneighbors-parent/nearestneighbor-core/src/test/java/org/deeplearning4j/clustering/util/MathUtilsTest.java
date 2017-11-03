/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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

package org.deeplearning4j.clustering.util;

import org.junit.Test;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.*;

/**
 * Unit tests for class {@link MathUtils}.
 *
 * @see MathUtils
 **/
public class MathUtilsTest {

  @Test
  public void testKroneckerDeltaReturningPositive() {
    assertEquals(1, MathUtils.kroneckerDelta(0.0, 0.0));
  }

  @Test
  public void testKroneckerDeltaReturningZero() {
    assertEquals(0, MathUtils.kroneckerDelta(0.0, (-2.9988278531841556E-102)));
  }

  @Test
  public void testEntropyWithEmptyArray() {
    assertEquals(0.0, MathUtils.entropy(new double[0]));
  }

  @Test
  public void testEntropyReturningPositive() {
    double[] doubleArray = new double[2];
    doubleArray[0] = 24.23;
    doubleArray[1] = 22.23;

    assertEquals(146.18041473180176, MathUtils.entropy(doubleArray));
  }

  @Test
  public void testEntropyWithNull() {
    assertEquals(0.0, MathUtils.entropy(null));
  }

  @Test
  public void testRootMeansSquaredError() {
    assertEquals(0.0, MathUtils.rootMeansSquaredError(new double[9], new double[9]));
  }

  @Test
  public void testLog2() {
    assertEquals(11.090112419664289, MathUtils.log2((2180L)));
  }

  @Test
  public void testXValsReturningNull() {
    assertNull(MathUtils.xVals(null));
  }

  @Test
  public void testSumOfProductsWithEmptyArray() {
    assertEquals(0.0, MathUtils.sumOfProducts(new double[0][0]));
  }

  @Test
  public void testSumOfProductsWithNull() {
    assertEquals(0.0, MathUtils.sumOfProducts((double[][]) null));
  }

  @Test
  public void testTimesWithEmptyArray() {
    assertEquals(0.0, MathUtils.times(new double[0]));
  }

  @Test
  public void testTimesWithNull() {
    assertEquals(0.0, MathUtils.times(null));
  }

  @Test
  public void testSquaredLoss() {
    assertEquals(310547.76803265, MathUtils.squaredLoss(new double[1], new double[1], 557.2681293889415, 557.2681293889415));
  }

  @Test
  public void testMergeCoords() {
    List<Double> coordListOne = new LinkedList<>();
    coordListOne.add(new Double(3671.9935076684487));
    coordListOne.add(new Double(2342.2342));
    List<Double> coordListTwo = new LinkedList<>();
    coordListTwo.add(new Double(2134.234));
    coordListTwo.add(new Double(3425.22));

    List<Double> mergedCoords = MathUtils.mergeCoords(coordListOne, coordListTwo);

    assertEquals(4, mergedCoords.size());
    assertEquals(3671.9935076684487, mergedCoords.get(0));
    assertEquals(2134.234, mergedCoords.get(1));
    assertEquals(2342.2342, mergedCoords.get(2));
    assertEquals(3425.22, mergedCoords.get(3));
  }

  @Test
  public void testMergeCoordsTaking2DoubleArraysThrowsIllegalArgumentException() {
    double[] doubleArray = new double[2];
    double[] doubleArrayTwo = new double[3];

    try {
      MathUtils.mergeCoords(doubleArray, doubleArrayTwo);
      fail("Expecting exception: IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals(MathUtils.class.getName(), e.getStackTrace()[0].getClassName());
    }
  }

  @Test
  public void testIdfReturningZero() {
    assertEquals(0.0, MathUtils.idf(0.0, 0.0));
  }

  @Test
  public void testIdfReturningPositive() {
    assertEquals(209.75960124794813, MathUtils.idf((1211.767005822405), 2.1077440662171152E-207));
  }

  @Test
  public void testVectorLengthWithNull() {
    assertEquals(0.0, MathUtils.vectorLength(null));
  }

  @Test
  public void testVectorLength() {
    double[] doubleArray = new double[6];

    assertEquals(0.0, MathUtils.vectorLength(doubleArray));

    doubleArray[0] = 1;
    doubleArray[1] = 2;
    doubleArray[2] = 3;
    doubleArray[3] = 4;

    assertEquals(30.0, MathUtils.vectorLength(doubleArray));
  }

  @Test
  public void testStringSimilarityReturningPositive() {
    String[] stringArray = new String[5];
    stringArray[0] = "probToLogOdds: probability must be in [0,1] ";
    stringArray[1] = "Wk/ds+f7";
    assertEquals(0.13245323570650439, MathUtils.stringSimilarity(stringArray));
  }

  @Test
  public void testStringSimilarityReturningZero() {
    assertEquals(0.0, MathUtils.stringSimilarity((String[]) null));
  }

  @Test
  public void testClamp() {
    assertEquals(-67108865, MathUtils.clamp(8, 93, (-67108865)));
  }

  @Test
  public void testDiscretizeThrowsIllegalArgumentException() {
    try {
      MathUtils.discretize(2966.23308737, 2966.23308737, 0.0, 974);
      fail("Expecting exception: IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals(MathUtils.class.getName(), e.getStackTrace()[0].getClassName());
    }
  }

  @Test
  public void testNextPowOfThree() {
    assertEquals(256, MathUtils.nextPowOf2((180L)));
  }

  @Test
  public void testTfidf() {
    assertEquals(48.884, MathUtils.tfidf(2.2, 22.22));
  }

  @Test
  public void testTf() {
    assertTrue(Double.isInfinite(MathUtils.tf(4198, 0)));
    assertEquals(2099.0, MathUtils.tf(4198, 2));
  }

  @Test
  public void testBernoullis() {
    assertEquals(Double.NaN, MathUtils.bernoullis(355.5756935413, 355.5756935413, 1.0), 0.01);
  }

  @Test
  public void testAdjustedrSquared() {
    assertEquals(1.7778582512901604, MathUtils.adjustedrSquared(0.06657009845180759, (-55), (-24)), 0.01);
  }

  @Test
  public void testW_0() {
    assertEquals(Double.NaN, MathUtils.w_0(new double[5], new double[5], 0), 0.01);
  }

  @Test
  public void testWeightsForTakingListThrowsNullPointerException() {
    try {
      MathUtils.weightsFor((List<Double>) null);
      fail("Expecting exception: NullPointerException");
    } catch (NullPointerException e) {
      assertEquals(MathUtils.class.getName(), e.getStackTrace()[0].getClassName());
    }
  }

  @Test
  public void testDeterminationCoefficient() {
    assertEquals(Double.NaN, MathUtils.determinationCoefficient(new double[5], new double[5], 1386), 0.01);
  }

  @Test
  public void testErrorFor() {
    assertEquals((-1840.704164293), MathUtils.errorFor((-2180L), (-339.295835707)), 0.01);
  }

  @Test
  public void testSigmoid() {
    assertEquals(1.0, MathUtils.sigmoid(2.0684484008569103E67), 0.01);
  }

  @Test
  public void testVariance() {
    assertEquals(Double.NaN, MathUtils.variance(new double[0]), 0.01);
  }

  @Test
  public void testUniform() {
    assertEquals(0.26912180929670915, MathUtils.uniform(new Random(1), 1, 0.0), 0.01);
  }

  @Test
  public void testWeightsForTakingDoubleArrayThrowsNullPointerException() {
    try {
      MathUtils.weightsFor((double[]) null);
      fail("Expecting exception: NullPointerException");
    } catch (NullPointerException e) {
      assertEquals(MathUtils.class.getName(), e.getStackTrace()[0].getClassName());
    }
  }

  @Test
  public void testDiscretize() {
    assertEquals(0, MathUtils.discretize(0, 0, 0, 3216));
  }

  @Test
  public void testNormalizeToOneThrowsIllegalArgumentExceptionTwo() {
    try {
      MathUtils.normalizeToOne(new double[9]);
      fail("Expecting exception: IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals(MathUtils.class.getName(), e.getStackTrace()[0].getClassName());
    }
  }

  @Test
  public void testSlope() {
    MathUtils mathUtils = new MathUtils();
    assertEquals(1054.0496925452, mathUtils.slope(0.0, 1.0, 0.0, 1054.0496925452), 0.01);
  }

  @Test
  public void testRoundFloat() {
    assertEquals(0.0F, MathUtils.roundFloat(0.0F, 0), 0.01F);
  }

  @Test
  public void testRoundDouble() {
    assertEquals(Double.NaN, MathUtils.roundDouble(0.0, (-2985)), 0.01);
  }
}