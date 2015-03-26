/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.berkeley;


import java.util.List;
import java.util.Map;

/**
 * The class <code>SloppyMath</code> contains methods for performing basic
 * numeric operations. In some cases, such as max and min, they cut a few
 * corners in the implementation for the sake of efficiency. In particular, they
 * may not handle special notions like NaN and -0.0 correctly. This was the
 * origin of the class name, but some other operations are just useful math
 * additions, such as logSum.
 * 
 * @author Christopher Manning
 * @version 2003/01/02
 */
public final class SloppyMath {

	  public static double abs(double x) {
		    if (x > 0)
		      return x;
		    return -1.0 * x;
		  }

			public static double lambert(double v, double u){
				double x = -(Math.log(-v)+u);//-Math.log(-z);
				double w = -x;
				double diff=1;
				while (Math.abs(diff)<1.0e-5){
				  double z = -x -Math.log(Math.abs(w));
				  diff = z-w;
				  w = z;
				}
				return w;

				/*
				//Use asymptotic expansion w = log(z) - log(log(z)) for most z
				double summand = (z==0) ? 1 : 0;
				double tmp = Math.log(z+summand);// + i*b*6.28318530717958648;
				double w = tmp - Math.log(tmp + summand);

				//For b = 0, use a series expansion when close to the branch point
				//k = find(b == 0 & abs(z + 0.3678794411714423216) <= 1.5);
				tmp = Math.sqrt(5.43656365691809047*z + 2) - 1;// + i*b*6.28318530717958648;
				//w(k) = tmp(k);
		    w = tmp;
		    
				for (int k=1; k<36; k++){
					// Converge with Halley's iterations, about 5 iterations satisfies
					//the tolerance for most z
					double c1 = Math.exp(w);
					double c2 = w*c1 - z;
					summand = (w != -1) ? 1 : 0;
					double w1 = w + summand;
					double dw = c2/(c1*w1 - ((w + 2)*c2/(2*w1)));
					w = w - dw;
			
				  if (Math.abs(dw) < 0.7e-16*(2+Math.abs(w)))
				     break;
				}
				return w;*/
 	}

  /**
	 * Returns the minimum of three int values.
	 */
  public static int max(int a, int b, int c) {
    int ma;
    ma = a;
    if (b > ma) {
      ma = b;
    }
    if (c > ma) {
      ma = c;
    }
    return ma;
  }


  /**
	 * Returns the minimum of three int values.
	 */
  public static int min(int a, int b, int c) {
    int mi;

    mi = a;
    if (b < mi) {
      mi = b;
    }
    if (c < mi) {
      mi = c;
    }
    return mi;

  }


  /**
	 * Returns the greater of two <code>float</code> values. That is, the
	 * result is the argument closer to positive infinity. If the arguments have
	 * the same value, the result is that same value. Does none of the special
	 * checks for NaN or -0.0f that <code>Math.max</code> does.
	 * 
	 * @param a
	 *            an argument.
	 * @param b
	 *            another argument.
	 * @return the larger of <code>a</code> and <code>b</code>.
	 */
  public static float max(float a, float b) {
    return (a >= b) ? a : b;
  }


  /**
	 * Returns the greater of two <code>double</code> values. That is, the
	 * result is the argument closer to positive infinity. If the arguments have
	 * the same value, the result is that same value. Does none of the special
	 * checks for NaN or -0.0f that <code>Math.max</code> does.
	 * 
	 * @param a
	 *            an argument.
	 * @param b
	 *            another argument.
	 * @return the larger of <code>a</code> and <code>b</code>.
	 */
  public static double max(double a, double b) {
    return (a >= b) ? a : b;
  }


  /**
	 * Returns the smaller of two <code>float</code> values. That is, the
	 * result is the value closer to negative infinity. If the arguments have
	 * the same value, the result is that same value. Does none of the special
	 * checks for NaN or -0.0f that <code>Math.max</code> does.
	 * 
	 * @param a
	 *            an argument.
	 * @param b
	 *            another argument.
	 * @return the smaller of <code>a</code> and <code>b.</code>
	 */
  public static float min(float a, float b) {
    return (a <= b) ? a : b;
  }


  /**
	 * Returns the smaller of two <code>double</code> values. That is, the
	 * result is the value closer to negative infinity. If the arguments have
	 * the same value, the result is that same value. Does none of the special
	 * checks for NaN or -0.0f that <code>Math.max</code> does.
	 * 
	 * @param a
	 *            an argument.
	 * @param b
	 *            another argument.
	 * @return the smaller of <code>a</code> and <code>b</code>.
	 */
  public static double min(double a, double b) {
    return (a <= b) ? a : b;
  }


  /**
	 * Returns true if the argument is a "dangerous" double to have around,
	 * namely one that is infinite, NaN or zero.
	 */
  public static boolean isDangerous(double d) {
    return Double.isInfinite(d) || Double.isNaN(d) || d == 0.0;
  }
  public static boolean isDangerous(float d) {
    return Float.isInfinite(d) || Float.isNaN(d) || d == 0.0;
  }

  public static boolean isGreater(double x, double y) {
	    if (x>1) return (((x-y) / x) > -0.01);
	  	return ((x-y) > -0.0001);
  }


  /**
	 * Returns true if the argument is a "very dangerous" double to have around,
	 * namely one that is infinite or NaN.
	 */
  public static boolean isVeryDangerous(double d) {
    return Double.isInfinite(d) || Double.isNaN(d);
  }

  public static double relativeDifferance(double a, double b) {
      a = Math.abs(a);
      b = Math.abs(b);
      double absMin = Math.min(a,b);
      return Math.abs(a-b) / absMin;      
  }

  public static boolean isDiscreteProb(double d, double tol)
  {
	  return d >=0.0 && d <= 1.0 + tol;
  }
  

  /**
	 * If a difference is bigger than this in log terms, then the sum or
	 * difference of them will just be the larger (to 12 or so decimal places
	 * for double, and 7 or 8 for float).
	 */
  public static final double LOGTOLERANCE = 30.0;
  static final float LOGTOLERANCE_F = 10.0f;


  /**
	 * Returns the log of the sum of two numbers, which are themselves input in
	 * log form. This uses natural logarithms. Reasonable care is taken to do
	 * this as efficiently as possible (under the assumption that the numbers
	 * might differ greatly in magnitude), with high accuracy, and without
	 * numerical overflow. Also, handle correctly the case of arguments being
	 * -Inf (e.g., probability 0).
	 * 
	 * @param lx
	 *            First number, in log form
	 * @param ly
	 *            Second number, in log form
	 * @return log(exp(lx) + exp(ly))
	 */
  public static float logAdd(float lx, float ly) {
    float max, negDiff;
    if (lx > ly) {
      max = lx;
      negDiff = ly - lx;
    } else {
      max = ly;
      negDiff = lx - ly;
    }
    if (max == Double.NEGATIVE_INFINITY) {
      return max;
    } else if (negDiff < -LOGTOLERANCE_F) {
      return max;
    } else {
      return max + (float)Math.log(1.0f + Math.exp(negDiff));
    }
  }


  /**
	 * Returns the log of the sum of two numbers, which are themselves input in
	 * log form. This uses natural logarithms. Reasonable care is taken to do
	 * this as efficiently as possible (under the assumption that the numbers
	 * might differ greatly in magnitude), with high accuracy, and without
	 * numerical overflow. Also, handle correctly the case of arguments being
	 * -Inf (e.g., probability 0).
	 * 
	 * @param lx
	 *            First number, in log form
	 * @param ly
	 *            Second number, in log form
	 * @return log(exp(lx) + exp(ly))
	 */
  public static double logAdd(double lx, double ly) {
    double max, negDiff;
    if (lx > ly) {
      max = lx;
      negDiff = ly - lx;
    } else {
      max = ly;
      negDiff = lx - ly;
    }
    if (max == Double.NEGATIVE_INFINITY) {
      return max;
    } else if (negDiff < -LOGTOLERANCE) {
      return max;
    } else {
      return max + Math.log(1.0 + Math.exp(negDiff));
    }
  }

  public static double logAdd(float[] logV) {
    double maxIndex = 0;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logV.length; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
    // compute the negative difference
    double threshold = max - LOGTOLERANCE;
    double sumNegativeDifferences = 0.0;
    for (int i = 0; i < logV.length; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp(logV[i] - max);
      }
    }
    if (sumNegativeDifferences > 0.0) {
      return max + Math.log(1.0 + sumNegativeDifferences);
    } else {
      return max;
    }
  }

  public static void logNormalize(double[] logV) {
      double logSum = logAdd(logV);      
      if (Double.isNaN(logSum)) {
        throw new RuntimeException("Bad log-sum");
      }
      if (logSum == 0.0) return;
      for (int i = 0; i < logV.length; i++) {          
        logV[i] -= logSum;
      }
  }

  public static double logAdd(double[] logV) {
    double maxIndex = 0;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logV.length; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
    // compute the negative difference
    double threshold = max - LOGTOLERANCE;
    double sumNegativeDifferences = 0.0;
    for (int i = 0; i < logV.length; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp(logV[i] - max);
      }
    }
    if (sumNegativeDifferences > 0.0) {
      return max + Math.log(1.0 + sumNegativeDifferences);
    } else {
      return max;
    }
  }

  public static double logAdd(List<Double> logV) {
	    double max = Double.NEGATIVE_INFINITY;
	    double maxIndex = 0;
	    for (int i = 0; i < logV.size(); i++) {
	      if (logV.get(i) > max) {
	        max = logV.get(i);
	        maxIndex = i;
	      }
	    }
	    if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
	    // compute the negative difference
	    double threshold = max - LOGTOLERANCE;
	    double sumNegativeDifferences = 0.0;
	    for (int i = 0; i < logV.size(); i++) {
	      if (i != maxIndex && logV.get(i) > threshold) {
	        sumNegativeDifferences += Math.exp(logV.get(i) - max);
	      }
	    }
	    if (sumNegativeDifferences > 0.0) {
	      return max + Math.log(1.0 + sumNegativeDifferences);
	    } else {
	      return max;
	    }
	  }

  
  public static float logAdd_Old(float[] logV) {
    float max = Float.NEGATIVE_INFINITY;
    float maxIndex = 0;
    for (int i = 0; i < logV.length; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Float.NEGATIVE_INFINITY) return Float.NEGATIVE_INFINITY;
    // compute the negative difference
    float threshold = max - LOGTOLERANCE_F;
    float sumNegativeDifferences = 0.0f;
    for (int i = 0; i < logV.length; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp(logV[i] - max);
      }
    }
    if (sumNegativeDifferences > 0.0) {
      return max + (float) Math.log(1.0f + sumNegativeDifferences);
    } else {
      return max;
    }
  }

  /*
	 * adds up the entries logV[0], logV[1], ... , logV[lastIndex-1]
	 */
  public static float logAdd(float[] logV, int lastIndex) {
  	if (lastIndex==0) return Float.NEGATIVE_INFINITY;
  	float max = Float.NEGATIVE_INFINITY;
    float maxIndex = 0;
    for (int i = 0; i < lastIndex; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Float.NEGATIVE_INFINITY) return Float.NEGATIVE_INFINITY;
    // compute the negative difference
    float threshold = max - LOGTOLERANCE_F;
    double sumNegativeDifferences = 0.0;
    for (int i = 0; i < lastIndex; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp((logV[i] - max));
      }
    }
    if (sumNegativeDifferences > 0.0) {
      return max + (float) Math.log(1.0 + sumNegativeDifferences);
    } else {
      return max;
    }
  }

  /*
	 * adds up the entries logV[0], logV[1], ... , logV[lastIndex-1]
	 */
  public static double logAdd(double[] logV, int lastIndex) {
  	if (lastIndex==0) return Double.NEGATIVE_INFINITY;
  	double max = Double.NEGATIVE_INFINITY;
  	double maxIndex = 0;
    for (int i = 0; i < lastIndex; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
    // compute the negative difference
    double threshold = max - LOGTOLERANCE;
    double sumNegativeDifferences = 0.0;
    for (int i = 0; i < lastIndex; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp((logV[i] - max));
      }
    }
    if (sumNegativeDifferences > 0.0) {
      return max + Math.log(1.0 + sumNegativeDifferences);
    } else {
      return max;
    }
  }
  /**
	 * Similar to logAdd, but without the final log. I.e. Sum_i exp(logV_i)
	 * 
	 * @param logV
	 * @return
	 */
  public static float addExp_Old(float[] logV) {
    float max = Float.NEGATIVE_INFINITY;
    float maxIndex = 0;
    for (int i = 0; i < logV.length; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Float.NEGATIVE_INFINITY) return Float.NEGATIVE_INFINITY;
    // compute the negative difference
    float threshold = max - LOGTOLERANCE_F;
    float sumNegativeDifferences = 0.0f;
    for (int i = 0; i < logV.length; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp(logV[i] - max);
      }
    }
    return (float) Math.exp(max) * (1.0f + sumNegativeDifferences);
  }

  /*
	 * adds up the entries logV[0], logV[1], ... , logV[lastIndex-1]
	 */
  public static float addExp(float[] logV, int lastIndex) {
  	if (lastIndex==0) return Float.NEGATIVE_INFINITY;
  	float max = Float.NEGATIVE_INFINITY;
    float maxIndex = 0;
    for (int i = 0; i < lastIndex; i++) {
      if (logV[i] > max) {
        max = logV[i];
        maxIndex = i;
      }
    }
    if (max == Float.NEGATIVE_INFINITY) return Float.NEGATIVE_INFINITY;
    // compute the negative difference
    float threshold = max - LOGTOLERANCE_F;
    float sumNegativeDifferences = 0.0f;
    for (int i = 0; i < lastIndex; i++) {
      if (i != maxIndex && logV[i] > threshold) {
        sumNegativeDifferences += Math.exp(logV[i] - max);
      }
    }
    return (float) Math.exp(max) * (1.0f + sumNegativeDifferences);
  }
  /**
	 * Computes n choose k in an efficient way. Works with k == 0 or k == n but
	 * undefined if k < 0 or k > n
	 * 
	 * @param n
	 * @param k
	 * @return fact(n) / fact(k) * fact(n-k)
	 */
  public static int nChooseK(int n, int k) {
    k = Math.min(k, n - k);
    if (k == 0) {
      return 1;
    }
    int accum = n;
    for (int i = 1; i < k; i++) {
      accum *= (n - i);
      accum /= i;
    }
    return accum / k;
  }

  /**
	 * exponentiation like we learned in grade school: multiply b by itself e
	 * times. Uses power of two trick. e must be nonnegative!!! no checking!!!
	 * 
	 * @param b
	 *            base
	 * @param e
	 *            exponent
	 * @return b^e
	 */
  public static int intPow(int b, int e) {
    if (e == 0) {
      return 1;
    }
    int result = 1;
    int currPow = b;
    do {
      if ((e & 1) == 1) result *= currPow;
      currPow = currPow * currPow;
      e >>= 1;
    } while (e > 0);
    return result;
  }

  /**
	 * exponentiation like we learned in grade school: multiply b by itself e
	 * times. Uses power of two trick. e must be nonnegative!!! no checking!!!
	 * 
	 * @param b
	 *            base
	 * @param e
	 *            exponent
	 * @return b^e
	 */
  public static float intPow(float b, int e) {
    if (e == 0) {
      return 1;
    }
    float result = 1;
    float currPow = b;
    do {
      if ((e & 1) == 1) result *= currPow;
      currPow = currPow * currPow;
      e >>= 1;
    } while (e > 0);
    return result;
  }

  /**
	 * exponentiation like we learned in grade school: multiply b by itself e
	 * times. Uses power of two trick. e must be nonnegative!!! no checking!!!
	 * 
	 * @param b
	 *            base
	 * @param e
	 *            exponent
	 * @return b^e
	 */
  public static double intPow(double b, int e) {
    if (e == 0) {
      return 1;
    }
    float result = 1;
    double currPow = b;
    do {
      if ((e & 1) == 1) result *= currPow;
      currPow = currPow * currPow;
      e >>= 1;
    } while (e > 0);
    return result;
  }

  /**
	 * Find a hypergeometric distribution. This uses exact math, trying fairly
	 * hard to avoid numeric overflow by interleaving multiplications and
	 * divisions. (To do: make it even better at avoiding overflow, by using
	 * loops that will do either a multiple or divide based on the size of the
	 * intermediate result.)
	 * 
	 * @param k
	 *            The number of black balls drawn
	 * @param n
	 *            The total number of balls
	 * @param r
	 *            The number of black balls
	 * @param m
	 *            The number of balls drawn
	 * @return The hypergeometric value
	 */
  public static double hypergeometric(int k, int n, int r, int m) {
    if (k < 0 || r > n || m > n || n <= 0 || m < 0 | r < 0) {
      throw new IllegalArgumentException("Invalid hypergeometric");
    }

    // exploit symmetry of problem
    if (m > n / 2) {
      m = n - m;
      k = r - k;
    }
    if (r > n / 2) {
      r = n - r;
      k = m - k;
    }
    if (m > r) {
      int temp = m;
      m = r;
      r = temp;
    }
    // now we have that k <= m <= r <= n/2

    if (k < (m + r) - n || k > m) {
      return 0.0;
    }

    // Do limit cases explicitly
    // It's unclear whether this is a good idea. I put it in fearing
    // numerical errors when the numbers seemed off, but actually there
    // was a bug in the Fisher's exact routine.
    if (r == n) {
      if (k == m) {
        return 1.0;
      } else {
        return 0.0;
      }
    } else if (r == n - 1) {
      if (k == m) {
        return (n - m) / (double) n;
      } else if (k == m - 1) {
        return m / (double) n;
      } else {
        return 0.0;
      }
    } else if (m == 1) {
      if (k == 0) {
        return (n - r) / (double) n;
      } else if (k == 1) {
        return r / (double) n;
      } else {
        return 0.0;
      }
    } else if (m == 0) {
      if (k == 0) {
        return 1.0;
      } else {
        return 0.0;
      }
    } else if (k == 0) {
      double ans = 1.0;
      for (int m0 = 0; m0 < m; m0++) {
        ans *= ((n - r) - m0);
        ans /= (n - m0);
      }
      return ans;
    }

    double ans = 1.0;
    // do (n-r)x...x((n-r)-((m-k)-1))/n x...x (n-((m-k-1)))
    // leaving rest of denominator to getFromOrigin to multimply by (n-(m-1))
    // that's k things which goes into next loop
    for (int nr = n - r, n0 = n; nr > (n - r) - (m - k); nr--, n0--) {
      // System.out.println("Multiplying by " + nr);
      ans *= nr;
      // System.out.println("Dividing by " + n0);
      ans /= n0;
    }
    // System.out.println("Done phase 1");
    for (int k0 = 0; k0 < k; k0++) {
      ans *= (m - k0);
      // System.out.println("Multiplying by " + (m-k0));
      ans /= ((n - (m - k0)) + 1);
      // System.out.println("Dividing by " + ((n-(m+k0)+1)));
      ans *= (r - k0);
      // System.out.println("Multiplying by " + (r-k0));
      ans /= (k0 + 1);
      // System.out.println("Dividing by " + (k0+1));
    }
    return ans;
  }


  /**
	 * Find a one tailed exact binomial test probability. Finds the chance of
	 * this or a higher result
	 * 
	 * @param k
	 *            number of successes
	 * @param n
	 *            Number of trials
	 * @param p
	 *            Probability of a success
	 */
  public static double exactBinomial(int k, int n, double p) {
    double total = 0.0;
    for (int m = k; m <= n; m++) {
      double nChooseM = 1.0;
      for (int r = 1; r <= m; r++) {
        nChooseM *= (n - r) + 1;
        nChooseM /= r;
      }
      // System.out.println(n + " choose " + m + " is " + nChooseM);
      // System.out.println("prob contribution is " +
      // (nChooseM * Math.pow(p, m) * Math.pow(1.0-p, n - m)));
      total += nChooseM * Math.pow(p, m) * Math.pow(1.0 - p, n - m);
    }
    return total;
  }


  /**
	 * Find a one-tailed Fisher's exact probability. Chance of having seen this
	 * or a more extreme departure from what you would have expected given
	 * independence. I.e., k >= the value passed in. Warning: this was done just
	 * for collocations, where you are concerned with the case of k being larger
	 * than predicted. It doesn't correctly handle other cases, such as k being
	 * smaller than expected.
	 * 
	 * @param k
	 *            The number of black balls drawn
	 * @param n
	 *            The total number of balls
	 * @param r
	 *            The number of black balls
	 * @param m
	 *            The number of balls drawn
	 * @return The Fisher's exact p-value
	 */
  public static double oneTailedFishersExact(int k, int n, int r, int m) {
    if (k < 0 || k < (m + r) - n || k > r || k > m || r > n || m > n) {
      throw new IllegalArgumentException("Invalid Fisher's exact: " + "k=" + k + " n=" + n + " r=" + r + " m=" + m + " k<0=" + (k < 0) + " k<(m+r)-n=" + (k < (m + r) - n) + " k>r=" + (k > r) + " k>m=" + (k > m) + " r>n=" + (r > n) + "m>n=" + (m > n));
    }
    // exploit symmetry of problem
    if (m > n / 2) {
      m = n - m;
      k = r - k;
    }
    if (r > n / 2) {
      r = n - r;
      k = m - k;
    }
    if (m > r) {
      int temp = m;
      m = r;
      r = temp;
    }
    // now we have that k <= m <= r <= n/2

    double total = 0.0;
    if (k > m / 2) {
      // sum from k to m
      for (int k0 = k; k0 <= m; k0++) {
        // System.out.println("Calling hypg(" + k0 + "; " + n +
        // ", " + r + ", " + m + ")");
        total += SloppyMath.hypergeometric(k0, n, r, m);
      }
    } else {
      // sum from max(0, (m+r)-n) to k-1, and then subtract from 1
      int min = Math.max(0, (m + r) - n);
      for (int k0 = min; k0 < k; k0++) {
        // System.out.println("Calling hypg(" + k0 + "; " + n +
        // ", " + r + ", " + m + ")");
        total += SloppyMath.hypergeometric(k0, n, r, m);
      }
      total = 1.0 - total;
    }
    return total;
  }


  /**
	 * Find a 2x2 chi-square value. Note: could do this more neatly using
	 * simplified formula for 2x2 case.
	 * 
	 * @param k
	 *            The number of black balls drawn
	 * @param n
	 *            The total number of balls
	 * @param r
	 *            The number of black balls
	 * @param m
	 *            The number of balls drawn
	 * @return The Fisher's exact p-value
	 */
  public static double chiSquare2by2(int k, int n, int r, int m) {
    int[][] cg = {{k, r - k}, {m - k, n - (k + (r - k) + (m - k))}};
    int[] cgr = {r, n - r};
    int[] cgc = {m, n - m};
    double total = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        double exp = (double) cgr[i] * cgc[j] / n;
        total += (cg[i][j] - exp) * (cg[i][j] - exp) / exp;
      }
    }
    return total;
  }

  public static double exp(double logX) {
    // if x is very near one, use the linear approximation
    if (Math.abs(logX) < 0.001)
      return 1 + logX;
    return Math.exp(logX);
  }

  /**
	 * Tests the hypergeometric distribution code, or other cooccurrences provided
	 * in this module.
	 * 
	 * @param args
	 *            Either none, and the log add rountines are tested, or the
	 *            following 4 arguments: k (cell), n (total), r (row), m (col)
	 */
  public static void main(String[] args) {
    
    System.out.println(approxLog(0.0));
//    if (args.length == 0) {
//      System.err.println("Usage: java edu.stanford.nlp.math.SloppyMath " + "[-logAdd|-fishers k n r m|-bionomial r n p");
//    } else if (args[0].equals("-logAdd")) {
//      System.out.println("Log adds of neg infinity numbers, etc.");
//      System.out.println("(logs) -Inf + -Inf = " + logAdd(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));
//      System.out.println("(logs) -Inf + -7 = " + logAdd(Double.NEGATIVE_INFINITY, -7.0));
//      System.out.println("(logs) -7 + -Inf = " + logAdd(-7.0, Double.NEGATIVE_INFINITY));
//      System.out.println("(logs) -50 + -7 = " + logAdd(-50.0, -7.0));
//      System.out.println("(logs) -11 + -7 = " + logAdd(-11.0, -7.0));
//      System.out.println("(logs) -7 + -11 = " + logAdd(-7.0, -11.0));
//      System.out.println("real 1/2 + 1/2 = " + logAdd(Math.log(0.5), Math.log(0.5)));
//    } else if (args[0].equals("-fishers")) {
//      int k = Integer.parseInt(args[1]);
//      int n = Integer.parseInt(args[2]);
//      int r = Integer.parseInt(args[3]);
//      int m = Integer.parseInt(args[4]);
//      double ans = SloppyMath.hypergeometric(k, n, r, m);
//      System.out.println("hypg(" + k + "; " + n + ", " + r + ", " + m + ") = " + ans);
//      ans = SloppyMath.oneTailedFishersExact(k, n, r, m);
//      System.out.println("1-tailed Fisher's exact(" + k + "; " + n + ", " + r + ", " + m + ") = " + ans);
//      double ansChi = SloppyMath.chiSquare2by2(k, n, r, m);
//      System.out.println("chiSquare(" + k + "; " + n + ", " + r + ", " + m + ") = " + ansChi);
//
//      System.out.println("Swapping arguments should give same hypg:");
//      ans = SloppyMath.hypergeometric(k, n, r, m);
//      System.out.println("hypg(" + k + "; " + n + ", " + m + ", " + r + ") = " + ans);
//      int othrow = n - m;
//      int othcol = n - r;
//      int cell12 = m - k;
//      int cell21 = r - k;
//      int cell22 = othrow - (r - k);
//      ans = SloppyMath.hypergeometric(cell12, n, othcol, m);
//      System.out.println("hypg(" + cell12 + "; " + n + ", " + othcol + ", " + m + ") = " + ans);
//      ans = SloppyMath.hypergeometric(cell21, n, r, othrow);
//      System.out.println("hypg(" + cell21 + "; " + n + ", " + r + ", " + othrow + ") = " + ans);
//      ans = SloppyMath.hypergeometric(cell22, n, othcol, othrow);
//      System.out.println("hypg(" + cell22 + "; " + n + ", " + othcol + ", " + othrow + ") = " + ans);
//    } else if (args[0].equals("-binomial")) {
//      int k = Integer.parseInt(args[1]);
//      int n = Integer.parseInt(args[2]);
//      double p = Double.parseDouble(args[3]);
//      double ans = SloppyMath.exactBinomial(k, n, p);
//      System.out.println("Binomial p(X >= " + k + "; " + n + ", " + p + ") = " + ans);
//    }
//		else if (args[0].equals("-approxExp"))
//    {
//    	int numTrials = 0;
//    	double sumError = 0;
//    	double maxError = 0;
//    	for (double x = -700; x < 700; x += 0.1)
//    	{
//    		final double approxExp = approxExp(x);
//			final double exp = Math.exp(x);
//			double error = Math.abs((exp - approxExp) / exp);
//    		if (isVeryDangerous(error)) continue;
//    		maxError = Math.max(error,maxError);
//    		sumError += error;
//    		numTrials++;
//    	}
//    	double avgError = sumError / numTrials;
//    	System.out.println("Avg error was: " + avgError);
//    	System.out.println("Max error was: " + maxError);
//    }
//    	else if (args[0].equals("-approxLog"))
//        {
//        	int numTrials = 0;
//        	double sumError = 0;
//        	double maxError = 0;
//        	double x = Double.MIN_VALUE; 
//        	while (x < Double.MAX_VALUE)
//        	{
//				//        		if (Math.abs(x - 1) < 0.3) continue;
//        		final double approxExp = approxLog(x);
//				final double exp = Math.log(x);
//    			double error = Math.abs((exp - approxExp) / exp);
//        		if (isVeryDangerous(error)) continue;
//        		maxError = Math.max(error,maxError);
//        		sumError += error;
//        		numTrials++;
//        		
//        		if (x < Double.MIN_VALUE * 1000000)
//					x *= 4;
//        		else x *= 1.0001;
//        	}
//        	double avgError = sumError / numTrials;
//        	System.out.println("Avg error was: " + avgError);
//        	System.out.println("Max error was: " + maxError);
//        	
//      
//    } else {
//      System.err.println("Unknown option: " + args[0]);
//    }
  }
  
  public static double noNaNDivide(double num, double denom)
	{
		return denom == 0.0 ? 0.0 : num / denom;
	}

  
	public static double approxLog(double val)
	{
    if (val < 0.0) return Double.NaN;
	  if (val == 0.0) return Double.NEGATIVE_INFINITY;
		double r = val - 1;
		if (Math.abs(r) < 0.3)
		{
			// use first few terms of taylor series
			
			final double rSquared = r * r;
			return r - rSquared / 2 + rSquared * r / 3;
		}
		final double x = (Double.doubleToLongBits(val) >> 32);
		return (x - 1072632447) / 1512775;

	}

	public static double approxExp(double val)
	{

		if (Math.abs(val) < 0.1) return 1 + val;
		final long tmp = (long) (1512775 * val + (1072693248 - 60801));
		return Double.longBitsToDouble(tmp << 32);

	}

	public static double approxPow(final double a, final double b)
	{
		final int tmp = (int) (Double.doubleToLongBits(a) >> 32);
		final int tmp2 = (int) (b * (tmp - 1072632447) + 1072632447);
		return Double.longBitsToDouble(((long) tmp2) << 32);
	}
	

	public static double logSubtract(double a, double b)
	{
		if (a > b)
		{
      // logA logB
      // (logA - logB) = (log
			return a + Math.log(1.0 - Math.exp(b - a));

		}
		else
		{
			return b + Math.log(-1.0 + Math.exp(a - b));
		}
	}

  public static double unsafeSubtract(double a, double b) {
    if (a == b) { // inf - inf (or -inf - -inf)
      return 0.0;
    }
    if (a == Double.NEGATIVE_INFINITY) {
      return a;
    }
    return a-b;
  }

  public static double unsafeAdd(double a, double b) {
    if (a == b) { // inf - inf (or -inf - -inf)
      return 0.0;
    }
    if (a == Double.POSITIVE_INFINITY) {
      return a;
    }
    return a+b;
  }

  public static <T> double logAdd(Counter<T> counts) {
    double[] arr = new double[counts.size()];
    int index = 0;
    for (Map.Entry<T,Double> entry : counts.entrySet()) {
      arr[index++] = entry.getValue();
    }
    return SloppyMath.logAdd(arr);
  }

//	public static double approxLogAdd(double a, double b)
//	{
//		
//		    final long tmp1 = (long) (1512775 * a + (1072693248 - 60801));
//		    double ea = Double.longBitsToDouble(tmp1 << 32);
//		    final long tmp2 = (long) (1512775 * b + (1072693248 - 60801));
//		    double eb = Double.longBitsToDouble(tmp2 << 32);
//		    
//		    final double x = (Double.doubleToLongBits(ea + eb) >> 32);
//		    return (x - 1072632447) / 1512775;
//		
//	}
  

}
