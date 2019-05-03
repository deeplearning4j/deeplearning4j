package org.deeplearning4j.plot;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class StatUtils {

    public static INDArray binary_search_perplexity(INDArray affinities, double desired_perplexity) {
        final int STEPS = 100;
        final double EPSILON_DBL = 1e-8;
        final double PERPLEXITY_TOLERANCE = 1e-5;

        double beta = 1.0;
        double sum_disti_Pi = 0.0;
        double beta_sum = 0.0;

        double desired_entropy = FastMath.log(desired_perplexity);
        INDArray P = Nd4j.create(affinities.rows(), affinities.rows());

        for (int i = 0; i < affinities.rows(); ++i) {

            double beta_min = -Double.NaN;
            double beta_max = Double.NaN;


            for (int k = 0; k < STEPS; ++k) {

                double sum_Pi = 0.0;
                for (int j = 0; j < affinities.columns(); ++j) {
                    if (j != i) {

                        double value = FastMath.exp(-affinities.getRow(i).getDouble(j) * beta);
                        P.putScalar(i, j, value);
                        sum_Pi += value;
                    }
                }

                 if (sum_Pi == 0.0) {
                   sum_Pi = EPSILON_DBL;
                   sum_disti_Pi = 0.0;
                 }

                for (int j = 0; j < affinities.columns(); ++j) {
                    double value = P.getRow(i).getDouble(j) / sum_Pi;
                    P.putScalar(i,j, value);
                    sum_disti_Pi += affinities.getRow(i).getDouble(j) * value;
                }

                double entropy = FastMath.log(sum_Pi) + beta * sum_disti_Pi;
                double entropy_diff = entropy - desired_entropy;

                if (FastMath.abs(entropy_diff) <= PERPLEXITY_TOLERANCE) {
                    break;
                }

                if (entropy_diff > 0.0) {
                    beta_min = beta;
                    if (beta_max == Double.NaN)
                        beta *= 2.0;
                    else
                        beta = (beta + beta_max) / 2.0;
                }
                else {
                    beta_max = beta;
                    if (beta_min == -Double.NaN)
                        beta /= 2.0;
                    else
                        beta = (beta + beta_min) / 2.0;
                }
            }
            beta_sum += beta;
        }
        return P;
    }
}
