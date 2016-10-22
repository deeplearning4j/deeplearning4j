package org.datavec.dataframe.util;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

/**
 *
 */
public class Stats {

  private String name;

  long n;
  double sum;
  double mean;
  double min;
  double max;
  double variance;
  double standardDeviation;
  double geometricMean;
  double quadraticMean;
  double secondMoment;
  double populationVariance;
  double sumOfLogs;
  double sumOfSquares;

  public static Stats create(final FloatColumn values) {
    SummaryStatistics summaryStatistics = new SummaryStatistics();
    for (float f : values) {
      summaryStatistics.addValue(f);
    }
    return getStats(values, summaryStatistics);
  }
  public static Stats create(final IntColumn ints) {
    FloatColumn values = FloatColumn.create(ints.name(), ints.toFloatArray());
    return create(values);
  }

  public static Stats create(final ShortColumn ints) {
    FloatColumn values = FloatColumn.create(ints.name(), ints.toFloatArray());
    return create(values);
  }

  public static Stats create(final LongColumn ints) {
    FloatColumn values = FloatColumn.create(ints.name(), ints.toFloatArray());
    return create(values);
  }

  public Stats(String name) {
    this.name = name;
  }

  public float range() {
    return (float) (max - min);
  }

  public float standardDeviation() {
    return (float) standardDeviation;
  }

  public long n() {
    return n;
  }

  public float mean() {
    return (float) (sum / (double) n);
  }

  public float min() {
    return (float) min;
  }

  public float max() {
    return (float) max;
  }

  public float sum() {
    return (float) sum;
  }

  public float variance() {
    return (float) variance;
  }

  public float sumOfSquares() {
    return (float) sumOfSquares;
  }

  public float populationVariance() {
    return (float) populationVariance;
  }

  public float sumOfLogs() {
    return (float) sumOfLogs;
  }

  public float geometricMean() {
    return (float) geometricMean;
  }

  public float quadraticMean() {
    return (float) quadraticMean;
  }

  public float secondMoment() {
    return (float) secondMoment;
  }

  public Table asTable() {
    Table t = Table.create(name);
    CategoryColumn measure = CategoryColumn.create("Measure");
    FloatColumn value = FloatColumn.create("Value");
    t.addColumn(measure);
    t.addColumn(value);

    measure.add("n");
    value.add(n);

    measure.add("sum");
    value.add(sum());

    measure.add("Mean");
    value.add(mean());

    measure.add("Min");
    value.add(min());

    measure.add("Max");
    value.add(max());

    measure.add("Range");
    value.add(range());

    measure.add("Variance");
    value.add(variance());

    measure.add("Std. Dev");
    value.add(standardDeviation());

    return t;
  }

  public Table asTableComplete() {
    Table t = asTable();

    CategoryColumn measure = t.categoryColumn("Measure");
    FloatColumn value = t.floatColumn("Value");

    measure.add("Sum of Squares");
    value.add(sumOfSquares());

    measure.add("Sum of Logs");
    value.add(sumOfLogs());

    measure.add("Population Variance");
    value.add(populationVariance());

    measure.add("Geometric Mean");
    value.add(geometricMean());

    measure.add("Quadratic Mean");
    value.add(quadraticMean());

    measure.add("Second Moment");
    value.add(secondMoment());

    return t;
  }

  private static Stats getStats(FloatColumn values, SummaryStatistics summaryStatistics) {
    Stats stats = new Stats("Column: " + values.name());
    stats.min = (float) summaryStatistics.getMin();
    stats.max = (float) summaryStatistics.getMax();
    stats.n = summaryStatistics.getN();
    stats.sum = summaryStatistics.getSum();
    stats.variance = summaryStatistics.getVariance();
    stats.populationVariance = summaryStatistics.getPopulationVariance();
    stats.quadraticMean = summaryStatistics.getQuadraticMean();
    stats.geometricMean = summaryStatistics.getGeometricMean();
    stats.mean = summaryStatistics.getMean();
    stats.standardDeviation = summaryStatistics.getStandardDeviation();
    stats.sumOfLogs = summaryStatistics.getSumOfLogs();
    stats.sumOfSquares = summaryStatistics.getSumsq();
    stats.secondMoment = summaryStatistics.getSecondMoment();
    return stats;
  }
}
