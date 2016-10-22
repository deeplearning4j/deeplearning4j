package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.table.Relation;
import org.datavec.dataframe.util.Selection;
import com.google.common.base.Stopwatch;
import io.codearte.jfairy.Fairy;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.stat.StatUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

/**
 * Unit tests for the FloatColumn class
 */
public class FloatColumnTest {

  @Ignore
  @Test
  public void testApplyFilter() {

    Fairy fairy = Fairy.create();
    fairy.baseProducer().trueOrFalse();

    Table table = Table.create("t");
    FloatColumn floatColumn = new FloatColumn("test", 1_000_000_000);
    BooleanColumn booleanColumn = new BooleanColumn("bools", 1_000_000_000);
    table.addColumn(floatColumn);
    table.addColumn(booleanColumn);
    for (int i = 0; i < 1_000_000_000; i++) {
      floatColumn.add((float) Math.random());
      booleanColumn.add(fairy.baseProducer().trueOrFalse());
    }
    Stopwatch stopwatch = Stopwatch.createStarted();
    table.sortOn("test");
    System.out.println("Sort time in ms = " + stopwatch.elapsed(TimeUnit.MILLISECONDS));
    stopwatch.reset().start();
    System.out.println(floatColumn.summary().print());
    stopwatch.reset().start();
    floatColumn.isLessThan(.5f);
    System.out.println("Search time in ms = " + stopwatch.elapsed(TimeUnit.MILLISECONDS));
  }

  @Ignore
  @Test
  public void testSortAndApplyFilter1() {

    FloatColumn floatColumn = new FloatColumn("test", 1_000_000_000);
    for (int i = 0; i < 1_000_000_000; i++) {
      floatColumn.add((float) Math.random());
    }
    Stopwatch stopwatch = Stopwatch.createStarted();
    System.out.println(floatColumn.sum());
    System.out.println(stopwatch.elapsed(TimeUnit.MILLISECONDS));
    stopwatch.reset().start();
    floatColumn.sortAscending();
    System.out.println("Sort time in ms = " + stopwatch.elapsed(TimeUnit.MILLISECONDS));

    stopwatch.reset().start();
    floatColumn.isLessThan(.5f);
    System.out.println("Search time in ms = " + stopwatch.elapsed(TimeUnit.MILLISECONDS));
  }

  @Ignore
  @Test
  public void testSort1() throws Exception {
    FloatColumn floatColumn = new FloatColumn("test", 1_000_000_000);
    System.out.println("Adding floats to column");
    for (int i = 0; i < 1_000_000_000; i++) {
      floatColumn.add((float) Math.random());
    }
    System.out.println("Sorting");
    Stopwatch stopwatch = Stopwatch.createStarted();
    floatColumn.sortAscending();
    System.out.println("Sort time in ms = " + stopwatch.elapsed(TimeUnit.MILLISECONDS));

  }

  @Test
  public void testIsLessThan() {
    int size = 1_000_000;
    Relation table = Table.create("t");
    FloatColumn floatColumn = new FloatColumn("test", size);
    table.addColumn(floatColumn);
    for (int i = 0; i < size; i++) {
      floatColumn.add((float) Math.random());
    }
    Selection results = floatColumn.isLessThan(.5f);
    int count = 0;
    for (int i = 0; i < size; i++) {
      if (results.contains(i)) {
        count++;
      }
    }
    // Probabilistic answer.
    assertTrue(count < 575_000);
    assertTrue(count > 425_000);
  }

  @Test
  public void testIsGreaterThan() {
    int size = 1_000_000;
    Relation table = Table.create("t");
    FloatColumn floatColumn = new FloatColumn("test", size);
    table.addColumn(floatColumn);
    for (int i = 0; i < size; i++) {
      floatColumn.add((float) Math.random());
    }
    Selection results = floatColumn.isGreaterThan(.5f);

    int count = 0;
    for (int i = 0; i < size; i++) {
      if (results.contains(i)) {
        count++;
      }
    }
    // Probabilistic answer.
    assertTrue(count < 575_000);
    assertTrue(count > 425_000);
  }

  @Test
  public void testSort() {
    int records = 1_000_000;
    FloatColumn floatColumn = new FloatColumn("test", records);
    for (int i = 0; i < records; i++) {
      floatColumn.add((float) Math.random());
    }
    floatColumn.sortAscending();
    float last = Float.NEGATIVE_INFINITY;
    for (float n : floatColumn) {
      assertTrue(n >= last);
      last = n;
    }
    floatColumn.sortDescending();
    last = Float.POSITIVE_INFINITY;
    for (float n : floatColumn) {
      assertTrue(n <= last);
      last = n;
    }
    records = 10;
    floatColumn = new FloatColumn("test", records);
    for (int i = 0; i < records; i++) {
      floatColumn.add((float) Math.random());
    }
    floatColumn.sortDescending();
    last = Float.POSITIVE_INFINITY;
    for (float n : floatColumn) {
      assertTrue(n <= last);
      last = n;
    }
  }

  @Test
  public void testIsEqualTo() {

    Relation table = Table.create("t");
    FloatColumn floatColumn = new FloatColumn("test", 1_000_000);
    float[] floats = new float[1_000_000];
    table.addColumn(floatColumn);
    for (int i = 0; i < 1_000_000; i++) {
      float f = (float) Math.random();
      floatColumn.add(f);
      floats[i] = f;
    }
    Selection results;
    RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    for (int i = 0; i < 100; i++) { // pick a hundred values at random and see if we can find them
      float f = floats[randomDataGenerator.nextInt(0, 999_999)];
      results = floatColumn.isEqualTo(f);
      assertEquals(f, floatColumn.get(results.iterator().next()), .001);
    }
  }

  @Test
  public void testMaxAndMin() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    FloatArrayList floats1 = floats.top(50);
    FloatArrayList floats2 = floats.bottom(50);
    double[] doubles1 = new double[50];
    double[] doubles2 = new double[50];
    for (int i = 0; i < floats1.size(); i++) {
      doubles1[i] = floats1.getFloat(i);
    }
    for (int i = 0; i < floats2.size(); i++) {
      doubles2[i] = floats2.getFloat(i);
    }
    // the smallest item in the max set is >= the largest in the min set
    assertTrue(StatUtils.min(doubles1) >= StatUtils.max(doubles2));
  }

  @Test
  public void testRound() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.round();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testLogN() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.logN();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testLog10() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.log10();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testLog1p() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.log1p();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testAbs() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.abs();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testClear() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    assertFalse(floats.isEmpty());
    floats.clear();
    assertTrue(floats.isEmpty());
  }

  @Test
  public void testCountMissing() {
    FloatColumn floats = new FloatColumn("floats", 10);
    for (int i = 0; i < 10; i++) {
      floats.add(RandomUtils.nextFloat(0, 1_000));
    }
    assertEquals(0, floats.countMissing());
    floats.clear();
    for (int i = 0; i < 10; i++) {
      floats.add(FloatColumn.MISSING_VALUE);
    }
    assertEquals(10, floats.countMissing());
  }


  @Test
  public void testCountUnique() {
    FloatColumn floats = new FloatColumn("floats", 10);
    float[] uniques = {0.0f, 0.00000001f, -0.000001f, 92923.29340f, 24252,23442f, 2252,2342f};
    for (float unique : uniques) {
      floats.add(unique);
    }
    assertEquals(uniques.length, floats.countUnique());

    floats.clear();
    float[] notUniques = {0.0f, 0.00000001f, -0.000001f, 92923.29340f, 24252,23442f, 2252,2342f, 0f};

    for (float notUnique : notUniques) {
      floats.add(notUnique);
    }
    assertEquals(notUniques.length -1, floats.countUnique());
  }

  @Test
  public void testUnique() {
    FloatColumn floats = new FloatColumn("floats", 10);
    float[] uniques = {0.0f, 0.00000001f, -0.000001f, 92923.29340f, 24252,23442f, 2252,2342f};
    for (float unique : uniques) {
      floats.add(unique);
    }
    assertEquals(uniques.length, floats.unique().size());

    floats.clear();
    float[] notUniques = {0.0f, 0.00000001f, -0.000001f, 92923.29340f, 24252,23442f, 2252,2342f, 0f};

    for (float notUnique : notUniques) {
      floats.add(notUnique);
    }
    assertEquals(notUniques.length -1, floats.unique().size());
  }

  @Test
  public void testIsMissingAndIsNotMissing() {
    FloatColumn floats = new FloatColumn("floats", 10);
    for (int i = 0; i < 10; i++) {
      floats.add(RandomUtils.nextFloat(0, 1_000));
    }
    Assert.assertEquals(0, floats.isMissing().size());
    Assert.assertEquals(10, floats.isNotMissing().size());
    floats.clear();
    for (int i = 0; i < 10; i++) {
      floats.add(FloatColumn.MISSING_VALUE);
    }
    Assert.assertEquals(10, floats.isMissing().size());
    Assert.assertEquals(0, floats.isNotMissing().size());
  }

  @Test
  public void testEmptyCopy() {
    FloatColumn floats = new FloatColumn("floats", 100);
    String comment = "This is a comment";
    floats.setComment(comment);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    FloatColumn empty = floats.emptyCopy();
    assertTrue(empty.isEmpty());
    Assert.assertEquals(floats.name(), empty.name());

    //TODO(lwhite): Decide what gets copied in an empty copy
    //assertEquals(floats.comment(), empty.comment());
  }

  @Test
  public void testSize() {
    int size = 100;
    FloatColumn floats = new FloatColumn("floats", size);
    assertEquals(0, floats.size());
    for (int i = 0; i < size; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    assertEquals(size, floats.size());
    floats.clear();
    assertEquals(0, floats.size());
  }

  @Test
  public void testNeg() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    Column newFloats = floats.neg();
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void tesMod() {
    FloatColumn floats = new FloatColumn("floats", 100);
    FloatColumn otherFloats = new FloatColumn("otherFloats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
      otherFloats.add(floats.get(i) - 1.0f);
    }
    Column newFloats = floats.remainder(otherFloats);
    assertFalse(newFloats.isEmpty());
  }

  @Test
  public void testSquareAndSqrt() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }

    FloatColumn newFloats = floats.square();
    FloatColumn revert = newFloats.sqrt();
    for (int i = 0; i < floats.size(); i++) {
      assertEquals(floats.get(i), revert.get(i), 0.01);
    }
  }

  @Test
  public void testType() {
    FloatColumn floats = new FloatColumn("floats", 100);
    assertEquals(ColumnType.FLOAT, floats.type());
  }

  @Test
  public void testCubeAndCbrt() {
    FloatColumn floats = new FloatColumn("floats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
    }
    FloatColumn newFloats = floats.cube();
    FloatColumn revert = newFloats.cubeRoot();
    for (int i = 0; i < floats.size(); i++) {
      assertEquals(floats.get(i), revert.get(i), 0.01);
    }
  }

  // todo - question - does this test really test difference method?
  @Test
  public void testDifference() {
    FloatColumn floats = new FloatColumn("floats", 100);
    FloatColumn otherFloats = new FloatColumn("otherFloats", 100);
    for (int i = 0; i < 100; i++) {
      floats.add(RandomUtils.nextFloat(0, 10_000));
      otherFloats.add(floats.get(i) - 1.0f);
    }
    FloatColumn diff = floats.subtract(otherFloats);
    for (int i = 0; i < floats.size(); i++) {
      assertEquals(floats.get(i), otherFloats.get(i) + 1.0, 0.01);
    }
  }

  @Test
  public void testDifferencePositive() {
    float[] originalValues = new float[] {32,42,40,57,52};
    float[] expectedValues = new float[] {Float.NaN,10,-2,17,-5};

    FloatColumn initial = new FloatColumn("Test",originalValues.length);
    for (float value : originalValues) {
      initial.add(value);
    }
    FloatColumn difference =  initial.difference();
    assertEquals("Both sets of data should be the same size.", expectedValues.length, difference.size());
    for (int index = 0; index < difference.size(); index++ ) {
      float actual = difference.get(index);
      if (index == 0) {
         assertTrue("difference operation at index:" + index + " failed", Float.isNaN(actual));
      } else {
        assertEquals("difference operation at index:" + index + " failed", expectedValues[index], actual, 0);
      }
    }
  }

  @Test
  public void testDifferenceNegative() {
    float[] originalValues = new float[] {32,42,40,57,52};
    float[] expectedValues = new float[] {Float.MAX_VALUE,Float.MIN_VALUE,-12,117,5};

    FloatColumn initial = new FloatColumn("Test",originalValues.length);
    for (float value : originalValues) {
      initial.add(value);
    }
    FloatColumn difference =  initial.difference();
    assertEquals("Both sets of data should be the same size.", expectedValues.length, difference.size());
    for (int index = 0; index < difference.size(); index++ ) {
      float actual = difference.get(index);
      if (index == 0) {
        assertTrue("difference operation at index:" + index + " failed", Float.isNaN(actual));
      } else {
        assertNotEquals("difference operation at index:" + index + " failed", expectedValues[index], actual, 0.0);
      }
    }
  }

}