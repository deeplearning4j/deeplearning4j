package org.datavec.dataframe.api;

import org.datavec.dataframe.filtering.Filter;
import org.datavec.dataframe.filtering.IntPredicate;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.junit.Before;
import org.junit.Test;

import static org.datavec.dataframe.api.QueryHelper.column;
import static org.junit.Assert.*;
/**
 *  Tests for int columns
 */
public class IntColumnTest {

  private IntColumn intColumn;

  @Before
  public void setUp() throws Exception {
    intColumn = new IntColumn("t1");
  }

  @Test
  public void testSum() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(1);
    }
    assertEquals(100, intColumn.sum());
  }

  @Test
  public void testMin() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(0.0, intColumn.min(), .001);
  }

  @Test
  public void testMax() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(99, intColumn.max(), .001);
  }

  @Test
  public void testIsLessThan() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(50, intColumn.isLessThan(50).size());
  }

  @Test
  public void testIsGreaterThan() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(49, intColumn.isGreaterThan(50).size());
  }

  @Test
  public void testIsGreaterThanOrEqualTo() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(50, intColumn.isGreaterThanOrEqualTo(50).size());
    assertEquals(50, intColumn.isGreaterThanOrEqualTo(50).get(0));
  }

  @Test
  public void testIsLessThanOrEqualTo() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(51, intColumn.isLessThanOrEqualTo(50).size());
    assertEquals(49, intColumn.isLessThanOrEqualTo(50).get(49));
  }

  @Test
  public void testIsEqualTo() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    assertEquals(1, intColumn.isEqualTo(10).size());
  }

  @Test
  public void testPercents() {
    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }
    FloatColumn floatColumn = intColumn.asRatio();
    assertEquals(1.0, floatColumn.sum(), 0.1);
  }

  @Test
  public void testSelectIf() {

    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }

    IntPredicate predicate = value -> value < 10;
    IntColumn column1 = intColumn.selectIf(predicate);
    assertEquals(10, column1.size());
    for (int i = 0; i < 10; i++) {
      assertTrue(column1.get(i) < 10);
    }
  }

  @Test
  public void testSelect() {

    for (int i = 0; i < 100; i++) {
      intColumn.add(i);
    }

    IntPredicate predicate = value -> value < 10;
    IntColumn column1 = intColumn.selectIf(predicate);
    assertEquals(10, column1.size());

    IntColumn column2 = intColumn.select(intColumn.select(predicate));
    assertEquals(10, column2.size());
    for (int i = 0; i < 10; i++) {
      assertTrue(column1.get(i) < 10);
    }
    for (int i = 0; i < 10; i++) {
      assertTrue(column2.get(i) < 10);
    }
  }

  @Test
  public void testDifference() {
    int[] originalValues = new int[] {32,42,40,57,52};
    int[] expectedValues = new int[] {IntColumn.MISSING_VALUE,10,-2,17,-5};

    IntColumn initial = new IntColumn("Test", originalValues.length);
    for (int value : originalValues) {
      initial.add(value);
    }
    IntColumn difference = initial.difference();
    assertEquals("Both sets of data should be the same size.", expectedValues.length, difference.size());
    for (int index = 0; index < difference.size(); index++ ) {
      int actual = difference.get(index);
      assertEquals("difference operation at index:" + index + " failed",  expectedValues[index], actual);
    }
  }

  @Test
  public void testIntIsIn() {
    int[] originalValues = new int[]{32, 42, 40, 57, 52, -2};
    int[] inValues = new int[]{10, -2, 57, -5};
    IntColumn inColumn = IntColumn.create("In", new IntArrayList(inValues));

    IntColumn initial = new IntColumn("Test", originalValues.length);
    Table t = Table.create("t", initial);

    for (int value : originalValues) {
      initial.add(value);
    }

    Filter filter =  column("Test").isIn(inColumn);
    Table result = t.selectWhere(filter);
    System.out.println(result.print());
  }

  @Test
  public void testDivide() {
    int[] originalValues = new int[]{32, 42, 40, 57, 52, -2};
    IntColumn originals = IntColumn.create("Originals", new IntArrayList(originalValues));

    Table t = Table.create("t", originals);

    FloatColumn divided = originals.divide(3);
    System.out.println(divided.print());
  }

  @Test
  public void testDivide2() {
    int[] originalValues = new int[]{32, 42, 40, 57, 52, -2};
    IntColumn originals = IntColumn.create("Originals", new IntArrayList(originalValues));

    Table t = Table.create("t", originals);

    FloatColumn divided = originals.divide(3.3);
    System.out.println(divided.print());
  }
}