package org.datavec.dataframe.api;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Tests for BooleanColumn
 */
public class BooleanColumnTest {

  private final BooleanColumn column = BooleanColumn.create("Test");

  @Before
  public void setup() {
    column.add(false);
    column.add(false);
    column.add(false);
    column.add(false);
    column.add(true);
    column.add(true);
    column.add(false);
  }

  @Test
  public void testGetElements() throws Exception {
    assertEquals(7, column.size());
  }

  @Test
  public void testAddCell() throws Exception {
    column.add(true);
    assertEquals(8, column.size());

    // Add some other types and ensure that they're correctly truthy
    column.addCell("true");
    assertTrue(lastEntry());
    column.addCell("false");
    assertFalse(lastEntry());
    column.addCell("TRUE");
    assertTrue(lastEntry());
    column.addCell("FALSE");
    assertFalse(lastEntry());
    column.addCell("T");
    assertTrue(lastEntry());
    column.addCell("F");
    assertFalse(lastEntry());
    column.addCell("Y");
    assertTrue(lastEntry());
    column.addCell("N");
    assertFalse(lastEntry());
  }

  @Test
  public void testGetType() throws Exception {
    assertEquals("Boolean".toUpperCase(), column.type().name());
  }

  @Test
  public void testSummary() throws Exception {
    Table summary = column.summary();
    assertEquals(2, summary.columnCount());
    assertEquals(2, summary.rowCount());
    assertEquals("true", summary.get(0, 1));
    assertEquals("false", summary.get(0, 0));
    assertEquals("5", summary.get(1, 0));
    assertEquals("2", summary.get(1, 1));
  }

  @Test
  public void testCountUnique() throws Exception {
    int result = column.countUnique();
    assertEquals(2, result);
  }

  /**
   * Tests construction from a bitmap. The test uses the isFalse() method, which inverts the values in the column it's
   * invoked on, so the true false counts are the opposite of those in the original
   */
  @Test
  public void testBitmapConstructor() throws Exception {
    BooleanColumn bc = new BooleanColumn("Is false", column.isFalse(), column.size());
    Table summary = bc.summary();
    assertEquals(2, summary.columnCount());
    assertEquals(2, summary.rowCount());
    assertEquals("true", summary.get(0, 1));
    assertEquals("false", summary.get(0, 0));
    assertEquals("5", summary.get(1, 1));
    assertEquals("2", summary.get(1, 0));
  }

  @Test
  public void testConversionToInt() {
    int[] array = column.toIntArray();
    assertTrue(array[0] == 0);
    assertTrue(array[1] == 0);
    assertTrue(array[2] == 0);
    assertTrue(array[3] == 0);
    assertTrue(array[4] == 1);
    assertTrue(array[5] == 1);
    assertTrue(array[6] == 0);
  }

  /**
   * Returns true if the last item added to the column is true and false otherwise
   */
  private boolean lastEntry() {
    return column.get(column.size() - 1);
  }
}
