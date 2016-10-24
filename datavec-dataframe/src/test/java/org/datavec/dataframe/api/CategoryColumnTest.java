package org.datavec.dataframe.api;

import org.datavec.dataframe.TestDataUtil;
import org.datavec.dataframe.util.Selection;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 *
 */
public class CategoryColumnTest {

  private CategoryColumn column = CategoryColumn.create("testing");

  @Before
  public void setUp() throws Exception {
    column.add("Value 1");
    column.add("Value 2");
    column.add("Value 3");
    column.add("Value 4");
  }

  @Test
  public void testDefaultReturnValue() {
    assertEquals(-1, column.dictionaryMap().get("test"));
  }

  @Test
  public void testType() {
    assertEquals(ColumnType.CATEGORY, column.type());
  }

  @Test
  public void testGetString() {
    assertEquals("Value 2", column.getString(1));
  }

  @Test
  public void testSize() {
    assertEquals(4, column.size());
  }

  @Test
  public void testGetDummies() {
    List<BooleanColumn> dummies = column.getDummies();
    assertEquals(4, dummies.size());
  }

  @Test
  public void testToString() {
    assertEquals("Category column: testing", column.toString());
  }

  @Test
  public void testMax() {
    CategoryColumn categoryColumn = CategoryColumn.create("US States");
    categoryColumn.addAll(TestDataUtil.usStates());
    assertTrue("Wyoming".equals(categoryColumn.top(5).get(0)));
  }

  @Test
  public void testMin() {
    CategoryColumn categoryColumn = CategoryColumn.create("US States");
    categoryColumn.addAll(TestDataUtil.usStates());
    assertTrue("Alabama".equals(categoryColumn.bottom(5).get(0)));
  }

  @Test
  public void testStartsWith() {
    CategoryColumn categoryColumn = CategoryColumn.create("US States");
    categoryColumn.addAll(TestDataUtil.usStates());
    Selection selection = categoryColumn.startsWith("A");
    assertEquals("Alabama", categoryColumn.get(selection.get(0)));
    assertEquals("Alaska", categoryColumn.get(selection.get(1)));
    assertEquals("Arizona", categoryColumn.get(selection.get(2)));
    assertEquals("Arkansas", categoryColumn.get(selection.get(3)));

    selection = categoryColumn.startsWith("T");
    assertEquals("Tennessee", categoryColumn.get(selection.get(0)));
    assertEquals("Texas", categoryColumn.get(selection.get(1)));
  }

  @Test
  public void testIsIn() {
    CategoryColumn categoryColumn = CategoryColumn.create("US States");
    categoryColumn.addAll(TestDataUtil.usStates());
    Selection selection = categoryColumn.isIn("Alabama", "Texas");
    assertEquals("Alabama", categoryColumn.get(selection.get(0)));
    assertEquals("Texas", categoryColumn.get(selection.get(1)));
    assertEquals(2, selection.size());
  }
}