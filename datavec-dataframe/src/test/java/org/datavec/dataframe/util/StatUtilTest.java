package org.datavec.dataframe.util;

import org.datavec.dataframe.api.FloatColumn;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 *
 */
public class StatUtilTest {

  @Test
  public void testSum() {
    Random random = new Random();
    float sum = 0.0f;
    FloatColumn column = FloatColumn.create("c1");
    for (int i = 0; i < 100; i++) {
      float f = random.nextFloat();
      column.add(f);
      sum += f;
    }
    assertEquals(sum, column.sum(), 0.01f);
  }

  @Test
  public void testMin() {
    Random random = new Random();
    float min = Float.MAX_VALUE;
    FloatColumn column = FloatColumn.create("c1");
    for (int i = 0; i < 100; i++) {
      float f = random.nextFloat();
      column.add(f);
      if (min > f) {
        min = f;
      }
    }
    assertEquals(min, column.min(), 0.01f);
  }

  @Test
  public void testMax() {
    Random random = new Random();
    float max = Float.MIN_VALUE;
    FloatColumn column = FloatColumn.create("c1");
    for (int i = 0; i < 100; i++) {
      float f = random.nextFloat();
      column.add(f);
      if (max < f) {
        max = f;
      }
    }
    assertEquals(max, column.max(), 0.01f);
  }

  @Test
  public void testStats() {
    Random random = new Random();
    // assertEquals(sum, column.sum(), 0.01f);

  }
}