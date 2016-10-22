package org.datavec.dataframe.util;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.table.ViewGroup;
import org.junit.Test;

/**
 *
 */
public class DoubleArraysTest {

  @Test
  public void testTo2dArray() throws Exception {
    Table table = Table.createFromCsv("data/tornadoes_1950-2014.csv");
    ViewGroup viewGroup = table.splitOn(table.shortColumn("Scale"));
    int columnNuumber = table.columnIndex("Injuries");
    double[][] results = DoubleArrays.to2dArray(viewGroup, columnNuumber);
  }

}