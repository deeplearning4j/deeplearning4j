package org.datavec.dataframe.examples;

import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.Table;
import it.unimi.dsi.fastutil.floats.FloatArrayList;

/**
 * Usage example using a Tornado dataset
 */
public class ServiceExample {

  public static void main(String[] args) throws Exception {

    Table ops = Table.create("data/operations.csv");

    out(ops.structure().print());

    out(ops.print());

    DateTimeColumn start = ops.dateColumn("Date").atTime(ops.timeColumn("Start"));
    DateTimeColumn end = ops.dateColumn("Date").atTime(ops.timeColumn("End"));

    for (int row : ops) {
      if (ops.timeColumn("End").get(row).isBefore(ops.timeColumn("Start").get(row))) {
        end.get(row).plusDays(1);
      }
    }

    // Calc duration
    LongColumn duration = start.differenceInSeconds(end);
    ops.addColumn(duration);
    duration.setName("Duration");

    out(ops.print());

    Table q2_429_assembly = ops.selectWhere(
          allOf
              (column("date").isInQ2(),
              (column("SKU").startsWith("429")),
              (column("Operation").isEqualTo("Assembly"))));

    Table durationByFacilityAndShift = q2_429_assembly.median("Duration").by("Facility", "Shift");
    // TODO(lwhite): We need a top() method that can be used to return the top table rows
    FloatArrayList tops = durationByFacilityAndShift.floatColumn("Median").top(5);

    out(durationByFacilityAndShift.print());

    durationByFacilityAndShift.exportToCsv("/tmp/durationByFacilityAndShift.csv");
  }

  private static void out(Object obj) {
    System.out.println(String.valueOf(obj));
  }

  private static void out() {
    System.out.println("");
  }

}