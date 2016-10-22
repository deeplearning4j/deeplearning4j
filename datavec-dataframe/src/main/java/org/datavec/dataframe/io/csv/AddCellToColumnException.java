package org.datavec.dataframe.io.csv;

import java.io.PrintStream;

/**
 * This Exception wraps another Exception thrown while adding a cell to a column.
 * <p/>
 * The methods of this exception allow the causing Exception, row number,
 * column index, columnNames and line to be retrieved.
 * <p/>
 * The dumpRow method allows the row in question to be printed to a
 * a PrintStream such as System.out
 */
public class AddCellToColumnException extends RuntimeException {

  /**
   * The index of the column that threw the Exception
   */
  private final int columnIndex;

  /**
   * The number of the row that caused the exception to be thrown
   */
  private final long rowNumber;

  /**
   * The column names stored as an array
   */
  private final String[] columnNames;

  /**
   * The original line that caused the Exception
   */
  private final String[] line;

  /**
   * Creates a new instance of this Exception
   * @param e The Exceeption that caused adding to fail
   * @param columnIndex The index of the column that threw the Exception
   * @param rowNumber The number of the row that caused the Exception to be thrown
   * @param columnNames The column names stored as an array
   * @param line The original line that caused the Exception
   */
  public AddCellToColumnException(Exception e, int columnIndex, long rowNumber, String[] columnNames, String[] line) {
    super("Error while addding cell from row " + rowNumber + " and column " + columnNames[columnIndex] + "(position:" + columnIndex + "): " + e.getMessage(), e);
    this.columnIndex = columnIndex;
    this.rowNumber = rowNumber;
    this.columnNames = columnNames;
    this.line = line;
  }

  /**
   * Returns the index of the column that threw the Exception
   */
  public int getColumnIndex() {
    return columnIndex;
  }

  /**
   * Returns the number of the row that caused the Exception to be thrown
   */
  public long getRowNumber() {
    return rowNumber;
  }

  /**
   * Returns the column names array
   */
  public String[] getColumnNames() {
    return columnNames;
  }

  /**
   * Returns the name of the column that caused the Exception
   */
  public String getColumnName() {
    return columnNames[columnIndex];
  }

  /**
   * Dumps to a PrintStream the information relative to the row that caused the problem
   * @param out The PrintStream to output to
   */
  public void dumpRow(PrintStream out) {
    for (int i = 0; i < columnNames.length; i++) {
      out.print("Column ");
      out.print(i);
      out.print(" ");
      out.print(columnNames[i]);
      out.print(" : ");
      try {
        out.println(line[i]);
      } catch (ArrayIndexOutOfBoundsException aioobe) {
        out.println("Unable to get cell "+i+ " of this line");
      }
    }
  }
}
