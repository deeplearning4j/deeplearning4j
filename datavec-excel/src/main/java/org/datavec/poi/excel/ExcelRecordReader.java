package org.datavec.poi.excel;

import org.apache.poi.ss.usermodel.*;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ExcelRecordReader extends FileRecordReader {
    private Iterator<Sheet> sheetIterator;
    private Iterator<Row> rows;
    // Create a DataFormatter to format and get each cell's value as String
    private DataFormatter dataFormatter = new DataFormatter();
    private Workbook currWorkBook;
    //we should ensure that the number of columns is consistent across all worksheets
    private int numColumns = -1;
    @Override
    public boolean hasNext() {
        return super.hasNext() ||
                sheetIterator != null && sheetIterator.hasNext()
                || rows != null && rows.hasNext();
    }

    @Override
    public List<Writable> next() {
        //start at top tracking rows
        if(rows != null && rows.hasNext()) {
            Row currRow = rows.next();
            List<Writable> ret = new ArrayList<>(currRow.getLastCellNum());
            for(Cell cell: currRow) {
                String cellValue = dataFormatter.formatCellValue(cell);
                ret.add(new Text(cellValue));
            }

            return ret;
        }
        // next track sheets
        else if(sheetIterator != null && sheetIterator.hasNext()) {
            Sheet sheet = sheetIterator.next();
            rows = sheet.rowIterator();
            return rowToRecord(rows.next());
        }


        //finally extract workbooks from files and iterate over those starting again at top
        File nextFile = super.nextFile();
        // Creating a Workbook from an Excel file (.xls or .xlsx)
        try {
            if(currWorkBook != null) {
                currWorkBook.close();
            }

            this.currWorkBook = WorkbookFactory.create(nextFile);
            this.sheetIterator = currWorkBook.sheetIterator();
            Sheet sheet = sheetIterator.next();
            rows = sheet.rowIterator();
            return rowToRecord(rows.next());

        } catch (Exception e) {
            throw new IllegalStateException("Error processing row",e);
        }

    }


    private List<Writable> rowToRecord(Row currRow) {
        if(numColumns < 0) {
            numColumns = currRow.getLastCellNum();
        }

        if(currRow.getLastCellNum() != numColumns) {
            throw new IllegalStateException("Invalid number of columns for row. First number of columns found was " + numColumns + " but row " + currRow.getRowNum() + " was " + currRow.getLastCellNum());
        }

        List<Writable> ret = new ArrayList<>(currRow.getLastCellNum());
        for(Cell cell: currRow) {
            String cellValue = dataFormatter.formatCellValue(cell);
            switch(cell.getCellTypeEnum()) {
                case BLANK: ret.add(new Text("")); break;
                case STRING: ret.add(new Text("")); break;
                case BOOLEAN: ret.add(new BooleanWritable(Boolean.valueOf(cellValue))); break;
                case NUMERIC: ret.add(new DoubleWritable(Double.parseDouble(cellValue))); break;
                default: ret.add(new Text(cellValue));
            }
        }

        return ret;

    }


}
