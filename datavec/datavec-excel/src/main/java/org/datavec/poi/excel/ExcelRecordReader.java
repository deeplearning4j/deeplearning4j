/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.poi.excel;

import org.apache.poi.ss.usermodel.*;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Excel record reader for loading rows of an excel spreadsheet
 * from multiple spreadsheets very similar to the
 * {@link org.datavec.api.records.reader.impl.csv.CSVRecordReader}
 *
 * Of note when you have multiple sheets, you must have the same number of
 * lines skipped at the top. For example, if you have a header as follows:
 * Header1 Header2 Header3
 *   1      2       3
 *   4     5        6
 *
 * Any other sheet you are trying to parse must also contain the
 * same number of header lines.
 *
 */
public class ExcelRecordReader extends FileRecordReader {
    //originally from CSVRecordReader
    private boolean skippedLines = false;
    protected int skipNumLines = 0;
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";

    private Iterator<Sheet> sheetIterator;
    private Iterator<Row> rows;
    // Create a DataFormatter to format and get each cell's value as String
    private DataFormatter dataFormatter = new DataFormatter();
    private Workbook currWorkBook;
    //we should ensure that the number of columns is consistent across all worksheets
    private int numColumns = -1;

    /**
     * Skip skipNumLines number of lines
     * @param skipNumLines the number of lines to skip
     */
    public ExcelRecordReader(int skipNumLines) {
        this.skipNumLines = skipNumLines;
    }



    public ExcelRecordReader() {
        this(0);
    }

    @Override
    public boolean hasNext() {
        if (!skipLines())
            throw new NoSuchElementException("No next element found!");
        return skipLines() && super.hasNext() ||
                sheetIterator != null && sheetIterator.hasNext()
                || rows != null && rows.hasNext();
    }


    private boolean skipLines() {
        if (!skippedLines && skipNumLines > 0) {
            for (int i = 0; i < skipNumLines; i++) {
                if (!super.hasNext()) {
                    return false;
                }
                super.next();
            }
            skippedLines = true;
        }
        return true;
    }

    @Override
    public List<Writable> next() {
        return nextRecord().getRecord();
    }

    @Override
    public Record nextRecord(){
        //start at top tracking rows
        if(rows != null && rows.hasNext()) {
            Row currRow = rows.next();
            List<Writable> ret = new ArrayList<>(currRow.getLastCellNum());
            for(Cell cell: currRow) {
                String cellValue = dataFormatter.formatCellValue(cell);
                ret.add(new Text(cellValue));
            }
            Record record = new org.datavec.api.records.impl.Record(ret,
                                    new RecordMetaDataIndex(
                                            currRow.getRowNum(),
                                            super.currentFile.toURI(),
                                            ExcelRecordReader.class));
            return record;
        }
        // next track sheets
        else if(sheetIterator != null && sheetIterator.hasNext()) {
            Sheet sheet = sheetIterator.next();
            rows = sheet.rowIterator();
            Row currRow = rows.next();
            Record record = new org.datavec.api.records.impl.Record(rowToRecord(currRow),
                                new RecordMetaDataIndex(
                                    currRow.getRowNum(),
                                    super.currentFile.toURI(),
                                    ExcelRecordReader.class));
            return record;

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
            Row currRow = rows.next();
            Record record = new org.datavec.api.records.impl.Record(rowToRecord(currRow),
                    new RecordMetaDataIndex(
                            currRow.getRowNum(),
                            super.currentFile.toURI(),
                            ExcelRecordReader.class));
            return record;

        } catch (Exception e) {
            throw new IllegalStateException("Error processing row",e);
        }

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES,0);
    }

    @Override
    public void reset() {
        super.reset();
        skippedLines = false;
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
