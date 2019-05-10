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

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.writable.*;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

public class ExcelRecordWriter extends FileRecordWriter {

    public final static String WORKSHEET_NAME = "org.datavec.poi.excel.worksheet.name";
    public final static String FILE_TYPE = "org.datavec.poi.excel.type";


    public final static String DEFAULT_FILE_TYPE = "xlsx";
    public final static String DEFAULT_WORKSHEET_NAME = "datavec-worksheet";


    private String workBookName = DEFAULT_WORKSHEET_NAME;
    private String fileTypeToUse = DEFAULT_FILE_TYPE;

    private Sheet sheet;
    private Workbook workbook;



    private void createRow(int rowNum,int numCols,List<Writable> value) {
        // Create a Row
        Row headerRow = sheet.createRow(rowNum);
        int col = 0;
        for(Writable writable : value) {
            // Creating cells
            Cell cell = headerRow.createCell(col++);
            setValueForCell(cell,writable);


        }

        // Resize all columns to fit the content size
        for(int i = 0; i < numCols; i++) {
            sheet.autoSizeColumn(i);
        }
    }

    private void setValueForCell(Cell cell,Writable value) {
        if(value instanceof DoubleWritable || value instanceof LongWritable || value instanceof FloatWritable || value instanceof IntWritable) {
            cell.setCellValue(value.toDouble());
        }
        else if(value instanceof BooleanWritable) {
            cell.setCellValue(((BooleanWritable) value).get());
        }
        else if(value instanceof Text) {
            cell.setCellValue(value.toString());
        }

    }


    @Override
    public boolean supportsBatch() {
        return true;
    }

    @Override
    public void initialize(InputSplit inputSplit, Partitioner partitioner) throws Exception {
        this.conf = new Configuration();
        this.partitioner = partitioner;
        partitioner.init(inputSplit);
        out = new DataOutputStream(partitioner.currentOutputStream());
        initPoi();


    }

    private void initPoi()  {
        if(fileTypeToUse.equals("xlsx"))
            workbook = new XSSFWorkbook();
        else {
            //xls
            workbook = new HSSFWorkbook();
        }

        this.sheet = workbook.createSheet(workBookName);


    }

    @Override
    public void initialize(Configuration configuration, InputSplit split, Partitioner partitioner) throws Exception {
        this.workBookName = configuration.get(WORKSHEET_NAME,DEFAULT_WORKSHEET_NAME);
        this.fileTypeToUse = configuration.get(FILE_TYPE,DEFAULT_FILE_TYPE);
        this.conf = configuration;
        partitioner.init(split);
        out = new DataOutputStream(partitioner.currentOutputStream());
        initPoi();
    }

    @Override
    public PartitionMetaData write(List<Writable> record) throws IOException {
        createRow(partitioner.numRecordsWritten(),record.size(),record);
        reinitIfNecessary();
        return PartitionMetaData.builder().numRecordsUpdated(1).build();
    }

    @Override
    public PartitionMetaData writeBatch(List<List<Writable>> batch) throws IOException {
        int numSoFar = 0;
        for (List<Writable> record : batch) {
            createRow(partitioner.numRecordsWritten() + numSoFar, record.size(), record);
            reinitIfNecessary();
            numSoFar++;
        }

        return PartitionMetaData.builder().numRecordsUpdated(batch.size()).build();
    }

    private void reinitIfNecessary() throws IOException {
        if(partitioner.needsNewPartition()) {
            workbook.write(out);
            out.flush();
            out.close();
            workbook.close();
            initPoi();
            this.out = new DataOutputStream(partitioner.openNewStream());
        }
    }

    @Override
    public void close() {
        if(workbook != null) {
            try {
                if(out != null) {
                    workbook.write(out);
                    out.flush();
                    out.close();
                }

                workbook.close();

            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }

}
