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

package org.datavec.api.transform.schema;

import au.com.bytecode.opencsv.CSVParser;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * If passed a CSV file that contains a header and a single row of sample data, it will return
 * a Schema.
 *
 * Only Double, Integer, Long, and String types are supported. If no number type can be inferred,
 * the field type will become the default type. Note that if your column is actually categorical but
 * is represented as a number, you will need to do additional transformation. Also, if your sample
 * field is blank/null, it will also become the default type.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class InferredSchema {
    protected Schema.Builder schemaBuilder;
    protected String pathToCsv;
    protected DataType defaultType;
    protected String quote;

    private CSVParser csvParser = new CSVParser();

    public InferredSchema(String pathToCsv) {
        this.pathToCsv = pathToCsv;
        this.defaultType = DataType.valueOf("STRING");
    }

    public InferredSchema(String pathToCsv, DataType defaultType) {
        this.pathToCsv = pathToCsv;
        this.defaultType = defaultType;
    }

    public InferredSchema(String pathToCsv, DataType defaultType, char delimiter) {
        this.pathToCsv = pathToCsv;
        this.defaultType = defaultType;
        this.csvParser = new CSVParser(delimiter);
    }

    public InferredSchema(String pathToCsv, DataType defaultType, char delimiter, char quote) {
        this.pathToCsv = pathToCsv;
        this.defaultType = defaultType;
        this.csvParser = new CSVParser(delimiter, quote);
    }

    public InferredSchema(String pathToCsv, DataType defaultType, char delimiter, char quote, char escape) {
        this.pathToCsv = pathToCsv;
        this.defaultType = defaultType;
        this.csvParser = new CSVParser(delimiter, quote, escape);
    }

    public Schema build() throws IOException {
        List<String> headersAndRows = null;
        this.schemaBuilder = new Schema.Builder();

        try {
            headersAndRows = FileUtils.readLines(new File(pathToCsv));
        } catch (IOException e) {
            log.error("An error occurred while parsing sample CSV for schema", e);
        }
        List<String> headers = parseLine(headersAndRows.get(0));
        List<String> samples = parseLine(headersAndRows.get(1));

        if(headers.size() != samples.size())
            throw new IllegalStateException("CSV headers length does not match number of sample columns. " +
                    "Please check that your CSV is valid, or check the delimiter used to parse the CSV.");

        for(int i = 0; i < headers.size(); i++) {
            inferAndAddType(schemaBuilder, headers.get(i), samples.get(i));
        }
        return schemaBuilder.build();
    }

    private Schema.Builder inferAndAddType(Schema.Builder builder, String header, String sample) {
        if(isParsableAsDouble(sample)) addOn(builder, header, DataType.DOUBLE);
        else if(isParsableAsInteger(sample)) addOn(builder, header, DataType.INTEGER);
        else if(isParsableAsLong(sample)) addOn(builder, header, DataType.LONG);
        else addOn(builder, header, defaultType);

        return schemaBuilder;
    }

    private static Schema.Builder addOn(Schema.Builder builder, String columnName, DataType columnType) {
        switch (columnType) {
            case DOUBLE:
                return builder.addColumnDouble(columnName, null, null, false, false); //no nans/inf
            case INTEGER:
                return builder.addColumnInteger(columnName);
            case LONG:
                return builder.addColumnLong(columnName);
            case STRING:
                return builder.addColumnString(columnName);
            default:
                throw new IllegalArgumentException("Schema inputs have to be string, integer or double");
        }
    }

    private List<String> parseLine(String line) throws IOException {
        String[] split = csvParser.parseLine(line);
        ArrayList ret = new ArrayList();
        String[] var4 = split;
        int var5 = split.length;

        for(int var6 = 0; var6 < var5; ++var6) {
            String s = var4[var6];
            if(this.quote != null && s.startsWith(this.quote) && s.endsWith(this.quote)) {
                int n = this.quote.length();
                s = s.substring(n, s.length() - n).replace(this.quote + this.quote, this.quote);
            }
            ret.add(s);
        }

        return ret;
    }

    private static boolean isParsableAsLong(final String s) {
        try {
            Long.valueOf(s);
            return true;
        } catch (NumberFormatException numberFormatException) {
            return false;
        }
    }

    private static boolean isParsableAsInteger(final String s) {
        try {
            Integer.valueOf(s);
            return true;
        } catch (NumberFormatException numberFormatException) {
            return false;
        }
    }


    private static boolean isParsableAsDouble(final String s) {
        try {
            Double.valueOf(s);
            return true;
        } catch (NumberFormatException numberFormatException) {
            return false;
        }
    }

    private enum DataType {
        STRING,
        INTEGER,
        DOUBLE,
        LONG
    }
}