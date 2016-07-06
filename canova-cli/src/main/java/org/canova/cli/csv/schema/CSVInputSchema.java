/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.cli.csv.schema;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.common.base.Strings;
import org.apache.commons.math3.util.Pair;
import org.canova.cli.csv.schema.CSVSchemaColumn.TransformType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
	purpose: to parse and represent the input schema + column transforms of CSV data to vectorize
*/
public class CSVInputSchema {

  private static final Logger log = LoggerFactory.getLogger(CSVInputSchema.class);

	public String relation = "";
	public String delimiter = "";
	private boolean hasComputedStats = false;

	// columns: { columnName, column Schema }
	private Map<String, CSVSchemaColumn> columnSchemas = new LinkedHashMap<>();

	public CSVSchemaColumn getColumnSchemaByName( String colName ) {

		return this.columnSchemas.get(colName);

	}

	public Map<String, CSVSchemaColumn> getColumnSchemas() {
		return this.columnSchemas;
	}

	private boolean validateRelationLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateDelimiterLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateAttributeLine( String[] lineParts ) {
		
		// first check that we have enough parts on the line
		
		if ( lineParts.length != 4 ) {
			return false;
		}
		
		// now check for combinations of { COLUMNTYPE, TRANSFORM } that we dont support
		
		CSVSchemaColumn colValue = this.parseColumnSchemaFromAttribute( lineParts );
		
		
		// 1. Unsupported: { NUMERIC + LABEL }
		
		//if (colValue.columnType == CSVSchemaColumn.ColumnType.NUMERIC && colValue.transform == CSVSchemaColumn.TransformType.LABEL) { 
		//	return false;
		//}
		

		// 2. Unsupported: { NOMINAL + BINARIZE }

		if (colValue.columnType == CSVSchemaColumn.ColumnType.NOMINAL && colValue.transform == CSVSchemaColumn.TransformType.BINARIZE) { 
			return false;
		}


		// 3. Unsupported: { DATE + anything } --- date columns arent finished yet!
		
		if (colValue.columnType == CSVSchemaColumn.ColumnType.DATE ) { 
			return false;
		}
		

		
		return true;
	}

	private boolean validateSchemaLine( String line ) {

		String lineCondensed = line.trim().replaceAll(" +", " ");
		String[] parts = lineCondensed.split(" ");

		if ( parts[ 0 ].toLowerCase().equals("@relation") ) {

			return this.validateRelationLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@delimiter") ) {

			return this.validateDelimiterLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			return this.validateAttributeLine(parts);

		} else if ( parts[ 0 ].trim().equals("") ) {

			return true;

		} else {

			// bad schema line
			log.error("Line attribute matched no known attribute in schema! --- {}", line);
			return false;

		}


		//return true;

	}

	private String parseRelationInformation(String[] parts) {

		return parts[1];

	}

	private String parseDelimiter(String[] parts) {

		return parts[1];

	}

	/**
	 * parse out lines like:
	 * 		@ATTRIBUTE sepallength  NUMERIC   !COPY
	 *
	 * @param parts
	 * @return
	 */
	private CSVSchemaColumn parseColumnSchemaFromAttribute( String[] parts ) {

		String columnName = parts[1];
		String columnType = parts[2];
		String columnTransform = parts[3];

		CSVSchemaColumn.ColumnType colTypeEnum =
        CSVSchemaColumn.ColumnType.valueOf(columnType.toUpperCase());
		CSVSchemaColumn.TransformType colTransformEnum =
        CSVSchemaColumn.TransformType.valueOf(columnTransform.toUpperCase().substring(1));

		return new CSVSchemaColumn( columnName, colTypeEnum, colTransformEnum );
	}

	private void addSchemaLine( String line ) {

		// parse out: columnName, columnType, columnTransform
		String lineCondensed = line.trim().replaceAll(" +", " ");
		String[] parts = lineCondensed.split(" ");

		if ( parts[ 0 ].toLowerCase().equals("@relation") ) {

		//	return this.validateRelationLine(parts);
			this.relation = parts[1];

		} else if ( parts[ 0 ].toLowerCase().equals("@delimiter") ) {

		//	return this.validateDelimiterLine(parts);
			this.delimiter = parts[1];

		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			String key = parts[1];
			CSVSchemaColumn colValue = this.parseColumnSchemaFromAttribute( parts );

			this.columnSchemas.put( key, colValue );
		}
	}
	
	public void parseSchemaFile(String schemaPath) throws Exception {
		try (BufferedReader br = new BufferedReader(new FileReader(schemaPath))) {
		    for (String line; (line = br.readLine()) != null; ) {
		        // process the line.
		    	if (!this.validateSchemaLine(line) ) {
		    		throw new Exception("Bad Schema for CSV Data: \n\t" + line);
		    	}

		    	// now add it to the schema cache
		    	this.addSchemaLine(line);

		    }
		    // line is not visible here.
		}
	}

	/**
	 * Returns how many columns a newly transformed vector should have
	 *
	 *
	 *
	 * @return
	 */
	public int getTransformedVectorSize() {

		int colCount = 0;

		for (Map.Entry<String, CSVSchemaColumn> entry : this.columnSchemas.entrySet()) {
			if (entry.getValue().transform != CSVSchemaColumn.TransformType.SKIP) {
				colCount++;
			}
		}

		return colCount;

	}

	public void evaluateInputRecord(String csvRecordLine) throws Exception {

		// does the record have the same number of columns that our schema expects?

		String[] columns = csvRecordLine.split( this.delimiter );

		if (Strings.isNullOrEmpty(columns[0])) {
			log.info("Skipping blank line");
			return;
		}

		if (columns.length != this.columnSchemas.size() ) {

			throw new Exception("Row column count does not match schema column count. (" + columns.length + " != " + this.columnSchemas.size() + ") ");

		}

		int colIndex = 0;

		for (Map.Entry<String, CSVSchemaColumn> entry : this.columnSchemas.entrySet()) {


			String colKey = entry.getKey();
		    CSVSchemaColumn colSchemaEntry = entry.getValue();

		    // now work with key and value...
		    colSchemaEntry.evaluateColumnValue( columns[ colIndex ] );

		    colIndex++;

		}

	}



	/**
	 * We call this method once we've scanned the entire dataset once to gather column stats
	 *
	 */
	public void computeDatasetStatistics() {
		this.hasComputedStats = true;
	}

	public void debugPringDatasetStatistics() {

		log.info("Print Schema --------");

		for (Map.Entry<String, CSVSchemaColumn> entry : this.columnSchemas.entrySet()) {

			String key = entry.getKey();
      CSVSchemaColumn value = entry.getValue();

		  // now work with key and value...

		  log.info("> " + value.name + ", " + value.columnType + ", " + value.transform);

		  if ( value.transform == TransformType.LABEL ) {

			  log.info("\t> Label > Class Balance Report ");

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

			  	// value.recordLabels.size()
			  	log.info("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());

			  }

      } else {

			    log.info("\t\tmin: {}", value.minValue);
			    log.info("\t\tmax: {}", value.maxValue);

		    }

		}

		log.info("End Print Schema --------\n\n");

	}

	public void debugPrintColumns() {

		for (Map.Entry<String, CSVSchemaColumn> entry : this.columnSchemas.entrySet()) {

			String key = entry.getKey();
		    CSVSchemaColumn value = entry.getValue();

		    // now work with key and value...

		    log.debug("> {} , {} , {}", value.name, value.columnType, value.transform);

		}

	}



}
