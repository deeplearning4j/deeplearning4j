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

import static org.junit.Assert.*;

import org.junit.Test;

public class TestCSVInputSchema {

	@Test
	public void testLoadAndValidateSchema() throws Exception {

		String schemaFilePath = "src/test/resources/csv/schemas/unit_test_schema.txt";
		CSVInputSchema inputSchema = new CSVInputSchema();
		inputSchema.parseSchemaFile( schemaFilePath );
		
		inputSchema.debugPrintColumns();
		
		assertEquals( ",", inputSchema.delimiter );
		assertEquals( "SytheticDatasetUnitTest", inputSchema.relation );
		
		assertEquals( CSVSchemaColumn.ColumnType.NUMERIC, inputSchema.getColumnSchemaByName( "sepallength" ).columnType );
		assertEquals( CSVSchemaColumn.TransformType.COPY, inputSchema.getColumnSchemaByName( "sepallength" ).transform );
		
		assertEquals( null, inputSchema.getColumnSchemaByName("foo") );

		assertEquals( CSVSchemaColumn.ColumnType.NOMINAL, inputSchema.getColumnSchemaByName( "class" ).columnType );
		assertEquals( CSVSchemaColumn.TransformType.LABEL, inputSchema.getColumnSchemaByName( "class" ).transform );
		
		
	}
	
	@Test
	public void testLoadingUnsupportedSchemas() throws Exception {
	
		boolean caughtException_0 = false;
		boolean caughtException_1 = false;
		boolean caughtException_2 = false;
		
		String schemaFilePath = "";
		CSVInputSchema inputSchema = null;
		
		// 1. Unsupported: { NUMERIC + LABEL }
		/*
		String schemaFilePath = "src/test/resources/csv/schemas/csv_unsupported_schema_0.txt";
		CSVInputSchema inputSchema = new CSVInputSchema();
		try {
			inputSchema.parseSchemaFile( schemaFilePath );
		} catch (Exception e) {
			caughtException_0 = true;
		}
		*/

		
		// 2. Unsupported: { NOMINAL + BINARIZE }

		schemaFilePath = "src/test/resources/csv/schemas/csv_unsupported_schema_1.txt";
		inputSchema = new CSVInputSchema();
		try {
			inputSchema.parseSchemaFile( schemaFilePath );
		} catch (Exception e) {
			caughtException_1 = true;
		}
		
		
		// 3. Unsupported: { DATE + anything } --- date columns arent finished yet!
		
		schemaFilePath = "src/test/resources/csv/schemas/csv_unsupported_schema_2.txt";
		inputSchema = new CSVInputSchema();
		try {
			inputSchema.parseSchemaFile( schemaFilePath );
		} catch (Exception e) {
			caughtException_2 = true;
		}
		
		//assertEquals( true, caughtException_0 );
		assertEquals( true, caughtException_1 );
		assertEquals( true, caughtException_2 );
		
	}

	@Test
	public void testEvaluateCSVRecords_NumericColumns() {

		String schemaFilePath = "";
		CSVInputSchema inputSchema = null;
/*		
		schemaFilePath = "src/test/resources/csv/schemas/csv_unsupported_schema_2.txt";
		inputSchema = new CSVInputSchema();
		try {
			inputSchema.parseSchemaFile( schemaFilePath );
		} catch (Exception e) {
			caughtException_2 = true;
		}
	*/	
		
		// { NUMERIC + NORMALIZE }

		
		// { NUMERIC + BINARIZE }		
		

		// { NUMERIC + COPY }		
		
		
		// { NUMERIC + LABEL }: same as copy

		
		// { NUMERIC + LABEL }		
		
	}
	
	@Test
	public void testEvaluateCSVRecords_NominalColumns() {

		// { NOMINAL + NORMALIZE }

		
		// { NOMINAL + BINARIZE }: Unsupported	
		

		// { NOMINAL + COPY }		
		
		
		// { NOMINAL + LABEL }		

		
		// { NOMINAL + LABEL }		
		
	}
	
	
}
