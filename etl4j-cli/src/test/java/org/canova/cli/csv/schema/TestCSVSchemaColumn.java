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

public class TestCSVSchemaColumn {

	@Test
	public void testClassBalanceReporting() throws Exception {

		CSVSchemaColumn schemaCol_0 = new CSVSchemaColumn( "a", CSVSchemaColumn.ColumnType.NOMINAL, CSVSchemaColumn.TransformType.LABEL );
	
		schemaCol_0.evaluateColumnValue("alpha");
		schemaCol_0.evaluateColumnValue("beta");
		schemaCol_0.evaluateColumnValue("gamma");

		schemaCol_0.evaluateColumnValue("alpha");
		schemaCol_0.evaluateColumnValue("beta");
		schemaCol_0.evaluateColumnValue("gamma");

		schemaCol_0.evaluateColumnValue("alpha");
		schemaCol_0.evaluateColumnValue("beta");

		schemaCol_0.evaluateColumnValue("alpha");
		
		schemaCol_0.debugPrintColumns();
		
		assertEquals( 3, schemaCol_0.getLabelCount("beta"), 0.0 );
		assertEquals( 2, schemaCol_0.getLabelCount("gamma"), 0.0 );
		assertEquals( 4, schemaCol_0.getLabelCount("alpha"), 0.0 );

	}

	@Test
	public void testMinMaxMetrics() throws Exception {
		
		CSVSchemaColumn schemaCol_0 = new CSVSchemaColumn( "a", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.COPY );
		
		schemaCol_0.evaluateColumnValue("1");
		schemaCol_0.evaluateColumnValue("2");
		schemaCol_0.evaluateColumnValue("3");
		
		//schemaCol_0.computeStatistics();
		
		assertEquals( 1, schemaCol_0.minValue, 0.0 );
		assertEquals( 3, schemaCol_0.maxValue, 0.0 );
	}

	@Test
	public void testMinMaxMetricsMixedInsertOrder() throws Exception {
		
		CSVSchemaColumn schemaCol_0 = new CSVSchemaColumn( "a", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.COPY );
		
		schemaCol_0.evaluateColumnValue("6");
		schemaCol_0.evaluateColumnValue("-2");
		schemaCol_0.evaluateColumnValue("3");
		
		//schemaCol_0.computeStatistics();
		
		assertEquals( -2, schemaCol_0.minValue, 0.0 );
		assertEquals( 6, schemaCol_0.maxValue, 0.0 );
	}
		
	
	@Test
	public void testEvaluateCSVRecords_NumericColumns_Normalize() throws Exception {
		
		

		// { NUMERIC + NORMALIZE }

		CSVSchemaColumn schemaCol_normalize = new CSVSchemaColumn( "a", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.NORMALIZE );

		schemaCol_normalize.evaluateColumnValue("6");
		schemaCol_normalize.evaluateColumnValue("-2");
		schemaCol_normalize.evaluateColumnValue("3");
				
		assertEquals( -2, schemaCol_normalize.minValue, 0.0 );
		assertEquals( 6, schemaCol_normalize.maxValue, 0.0 );

		double col_val = schemaCol_normalize.transformColumnValue("3");
		
		assertEquals( ((3.0 - -2) / 8.0), col_val, 0.01 );
		
	}
	
	@Test
	public void testEvaluateCSVRecords_NumericColumns_Binarize() throws Exception {
	
		// { NUMERIC + BINARIZE }		
		
		CSVSchemaColumn schemaCol_binarize = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.BINARIZE );

		schemaCol_binarize.evaluateColumnValue("6");
		schemaCol_binarize.evaluateColumnValue("-6");
				
		assertEquals( -6, schemaCol_binarize.minValue, 0.0 );
		assertEquals( 6, schemaCol_binarize.maxValue, 0.0 );

		double col_val = schemaCol_binarize.transformColumnValue("2");

		assertEquals( 1.0, col_val, 0.0 );
		
	}
	
	@Test
	public void testEvaluateCSVRecords_NumericColumns_Copy() throws Exception {
	
		// { NUMERIC + COPY }		
		CSVSchemaColumn schemaCol = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.COPY );

		schemaCol.evaluateColumnValue("6");
		schemaCol.evaluateColumnValue("-6");
				
		assertEquals( -6, schemaCol.minValue, 0.0 );
		assertEquals( 6, schemaCol.maxValue, 0.0 );

		double col_val = schemaCol.transformColumnValue("2");

		assertEquals( 2.0, col_val, 0.0 );
		
	}
	
	@Test
	public void testEvaluateCSVRecords_NumericColumns_Label() throws Exception {
	
		
		// { NUMERIC + LABEL }: same as copy
		
		CSVSchemaColumn schemaCol = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NUMERIC, CSVSchemaColumn.TransformType.LABEL );

		schemaCol.evaluateColumnValue("6");
		schemaCol.evaluateColumnValue("-6");
				
		assertEquals( -6, schemaCol.minValue, 0.0 );
		assertEquals( 6, schemaCol.maxValue, 0.0 );

		double col_val = schemaCol.transformColumnValue("2");

		assertEquals( 2.0, col_val, 0.0 );
		

	}
	

	@Test
	public void testEvaluateCSVRecords_NominalColumns_Normalize() throws Exception {
	
		
		// { Nominal + Normalize }
		
		CSVSchemaColumn schemaCol = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NOMINAL, CSVSchemaColumn.TransformType.NORMALIZE );

		schemaCol.evaluateColumnValue("alpha");
		
		schemaCol.evaluateColumnValue("beta");
		schemaCol.evaluateColumnValue("beta");
		schemaCol.evaluateColumnValue("beta");
		
		schemaCol.evaluateColumnValue("gamma");
		schemaCol.evaluateColumnValue("delta");
				
		assertEquals( 1, schemaCol.getLabelCount("alpha"), 0.0 );
		assertEquals( 3, schemaCol.getLabelCount("beta"), 0.0 );

		double col_val = schemaCol.transformColumnValue("alpha");

		assertEquals( 0.25, col_val, 0.0 );
		
		double col_val_1 = schemaCol.transformColumnValue("beta");

		assertEquals( 0.5, col_val_1, 0.0 );

		double col_val_2 = schemaCol.transformColumnValue("gamma");

		assertEquals( 0.75, col_val_2, 0.0 );

		double col_val_3 = schemaCol.transformColumnValue("delta");

		assertEquals( 1.0, col_val_3, 0.0 );
		
		
	}	
	
	@Test
	public void testEvaluateCSVRecords_NominalColumns_Binarize() throws Exception {
	
		
		// { Nominal + Binarize }: unsupported
			

	}	

	@Test
	public void testEvaluateCSVRecords_NominalColumns_Copy() throws Exception {
	
		
		// { Nominal + Normalize }
		
		CSVSchemaColumn schemaCol = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NOMINAL, CSVSchemaColumn.TransformType.COPY );

		schemaCol.evaluateColumnValue("a");		
		schemaCol.evaluateColumnValue("b");
		schemaCol.evaluateColumnValue("c");
				
		// should not be labeling!
		assertEquals( 1, schemaCol.getLabelCount("a"), 0.0 );
		assertEquals( 1, schemaCol.getLabelCount("b"), 0.0 );
		assertEquals( 0, schemaCol.getLabelCount("d"), 0.0 );

		double col_val = schemaCol.transformColumnValue("a");

		assertEquals( 0.0, col_val, 0.0 );
		
		double col_val_b = schemaCol.transformColumnValue("b");

		assertEquals( 1.0, col_val_b, 0.0 );

		
	}			
	

	@Test
	public void testEvaluateCSVRecords_NominalColumns_Label() throws Exception {
	
		
		// { Nominal + Normalize }
		
		CSVSchemaColumn schemaCol = new CSVSchemaColumn( "b", CSVSchemaColumn.ColumnType.NOMINAL, CSVSchemaColumn.TransformType.LABEL );

		schemaCol.evaluateColumnValue("a");		
		schemaCol.evaluateColumnValue("b");
		schemaCol.evaluateColumnValue("c");
				
		// should not be labeling!
		assertEquals( 1, schemaCol.getLabelCount("a"), 0.0 );
		assertEquals( 1, schemaCol.getLabelCount("b"), 0.0 );
		assertEquals( 0, schemaCol.getLabelCount("d"), 0.0 );

		double col_val = schemaCol.transformColumnValue("c");

		assertEquals( 2.0, col_val, 0.0 );
		
		double col_val_b = schemaCol.transformColumnValue("b");

		assertEquals( 1.0, col_val_b, 0.0 );

		
	}	
	
	
}
