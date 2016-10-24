package org.datavec.dataframe;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Make sure our test data is available
 */
public class TestDataTest {

    @Test
    public void verifyAllTestData() {
        for (TestData testData : TestData.values()) {
            verify(testData);
        }
    }

    private void verify(TestData testData) {

        assertNotNull("Table available", testData.getTable());

        // cheap attempt at testing data integrity
        assertEquals("Column name count matches column type count", testData.getColumnNames().length,
                testData.getColumnTypes().length);

        assertNotNull("Data path available", testData.getSource());
    }

}
