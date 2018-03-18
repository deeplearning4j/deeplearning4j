package org.datavec.api.split;

import org.junit.Test;

import java.net.URI;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class NumberedFileInputSplitTests {
    @Test
    public void testNumberedFileInputSplitBasic() {
        String baseString = "/path/to/files/prefix%d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test
    public void testNumberedFileInputSplitVaryIndeces() {
        String baseString = "/path/to/files/prefix-%d.suffix";
        int minIdx = 3;
        int maxIdx = 27;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test
    public void testNumberedFileInputSplitBasicNoPrefix() {
        String baseString = "/path/to/files/%d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test
    public void testNumberedFileInputSplitWithLeadingZeroes() {
        String baseString = "/path/to/files/prefix-%07d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test
    public void testNumberedFileInputSplitWithLeadingZeroesNoSuffix() {
        String baseString = "/path/to/files/prefix-%d";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithLeadingSpaces() {
        String baseString = "/path/to/files/prefix-%5d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithNoLeadingZeroInPadding() {
        String baseString = "/path/to/files/prefix%5d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithLeadingPlusInPadding() {
        String baseString = "/path/to/files/prefix%+5d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithLeadingMinusInPadding() {
        String baseString = "/path/to/files/prefix%-5d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithTwoDigitsInPadding() {
        String baseString = "/path/to/files/prefix%011d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithInnerZerosInPadding() {
        String baseString = "/path/to/files/prefix%101d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNumberedFileInputSplitWithRepeatInnerZerosInPadding() {
        String baseString = "/path/to/files/prefix%0505d.suffix";
        int minIdx = 0;
        int maxIdx = 10;
        runNumberedFileInputSplitTest(baseString, minIdx, maxIdx);
    }


    private static void runNumberedFileInputSplitTest(String baseString, int minIdx, int maxIdx) {
        NumberedFileInputSplit split = new NumberedFileInputSplit(baseString, minIdx, maxIdx);
        URI[] locs = split.locations();
        assertEquals(locs.length, (maxIdx - minIdx) + 1);
        int j = 0;
        for (int i = minIdx; i <= maxIdx; i++) {
            String path = locs[j++].getPath();
            String exp = String.format(baseString, i);
            String msg = exp + " vs " + path;
            assertTrue(msg, path.endsWith(exp));    //Note: on Windows, Java can prepend drive to path - "/C:/"
        }
    }
}
