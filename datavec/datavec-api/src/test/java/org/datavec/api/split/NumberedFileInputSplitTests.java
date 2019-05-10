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
