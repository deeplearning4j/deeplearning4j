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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.UUID;


/**
 * Created by nyghtowl on 11/8/15.
 */
public class FileSplitTest {

    protected File file, file1, file2, file3, file4, file5, file6, newPath;
    protected String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};
    private static String localPath = System.getProperty("java.io.tmpdir") + File.separator;
    private static String testPath = localPath + "test" + File.separator;

    // These cannot run on TravisCI - uncomment when checking locally

    @Test(expected = IllegalArgumentException.class)
    public void testEmptySplit() {
        new FileSplit(new File("THE_TEST_WILL_PASS_UNLESS_YOU_CAN_PREDICT_THIS_UUID: " + UUID.randomUUID()));
    }

    @Rule
    public TemporaryFolder mainFolder = new TemporaryFolder();

    //
    //    @Before
    //    public void doBefore() throws IOException {
    //        file = mainFolder.newFile("myfile.txt");
    //
    //        newPath = new File(testPath);
    //
    //        newPath.mkdir();
    //
    //        file1 = File.createTempFile("myfile_1", ".jpg", newPath);
    //        file2 = File.createTempFile("myfile_2", ".txt", newPath);
    //        file3 = File.createTempFile("myfile_3", ".jpg", newPath);
    //        file4 = File.createTempFile("treehouse_4", ".csv", newPath);
    //        file5 = File.createTempFile("treehouse_5", ".csv", newPath);
    //        file6 = File.createTempFile("treehouse_6", ".jpg", newPath);
    //
    //    }
    //
    //    @Test
    //    public void testInitializeLoadSingleFile(){
    //        InputSplit split = new FileSplit(file, allForms);
    //        assertEquals(split.locations()[0], file.toURI());
    //
    //    }
    //
    //    @Test
    //    public void testInitializeLoadMulFiles() throws IOException{
    //        InputSplit split = new FileSplit(newPath, allForms, true);
    //        assertEquals(3, split.locations().length);
    //        assertEquals(file1.toURI(), split.locations()[0]);
    //        assertEquals(file3.toURI(), split.locations()[1]);
    //    }
    //
    //    @Test
    //    public void testInitializeMulFilesShuffle() throws IOException{
    //        InputSplit split = new FileSplit(newPath, new Random(123));
    //        InputSplit split2 = new FileSplit(newPath, new Random(123));
    //        assertEquals(6, split.locations().length);
    //        assertEquals(6, split2.locations().length);
    //        assertEquals(split.locations()[3], split2.locations()[3]);
    //    }
    //
    //    @After
    //    public void doAfter(){
    //        mainFolder.delete();
    //        file.delete();
    //        file1.delete();
    //        file2.delete();
    //        file3.delete();
    //        file4.delete();
    //        file5.delete();
    //        file6.delete();
    //        newPath.delete();
    //
    //    }

}
