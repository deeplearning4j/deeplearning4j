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

package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class ZipTests {

    @Test
    public void testZip() throws Exception {

        File testFile = File.createTempFile("adasda","Dsdasdea");

        INDArray arr = Nd4j.create(new double[]{1,2,3,4,5,6,7,8,9,0});

        final FileOutputStream fileOut = new FileOutputStream(testFile);
        final ZipOutputStream zipOut = new ZipOutputStream(fileOut);
        zipOut.putNextEntry(new ZipEntry("params"));
        Nd4j.write(zipOut, arr);
        zipOut.flush();
        zipOut.close();


        final FileInputStream fileIn = new FileInputStream(testFile);
        final ZipInputStream zipIn = new ZipInputStream(fileIn);
        ZipEntry entry = zipIn.getNextEntry();
        INDArray read = Nd4j.read(zipIn);
        zipIn.close();


        assertEquals(arr, read);
    }
}
