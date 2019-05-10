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

package org.nd4j.linalg.api.ndarray;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 6/16/16.
 */
@Deprecated
public class NdArrayJSONReader {

    public INDArray read(File jsonFile) {
        INDArray result = this.loadNative(jsonFile);
        if (result == null) {
            //Must write support for parsing/normal json parsing - which will be inefficient
            this.loadNonNative(jsonFile);
        }
        return result;
    }

    private INDArray loadNative(File jsonFile) {
        /*
          We could dump an ndarray to a file with the tostring (since that is valid json) and use put/get to parse it as json
        
          But here we leverage our information of the tostring method to be more efficient
          With our current toString format we use tads along dimension (rank-1,rank-2) to write to the array in two dimensional chunks at a time.
          This is more efficient than setting each value at a time with putScalar.
          This also means we can read the file one line at a time instead of loading the whole thing into memory
        
          Future work involves enhancing the write json method to provide more features to make the load more efficient
         */
        int lineNum = 0;
        int rowNum = 0;
        int tensorNum = 0;
        char theOrder = 'c';
        int[] theShape = {1, 1};
        int rank = 0;
        double[][] subsetArr = {{0.0, 0.0}, {0.0, 0.0}};
        INDArray newArr = Nd4j.zeros(2, 2);
        try {
            LineIterator it = FileUtils.lineIterator(jsonFile);
            try {
                while (it.hasNext()) {
                    String line = it.nextLine();
                    lineNum++;
                    line = line.replaceAll("\\s", "");
                    if (line.equals("") || line.equals("}"))
                        continue;
                    // is it from dl4j?
                    if (lineNum == 2) {
                        String[] lineArr = line.split(":");
                        String fileSource = lineArr[1].replaceAll("\\W", "");
                        if (!fileSource.equals("dl4j"))
                            return null;
                    }
                    // parse ordering
                    if (lineNum == 3) {
                        String[] lineArr = line.split(":");
                        theOrder = lineArr[1].replace("\\W", "").charAt(0);
                        continue;
                    }
                    // parse shape
                    if (lineNum == 4) {
                        String[] lineArr = line.split(":");
                        String dropJsonComma = lineArr[1].split("]")[0];
                        String[] shapeString = dropJsonComma.replace("[", "").split(",");
                        rank = shapeString.length;
                        theShape = new int[rank];
                        for (int i = 0; i < rank; i++) {
                            try {
                                theShape[i] = Integer.parseInt(shapeString[i]);
                            } catch (NumberFormatException nfe) {
                            } ;
                        }
                        subsetArr = new double[theShape[rank - 2]][theShape[rank - 1]];
                        newArr = Nd4j.zeros(theShape, theOrder);
                        continue;
                    }
                    //parse data
                    if (lineNum > 5) {
                        String[] entries =
                                        line.replace("\\],", "").replaceAll("\\[", "").replaceAll("\\]", "").split(",");
                        for (int i = 0; i < theShape[rank - 1]; i++) {
                            try {
                                subsetArr[rowNum][i] = Double.parseDouble(entries[i]);
                            } catch (NumberFormatException nfe) {
                            }
                        }
                        rowNum++;
                        if (rowNum == theShape[rank - 2]) {
                            INDArray subTensor = Nd4j.create(subsetArr);
                            newArr.tensorAlongDimension(tensorNum, rank - 1, rank - 2).addi(subTensor);
                            rowNum = 0;
                            tensorNum++;
                        }
                    }
                }
            } finally {
                LineIterator.closeQuietly(it);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading input", e);
        }
        return newArr;
    }

    private INDArray loadNonNative(File jsonFile) {
        /* WIP
        JSONTokener tokener = new JSONTokener(new FileReader("test.json"));
        JSONObject obj = new JSONObject(tokener);
        JSONArray objArr = obj.optJSONArray("shape");
        int rank = objArr.length();
        int[] theShape = new int[rank];
        int rows = 1;
        for (int i = 0; i < rank; ++i) {
            theShape[i] = objArr.optInt(i);
            if (i != objArr.length() - 1)
                rows *= theShape[i];
        }
        */
        System.out.println("API_Error: Current support only for files written from dl4j");
        return null;
    }
}
