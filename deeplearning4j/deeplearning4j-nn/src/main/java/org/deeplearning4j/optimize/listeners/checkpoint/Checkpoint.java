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

package org.deeplearning4j.optimize.listeners.checkpoint;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A model checkpoint, used with {@link CheckpointListener}
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class Checkpoint implements Serializable {

    private int checkpointNum;
    private long timestamp;
    private int iteration;
    private int epoch;
    private String modelType;
    private String filename;

    public static String getFileHeader(){
        return "checkpointNum,timestamp,iteration,epoch,modelType,filename";
    }

    public static Checkpoint fromFileString(String str){
        String[] split = str.split(",");
        if(split.length != 6){
            throw new IllegalStateException("Cannot parse checkpoint entry: expected 6 entries, got " + split.length
                    + " - values = " + Arrays.toString(split));
        }
        return new Checkpoint(
                Integer.parseInt(split[0]),
                Long.parseLong(split[1]),
                Integer.parseInt(split[2]),
                Integer.parseInt(split[3]),
                split[4],
                split[5]);
    }

    public String toFileString(){
        return checkpointNum + "," + timestamp + "," + iteration + "," + epoch + "," + modelType + "," + filename;
    }
}
