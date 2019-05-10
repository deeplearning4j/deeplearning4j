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

package org.datavec.spark.transform;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.InvalidClassException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by kepricon on 17. 6. 20.
 */
@Data
@Slf4j
public class SparkTransformServerChooser {
    private SparkTransformServer sparkTransformServer = null;
    private TransformDataType transformDataType = null;

    public void runMain(String[] args) throws Exception {

        int pos = getMatchingPosition(args, "-dt", "--dataType");
        if (pos == -1) {
            log.error("no valid options");
            log.error("-dt, --dataType   Options: [CSV, IMAGE]");
            throw new Exception("no valid options");
        } else {
            transformDataType = TransformDataType.valueOf(args[pos + 1]);
        }

        switch (transformDataType) {
            case CSV:
                sparkTransformServer = new CSVSparkTransformServer();
                break;
            case IMAGE:
                sparkTransformServer = new ImageSparkTransformServer();
                break;
            default:
                throw new InvalidClassException("no matching SparkTransform class");
        }

        sparkTransformServer.runMain(args);
    }

    private int getMatchingPosition(String[] args, String... options) {
        List optionList = Arrays.asList(options);

        for (int i = 0; i < args.length; i++) {
            if (optionList.contains(args[i])) {
                return i;
            }
        }
        return -1;
    }


    public static void main(String[] args) throws Exception {
        new SparkTransformServerChooser().runMain(args);
    }
}
