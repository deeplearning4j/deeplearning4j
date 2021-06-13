/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.recordreader.objdetect.impl;

import lombok.Getter;
import lombok.NonNull;
import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.coco.COCODataSet;
import org.nd4j.serde.json.JsonMappers;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class COCOLabelProvider implements ImageObjectLabelProvider {


    private String annotationsFile;
    private String rootDir;
    @Getter
    private COCODataSet cocoDataSet;

    @SneakyThrows
    public COCOLabelProvider(@NonNull String annotationsFile) {
        this(annotationsFile, null);
    }

    @SneakyThrows
    public COCOLabelProvider(@NonNull String annotationsFile,String rootDir) {
        //ideally, file names are stored in the coco with all the images being in a directory
        //specified by the user
        this.annotationsFile = annotationsFile;
        this.rootDir = rootDir;

        if (!new File(annotationsFile).exists()) {
            throw new IllegalStateException("COCO Annotations file does not currently exist.");
        }

        cocoDataSet = JsonMappers.getMapper().readValue(FileUtils.readFileToString(new File(annotationsFile), Charset.defaultCharset()),COCODataSet.class);
        cocoDataSet.init();
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(String path) {
        //probably not correct, need to understand original voc implementation
        if(rootDir != null) {
            if (cocoDataSet.hasImage(new File(rootDir, path).getAbsolutePath()))
                return Arrays.asList(cocoDataSet.getImageForName(new File(rootDir, path).getAbsolutePath()));
            //sometimes absolute path is passed in but omitted in the json file, allow file names stripped from absolute paths to also match
        }     else if(!cocoDataSet.hasImage(path)) {
            File absPath = new File(path);
            if(cocoDataSet.hasImage(absPath.getName())) {
                return Arrays.asList(cocoDataSet.getImageForName(absPath.getName()));
            }
        }
        return Arrays.asList(cocoDataSet.getImageForName(path));
    }


    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return getImageObjectsForPath(uri.toString());
    }

}
