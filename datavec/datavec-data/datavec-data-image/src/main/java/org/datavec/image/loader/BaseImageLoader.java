/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.image.loader;

import org.apache.commons.io.FileUtils;
import org.datavec.image.data.Image;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.net.URL;
import java.util.Map;
import java.util.Random;

/**
 * Created by nyghtowl on 12/17/15.
 */
public abstract class BaseImageLoader implements Serializable {

    protected static final Logger log = LoggerFactory.getLogger(BaseImageLoader.class);

    public static final File BASE_DIR = new File(System.getProperty("user.home"));
    public static final String[] ALLOWED_FORMATS = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
    protected Random rng = new Random(System.currentTimeMillis());

    protected long height = -1;
    protected long width = -1;
    protected long channels = -1;
    protected boolean centerCropIfNeeded = false;
    protected ImageTransform imageTransform = null;
    protected NativeImageLoader.MultiPageMode multiPageMode = null;

    public String[] getAllowedFormats() {
        return ALLOWED_FORMATS;
    }

    public abstract INDArray asRowVector(File f) throws IOException;

    public abstract INDArray asRowVector(InputStream inputStream) throws IOException;

    public abstract INDArray asMatrix(File f) throws IOException;

    public abstract INDArray asMatrix(InputStream inputStream) throws IOException;

    public abstract Image asImageMatrix(File f) throws IOException;

    public abstract Image asImageMatrix(InputStream inputStream) throws IOException;


    public static void downloadAndUntar(Map urlMap, File fullDir) {
        try {
            File file = new File(fullDir, urlMap.get("filesFilename").toString());
            if (!file.isFile()) {
                FileUtils.copyURLToFile(new URL(urlMap.get("filesURL").toString()), file);
            }

            String fileName = file.toString();
            if (fileName.endsWith(".tgz") || fileName.endsWith(".tar.gz") || fileName.endsWith(".gz")
                            || fileName.endsWith(".zip"))
                ArchiveUtils.unzipFileTo(file.getAbsolutePath(), fullDir.getAbsolutePath());
        } catch (IOException e) {
            throw new IllegalStateException("Unable to fetch images", e);
        }
    }

}
