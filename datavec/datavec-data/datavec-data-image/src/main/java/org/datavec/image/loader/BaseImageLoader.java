/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.loader;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.image.data.Image;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.util.ArchiveUtils;

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
@Slf4j
public abstract class BaseImageLoader implements Serializable {

    public enum MultiPageMode {
        MINIBATCH, FIRST //, CHANNELS,
    }

    public static final File BASE_DIR = new File(System.getProperty("user.home"));
    public static final String[] ALLOWED_FORMATS = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
    protected Random rng = new Random(System.currentTimeMillis());

    protected long height = -1;
    protected long width = -1;
    protected long channels = -1;
    protected boolean centerCropIfNeeded = false;
    protected ImageTransform imageTransform = null;
    protected MultiPageMode multiPageMode = null;

    public String[] getAllowedFormats() {
        return ALLOWED_FORMATS;
    }

    public abstract INDArray asRowVector(File f) throws IOException;

    public abstract INDArray asRowVector(InputStream inputStream) throws IOException;

    /** As per {@link #asMatrix(File, boolean)} but NCHW/channels_first format */
    public abstract INDArray asMatrix(File f) throws IOException;

    /**
     * Load an image from a file to an INDArray
     * @param f    File to load the image from
     * @param nchw If true: return image in NCHW/channels_first [1, channels, height width] format; if false, return
     *             in NHWC/channels_last [1, height, width, channels] format
     * @return Image file as as INDArray
     */
    public abstract INDArray asMatrix(File f, boolean nchw) throws IOException;

    public abstract INDArray asMatrix(InputStream inputStream) throws IOException;
    /**
     * Load an image file from an input stream to an INDArray
     * @param inputStream Input stream to load the image from
     * @param nchw If true: return image in NCHW/channels_first [1, channels, height width] format; if false, return
     *             in NHWC/channels_last [1, height, width, channels] format
     * @return Image file stream as as INDArray
     */
    public abstract INDArray asMatrix(InputStream inputStream, boolean nchw) throws IOException;

    /** As per {@link #asMatrix(File)} but as an {@link Image}*/
    public abstract Image asImageMatrix(File f) throws IOException;
    /** As per {@link #asMatrix(File, boolean)} but as an {@link Image}*/
    public abstract Image asImageMatrix(File f, boolean nchw) throws IOException;

    /** As per {@link #asMatrix(InputStream)} but as an {@link Image}*/
    public abstract Image asImageMatrix(InputStream inputStream) throws IOException;
    /** As per {@link #asMatrix(InputStream, boolean)} but as an {@link Image}*/
    public abstract Image asImageMatrix(InputStream inputStream, boolean nchw) throws IOException;


    public static void downloadAndUntar(Map urlMap, File fullDir) {
        try {
            File file = new File(fullDir, urlMap.get("filesFilename").toString());
            if (!file.isFile()) {
                FileUtils.copyURLToFile(new URL(urlMap.get("filesURL").toString()), file);
            }

            String fileName = file.toString();
            if (fileName.endsWith(".tgz") || fileName.endsWith(".tar.gz") || fileName.endsWith(".gz")
                            || fileName.endsWith(".zip"))
                ArchiveUtils.unzipFileTo(file.getAbsolutePath(), fullDir.getAbsolutePath(), false);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to fetch images", e);
        }
    }

}
