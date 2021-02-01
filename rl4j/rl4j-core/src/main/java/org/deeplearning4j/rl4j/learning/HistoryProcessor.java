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

package org.deeplearning4j.rl4j.learning;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.bytedeco.javacv.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.util.VideoRecorder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/27/16.
 *
 * An IHistoryProcessor implementation using JavaCV
 */
@Slf4j
public class HistoryProcessor implements IHistoryProcessor {

    @Getter
    final private Configuration conf;
    private CircularFifoQueue<INDArray> history;
    private VideoRecorder videoRecorder;

    public HistoryProcessor(Configuration conf) {
        this.conf = conf;
        history = new CircularFifoQueue<>(conf.getHistoryLength());
    }


    public void add(INDArray obs) {
        INDArray processed = transform(obs);
        history.add(processed);
    }

    public void startMonitor(String filename, int[] shape) {
        if(videoRecorder == null) {
            videoRecorder = VideoRecorder.builder(shape[1], shape[2])
                    .build();
        }

        try {
            videoRecorder.startRecording(filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stopMonitor() {
        if(videoRecorder != null) {
            try {
                videoRecorder.stopRecording();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public boolean isMonitoring() {
        return videoRecorder != null && videoRecorder.isRecording();
    }

    public void record(INDArray pixelArray) {
        if(isMonitoring()) {
            // before accessing the raw pointer, we need to make sure that array is actual on the host side
            Nd4j.getAffinityManager().ensureLocation(pixelArray, AffinityManager.Location.HOST);

            try {
                videoRecorder.record(pixelArray);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public INDArray[] getHistory() {
        INDArray[] array = new INDArray[getConf().getHistoryLength()];
        for (int i = 0; i < conf.getHistoryLength(); i++) {
            array[i] = history.get(i).castTo(Nd4j.dataType());
        }
        return array;
    }


    private INDArray transform(INDArray raw) {
        long[] shape = raw.shape();

        // before accessing the raw pointer, we need to make sure that array is actual on the host side
        Nd4j.getAffinityManager().ensureLocation(raw, AffinityManager.Location.HOST);

        Mat ocvmat = new Mat((int)shape[0], (int)shape[1], CV_32FC(3), raw.data().pointer());
        Mat cvmat = new Mat(shape[0], shape[1], CV_8UC(3));
        ocvmat.convertTo(cvmat, CV_8UC(3), 255.0, 0.0);
        cvtColor(cvmat, cvmat, COLOR_RGB2GRAY);
        Mat resized = new Mat(conf.getRescaledHeight(), conf.getRescaledWidth(), CV_8UC(1));
        resize(cvmat, resized, new Size(conf.getRescaledWidth(), conf.getRescaledHeight()));
        //   show(resized);
        //   waitKP();
        //Crop by croppingHeight, croppingHeight
        Mat cropped = resized.apply(new Rect(conf.getOffsetX(), conf.getOffsetY(), conf.getCroppingWidth(),
                        conf.getCroppingHeight()));
        //System.out.println(conf.getCroppingWidth() + " " + cropped.data().asBuffer().array().length);

        INDArray out = null;
        try {
            out = new NativeImageLoader(conf.getCroppingHeight(), conf.getCroppingWidth()).asMatrix(cropped);
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println(out.shapeInfoToString());
        out = out.reshape(1, conf.getCroppingHeight(), conf.getCroppingWidth());
        INDArray compressed = out.castTo(DataType.UBYTE);
        return compressed;
    }

    public double getScale() {
        return 255;
    }

    public void waitKP() {
        try {
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void show(Mat m) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame canvas = new CanvasFrame("LOL", 1);
        canvas.showImage(converter.convert(m));
    }


}
