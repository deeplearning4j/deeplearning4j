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

package org.datavec.image.transform;

import org.datavec.image.data.ImageWritable;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by kepricon on 17. 5. 25.
 */
public class JsonYamlTest {
    @Test
    public void testJsonYamlImageTransformProcess() throws IOException {
        int seed = 12345;
        Random random = new Random(seed);

        //from org.bytedeco.javacpp.opencv_imgproc
        int COLOR_BGR2Luv = 50;
        int CV_BGR2GRAY = 6;


        ImageTransformProcess itp = new ImageTransformProcess.Builder().colorConversionTransform(COLOR_BGR2Luv)
                        .cropImageTransform(10).equalizeHistTransform(CV_BGR2GRAY).flipImageTransform(0)
                        .resizeImageTransform(300, 300).rotateImageTransform(30).scaleImageTransform(3)
                        .warpImageTransform((float) 0.5)

                        // Note : since randomCropTransform use random value
                        // the results from each case(json, yaml, ImageTransformProcess)
                        // can be different
                        // don't use the below line
                        // if you uncomment it, you will get fail from below assertions
                        //  .randomCropTransform(seed, 50, 50)

                        // Note : you will get "java.lang.NoClassDefFoundError: Could not initialize class org.bytedeco.javacpp.avutil"
                        // it needs to add the below dependency
                        // <dependency>
                        //     <groupId>org.bytedeco.javacpp-presets</groupId>
                        //     <artifactId>ffmpeg-platform</artifactId>
                        // </dependency>
                        // FFmpeg has license issues, be careful to use it
                        //.filterImageTransform("noise=alls=20:allf=t+u,format=rgba", 100, 100, 4)

                        .build();

        String asJson = itp.toJson();
        String asYaml = itp.toYaml();

        System.out.println(asJson);
        System.out.println("\n\n\n");
        System.out.println(asYaml);

        ImageWritable img = TestImageTransform.makeRandomImage(0, 0, 3);
        ImageWritable imgJson = new ImageWritable(img.getFrame().clone());
        ImageWritable imgYaml = new ImageWritable(img.getFrame().clone());
        ImageWritable imgAll = new ImageWritable(img.getFrame().clone());

        ImageTransformProcess itpFromJson = ImageTransformProcess.fromJson(asJson);
        ImageTransformProcess itpFromYaml = ImageTransformProcess.fromYaml(asYaml);

        List<ImageTransform> transformList = itp.getTransformList();
        List<ImageTransform> transformListJson = itpFromJson.getTransformList();
        List<ImageTransform> transformListYaml = itpFromYaml.getTransformList();

        for (int i = 0; i < transformList.size(); i++) {
            ImageTransform it = transformList.get(i);
            ImageTransform itJson = transformListJson.get(i);
            ImageTransform itYaml = transformListYaml.get(i);

            System.out.println(i + "\t" + it);

            img = it.transform(img);
            imgJson = itJson.transform(imgJson);
            imgYaml = itYaml.transform(imgYaml);

            if (it instanceof RandomCropTransform) {
                assertTrue(img.getFrame().imageHeight == imgJson.getFrame().imageHeight);
                assertTrue(img.getFrame().imageWidth == imgJson.getFrame().imageWidth);

                assertTrue(img.getFrame().imageHeight == imgYaml.getFrame().imageHeight);
                assertTrue(img.getFrame().imageWidth == imgYaml.getFrame().imageWidth);
            } else if (it instanceof FilterImageTransform) {
                assertEquals(img.getFrame().imageHeight, imgJson.getFrame().imageHeight);
                assertEquals(img.getFrame().imageWidth, imgJson.getFrame().imageWidth);
                assertEquals(img.getFrame().imageChannels, imgJson.getFrame().imageChannels);

                assertEquals(img.getFrame().imageHeight, imgYaml.getFrame().imageHeight);
                assertEquals(img.getFrame().imageWidth, imgYaml.getFrame().imageWidth);
                assertEquals(img.getFrame().imageChannels, imgYaml.getFrame().imageChannels);
            } else {
                assertEquals(img, imgJson);

                assertEquals(img, imgYaml);
            }
        }

        imgAll = itp.execute(imgAll);

        assertEquals(imgAll, img);
    }
}
