/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.plot;

import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Render ndarrays as images
 *
 * @author Adam Gibson
 */
public class ImageRender {
    public static void render(INDArray image,String path) throws IOException {
        BufferedImage imageToRender = null;
        if(image.rank() == 3) {
            ImageLoader loader = new ImageLoader(image.size(-1),image.size(-2),image.size(-3));
            imageToRender = new BufferedImage(image.size(-1),image.size(-2),BufferedImage.TYPE_3BYTE_BGR);
            loader.toBufferedImageRGB(image,imageToRender);
        }

        else if(image.rank() == 2) {
            imageToRender = new BufferedImage(image.size(-1),image.size(-2),BufferedImage.TYPE_BYTE_GRAY);
            for( int i = 0; i < image.length(); i++ ){
                imageToRender.getRaster().setSample(i % image.size(-1), i / image.size(-2), 0, (int) (255 * image.getDouble(i)));
            }

        }

        ImageIO.write(imageToRender,"png",new File(path));

    }


}
