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
package org.datavec.image.transform;

import lombok.Data;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.datavec.image.data.ImageWritable;

import javax.swing.*;
import java.util.Random;

/**
 * Shows images on the screen, does not actually transform them.
 * To continue to the next image, press any key in the window of the CanvasFrame.
 *
 * @author saudet
 */
@Data
public class ShowImageTransform extends BaseImageTransform {

    CanvasFrame canvas;
    String title;
    int delay;

    /** Calls {@code this(canvas, -1)}. */
    public ShowImageTransform(CanvasFrame canvas) {
        this(canvas, -1);
    }

    /**
     * Constructs an instance of the ImageTransform from a {@link CanvasFrame}.
     *
     * @param canvas to display images in
     * @param delay  max time to wait in milliseconds (0 == infinity, negative == no wait)
     */
    public ShowImageTransform(CanvasFrame canvas, int delay) {
        super(null);
        this.canvas = canvas;
        this.delay = delay;
    }

    /** Calls {@code this(title, -1)}. */
    public ShowImageTransform(String title) {
        this(title, -1);
    }

    /**
     * Constructs an instance of the ImageTransform with a new {@link CanvasFrame}.
     *
     * @param title of the new CanvasFrame to display images in
     * @param delay max time to wait in milliseconds (0 == infinity, negative == no wait)
     */
    public ShowImageTransform(String title, int delay) {
        super(null);
        this.canvas = null;
        this.title = title;
        this.delay = delay;
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (canvas == null) {
            canvas = new CanvasFrame(title, 1.0);
            canvas.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        }
        if (image == null) {
            canvas.dispose();
            return null;
        }
        if (!canvas.isVisible()) {
            return image;
        }
        Frame frame = image.getFrame();
        canvas.setCanvasSize(frame.imageWidth, frame.imageHeight);
        canvas.showImage(frame);
        if (delay >= 0) {
            try {
                canvas.waitKey(delay);
            } catch (InterruptedException ex) {
                // reset interrupt to be nice
                Thread.currentThread().interrupt();
            }
        }
        return image;
    }

    @Override
    public float[] query(float... coordinates) {
        return coordinates;
    }
}
