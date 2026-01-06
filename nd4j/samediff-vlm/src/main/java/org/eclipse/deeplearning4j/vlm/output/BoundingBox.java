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

package org.eclipse.deeplearning4j.vlm.output;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a bounding box for document elements.
 *
 * Coordinates are typically in the range 0-1000 for DocTags format,
 * representing normalized positions on the page.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BoundingBox {

    private int x1;
    private int y1;
    private int x2;
    private int y2;

    /**
     * Get the width of the bounding box.
     *
     * @return width
     */
    public int getWidth() {
        return x2 - x1;
    }

    /**
     * Get the height of the bounding box.
     *
     * @return height
     */
    public int getHeight() {
        return y2 - y1;
    }

    /**
     * Get the center X coordinate.
     *
     * @return center X
     */
    public int getCenterX() {
        return (x1 + x2) / 2;
    }

    /**
     * Get the center Y coordinate.
     *
     * @return center Y
     */
    public int getCenterY() {
        return (y1 + y2) / 2;
    }

    /**
     * Get the area of the bounding box.
     *
     * @return area
     */
    public int getArea() {
        return getWidth() * getHeight();
    }

    /**
     * Convert to normalized coordinates (0-1).
     *
     * @param maxValue the maximum value (typically 1000 for DocTags)
     * @return normalized bounding box as float array [x1, y1, x2, y2]
     */
    public float[] toNormalized(int maxValue) {
        return new float[]{
                (float) x1 / maxValue,
                (float) y1 / maxValue,
                (float) x2 / maxValue,
                (float) y2 / maxValue
        };
    }

    /**
     * Convert to pixel coordinates.
     *
     * @param imageWidth the image width
     * @param imageHeight the image height
     * @param maxValue the maximum normalized value (typically 1000)
     * @return pixel bounding box as int array [x1, y1, x2, y2]
     */
    public int[] toPixels(int imageWidth, int imageHeight, int maxValue) {
        return new int[]{
                x1 * imageWidth / maxValue,
                y1 * imageHeight / maxValue,
                x2 * imageWidth / maxValue,
                y2 * imageHeight / maxValue
        };
    }

    /**
     * Check if this bounding box contains a point.
     *
     * @param x the X coordinate
     * @param y the Y coordinate
     * @return true if the point is inside
     */
    public boolean contains(int x, int y) {
        return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    }

    /**
     * Check if this bounding box overlaps with another.
     *
     * @param other the other bounding box
     * @return true if they overlap
     */
    public boolean overlaps(BoundingBox other) {
        return !(x2 < other.x1 || other.x2 < x1 || y2 < other.y1 || other.y2 < y1);
    }

    /**
     * Calculate IoU (Intersection over Union) with another bounding box.
     *
     * @param other the other bounding box
     * @return IoU value between 0 and 1
     */
    public double iou(BoundingBox other) {
        int intersectX1 = Math.max(x1, other.x1);
        int intersectY1 = Math.max(y1, other.y1);
        int intersectX2 = Math.min(x2, other.x2);
        int intersectY2 = Math.min(y2, other.y2);

        if (intersectX2 < intersectX1 || intersectY2 < intersectY1) {
            return 0.0;
        }

        int intersectArea = (intersectX2 - intersectX1) * (intersectY2 - intersectY1);
        int unionArea = getArea() + other.getArea() - intersectArea;

        return unionArea > 0 ? (double) intersectArea / unionArea : 0.0;
    }

    @Override
    public String toString() {
        return String.format("BoundingBox[%d,%d,%d,%d]", x1, y1, x2, y2);
    }
}
