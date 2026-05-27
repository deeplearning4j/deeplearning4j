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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;

import static org.junit.jupiter.api.Assumptions.assumeTrue;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@Slf4j
public class TestPage10ImageDiagnostics {

    private static final int TARGET_SIZE = 512;
    private static final int LONGEST_EDGE = 2048;
    private static final int RENDER_DPI = 150;
    private static final int PDF_PAGE = 10;

    private static File pdfFile;
    private static File outputDir;
    private static boolean initialized;

    private static final int ZOOM_X = 200;
    private static final int ZOOM_Y = 150;
    private static final int ZOOM_W = 400;
    private static final int ZOOM_H = 120;

    @BeforeAll
    public static void setup() throws Exception {
        if (initialized) {
            return;
        }

        String pdfPath = System.getProperty("vlm.test.pdf.path");
        pdfFile = pdfPath != null ? new File(pdfPath) : new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        assumeTrue(pdfFile.exists(), "PDF not found at " + pdfFile.getAbsolutePath()
                + ". Place pathfinder-mythic.pdf in platform-tests/ or set -Dvlm.test.pdf.path");

        outputDir = new File("/tmp/vlm-diagnostics");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        initialized = true;
        log.info("Image diagnostics output dir: {}", outputDir.getAbsolutePath());
    }

    @Test
    @DisplayName("Page-10 image pipeline diagnostics: save intermediate images at each stage")
    public void testPage10ImageDiagnostics() throws Exception {
        setup();

        BufferedImage original = renderPdfPage(pdfFile, PDF_PAGE, RENDER_DPI);
        saveImage(original, "01-original-rendered.png", original.getWidth(), original.getHeight());

        BufferedImage resized = ImageTiler.resizeLongestEdge(original, LONGEST_EDGE);
        saveImage(resized, "02-resized-longest-edge.png", resized.getWidth(), resized.getHeight());

        int width = resized.getWidth();
        int height = resized.getHeight();
        int numCols = 3;
        int numRows = 3;
        int optimalWidth = (int) Math.ceil((double) width / numCols);
        int optimalHeight = (int) Math.ceil((double) height / numRows);

        int tileX = 0;
        int tileY = 0;
        int tileW = Math.min(optimalWidth, width - tileX);
        int tileH = Math.min(optimalHeight, height - tileY);

        BufferedImage tile = resized.getSubimage(tileX, tileY, tileW, tileH);
        saveImage(tile, "03-tile-0-0.png", tileW, tileH);

        ImageTiler.ResizeResult tileResized = ImageTiler.resizeToFit(tile, TARGET_SIZE, TARGET_SIZE);
        saveImage(tileResized.image, "04-tile-0-0-resized.png", tileResized.width, tileResized.height);

        BufferedImage tilePaddedBlack = ImageTiler.padToSize(tileResized.image, TARGET_SIZE, TARGET_SIZE);
        saveImage(tilePaddedBlack, "05-tile-0-0-padded.png", TARGET_SIZE, TARGET_SIZE);

        BufferedImage tilePaddedWhite = padToSizeWithColor(tileResized.image, TARGET_SIZE, TARGET_SIZE, Color.WHITE);
        saveImage(tilePaddedWhite, "06-tile-0-0-padded-white.png", TARGET_SIZE, TARGET_SIZE);

        saveZoomCrop(original, "01-zoom.png", "original rendered");
        saveZoomCrop(resized, "02-zoom.png", "resized longest edge");
        saveZoomCrop(tile, "03-zoom.png", "tile 0,0 (before resize)");
        saveZoomCrop(tileResized.image, "04-zoom.png", "tile 0,0 resized");
        saveZoomCrop(tilePaddedBlack, "05-zoom.png", "tile 0,0 padded black");
        saveZoomCrop(tilePaddedWhite, "06-zoom.png", "tile 0,0 padded white");

        log.info("All diagnostic images saved to {}", outputDir.getAbsolutePath());
    }

    private static BufferedImage renderPdfPage(File pdf, int page, int dpi) throws IOException {
        try (PDDocument document = PDDocument.load(pdf)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(page, dpi, ImageType.RGB);
        }
    }

    private static void saveImage(BufferedImage image, String filename, int w, int h) throws IOException {
        File out = new File(outputDir, filename);
        ImageIO.write(image, "PNG", out);
        log.info("Saved {} → {} ({}x{})", filename, out.getAbsolutePath(), w, h);
    }

    private static void saveZoomCrop(BufferedImage image, String filename, String stageLabel) throws IOException {
        int srcW = image.getWidth();
        int srcH = image.getHeight();

        int cropX = Math.min(ZOOM_X, Math.max(0, srcW - ZOOM_W));
        int cropY = Math.min(ZOOM_Y, Math.max(0, srcH - ZOOM_H));
        int cropW = Math.min(ZOOM_W, srcW - cropX);
        int cropH = Math.min(ZOOM_H, srcH - cropY);

        if (cropW <= 0 || cropH <= 0) {
            log.warn("Cannot extract zoom crop for {} from {}x{} image", filename, srcW, srcH);
            return;
        }

        BufferedImage zoom = image.getSubimage(cropX, cropY, cropW, cropH);
        File out = new File(outputDir, filename);
        ImageIO.write(zoom, "PNG", out);
        log.info("Saved {} → {} (crop region: ({},{}) {}x{}, stage='{}')",
                filename, out.getAbsolutePath(), cropX, cropY, cropW, cropH, stageLabel);
    }

    private static BufferedImage padToSizeWithColor(
            BufferedImage image, int targetWidth, int targetHeight, Color padColor) {
        if (image.getWidth() == targetWidth && image.getHeight() == targetHeight) {
            return image;
        }

        BufferedImage padded = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = padded.createGraphics();
        g2d.setColor(padColor);
        g2d.fillRect(0, 0, targetWidth, targetHeight);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return padded;
    }
}
