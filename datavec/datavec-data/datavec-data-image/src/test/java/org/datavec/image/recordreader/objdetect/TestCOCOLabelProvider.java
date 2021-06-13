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
package org.datavec.image.recordreader.objdetect;

import org.apache.commons.io.FileUtils;
import org.datavec.image.recordreader.objdetect.coco.*;
import org.datavec.image.recordreader.objdetect.impl.COCOLabelProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import static org.junit.Assert.*;

@Tag(TagNames.FILE_IO)
@Tag(TagNames.JAVA_ONLY)
public class TestCOCOLabelProvider {

    @Test
    public void testCaptions() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("captions_test_coco.json");
        File f = classPathResource.getFile();
        COCOLabelProvider cocoLabelProvider = new COCOLabelProvider(f.getAbsolutePath());
        COCODataSet cocoDataSet = cocoLabelProvider.getCocoDataSet();
        assertEquals(1,cocoDataSet.getImages().size());
        COCOImage cocoImage = cocoDataSet.getImages().get(0);
        assertEquals(640,cocoImage.getHeight());
        assertEquals(428,cocoImage.getWidth());
        assertEquals(1,cocoImage.getId());
        assertEquals(3,cocoImage.getLicense());
        assertEquals("COCO_val2014_000000074478.jpg",cocoImage.getFileName());
        assertEquals("http://farm3.staticflickr.com/2753/4318988969_653bb58b41_z.jpg",cocoImage.getFlickrUrl());
        assertNotNull(cocoDataSet.getInfo());
        COCOInfo cocoInfo = cocoDataSet.getInfo();
        assertEquals("COCO Consortium",cocoInfo.getContributor());
        assertEquals("COCO 2014 Dataset",cocoInfo.getDescription());
        assertEquals("http://cocodataset.org",cocoInfo.getUrl());
        assertEquals("1.0",cocoInfo.getVersion());

        assertNotNull(cocoDataSet.getLicenses());
        assertEquals(8,cocoDataSet.getLicenses().size());
        COCOLicense cocoLicense = cocoDataSet.getLicenses().get(0);
        assertEquals(1,cocoLicense.getId());
        assertEquals("Attribution-NonCommercial-ShareAlike License",cocoLicense.getName());
        assertEquals("http://creativecommons.org/licenses/by-nc-sa/2.0/",cocoLicense.getUrl());

        assertNotNull(cocoDataSet.getAnnotations());
        COCOAnnotation cocoAnnotation = cocoDataSet.getAnnotations().get(0);
        assertEquals(1,cocoAnnotation.getId());
        assertEquals(1,cocoAnnotation.getImageId());
        assertEquals("A bicycle replica with a clock as the front wheel.",cocoAnnotation.getCaption());

        List<ImageObject> imageObjectsForPath = cocoLabelProvider.getImageObjectsForPath("COCO_val2014_000000074478.jpg");
        assertFalse(imageObjectsForPath.isEmpty());
        ImageObject imageObject = imageObjectsForPath.get(0);


    }

    @Test
    public void testKeyPoints() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("keypoints_test_coco.json");
        File f = classPathResource.getFile();
        COCOLabelProvider cocoLabelProvider = new COCOLabelProvider(f.getAbsolutePath());
        COCODataSet cocoDataSet = cocoLabelProvider.getCocoDataSet();
        COCOAnnotations cocoAnnotations = cocoDataSet.getAnnotations();
        COCOAnnotation cocoAnnotation = cocoAnnotations.get(0);
        assertEquals(0,cocoAnnotation.getIsCrowd());
        assertEquals(1,cocoAnnotation.getImageId());
        assertEquals(1,cocoAnnotation.getCategoryId());
        assertEquals(28292.08625,cocoAnnotation.getArea(),1e-3);

        List<Float> bbox = cocoAnnotation.getBbox();
        assertEquals(4,bbox.size());
        assertEquals(267.03,cocoAnnotation.getBbox().get(0),1e-3);
        assertEquals(104.32,cocoAnnotation.getBbox().get(1),1e-3);
        assertEquals(229.19,cocoAnnotation.getBbox().get(2),1e-3);
        assertEquals(320,cocoAnnotation.getBbox().get(3),1e-3);


        COCOImage cocoImage = cocoDataSet.getImages().get(0);
        assertEquals("http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg",cocoImage.getCocoUrl());
        assertEquals("COCO_train2014_000000057870.jpg",cocoImage.getFileName());
        assertEquals(480,cocoImage.getHeight());
        assertEquals(640,cocoImage.getWidth());
        assertEquals(1,cocoImage.getId());


    }

    @Test
    public void testObjectRecognition() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("object_recognition_train.json");
        File f = classPathResource.getFile();
        COCOLabelProvider cocoLabelProvider = new COCOLabelProvider(f.getAbsolutePath());
        COCODataSet cocoDataSet = cocoLabelProvider.getCocoDataSet();
        COCOImage cocoImage = cocoDataSet.getImages().get(0);
        assertEquals(5,cocoImage.getLicense());
        assertEquals(480,cocoImage.getHeight());
        assertEquals(640,cocoImage.getWidth());
        assertEquals("COCO_train2014_000000057870.jpg",cocoImage.getFileName());
        assertEquals("http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg",cocoImage.getCocoUrl());
        COCOAnnotation cocoAnnotation = cocoDataSet.getAnnotations().get(0);
        assertEquals(54652.9556,cocoAnnotation.getArea(),1e-1);
        assertEquals(58,cocoAnnotation.getCategoryId());
        assertEquals(0,cocoAnnotation.getIsCrowd());
        assertEquals(1,cocoAnnotation.getImageId());
        assertEquals(1,cocoAnnotation.getSegmentation().size());
        assertEquals(36,cocoAnnotation.getSegmentation().get(0).size());
        assertEquals(1,cocoAnnotation.getId());


    }
}
