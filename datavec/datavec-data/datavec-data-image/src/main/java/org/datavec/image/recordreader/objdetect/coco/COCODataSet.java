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
package org.datavec.image.recordreader.objdetect.coco;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.io.Serializable;
import java.util.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
public class COCODataSet implements Serializable {

    private COCOImages images;
    private COCOCategories categories;
    private COCOLicenses licenses;
    private COCOSegmentations segmentations;
    private COCOInfo info;
    private COCOAnnotations annotations;

    @JsonIgnore
    private Map<String, List<ImageObject>>fileNameToImageObject;
    @JsonIgnore
    private boolean initialized;
    @JsonIgnore
    private  Map<Integer,COCOImage> idToImage;
    @JsonIgnore
    private  Map<Integer,COCOSegmentation> idToSegmentations;
    @JsonIgnore
    private Map<Integer,String> categoryIdToName;
    @JsonIgnore
    private Map<Integer,COCOAnnotation> annotationsById;

    public boolean hasImage(String fileName) {
        return fileNameToImageObject.containsKey(fileName);
    }

    public void init() {
        //create a list of file name to dynamically created image object
        idToImage = new HashMap<>();
        idToSegmentations = new HashMap<>();
        categoryIdToName = new HashMap<>();
        fileNameToImageObject = new HashMap<>();
        annotationsById = new HashMap<>();
        if(categories != null)
            for(COCOCategory cocoCategory : categories) {
                categoryIdToName.put(cocoCategory.getId(), cocoCategory.getName());
            }

        if(images != null)
            for(COCOImage cocoImage : images) {
                //note that some datasets might not initialize both, for consistency make sure they're the same
                cocoImage.setImageId(cocoImage.getId());
                idToImage.put(cocoImage.getId(),cocoImage);
            }

        if(segmentations != null)
            for(COCOSegmentation segmentation : segmentations) {
                idToSegmentations.put(segmentation.getImageId(),segmentation);
            }

        if(annotations != null) {
            for(COCOAnnotation annotation : annotations) {
                annotationsById.put(annotation.getImageId(),annotation);
            }
        }

        Set<Integer> removeEmpty = new HashSet<>();
        for(Integer id : idToImage.keySet()) {
            COCOImage cocoImage = idToImage.get(id);
            if(!idToSegmentations.isEmpty() && idToSegmentations.containsKey(id)) {
                COCOSegmentation segmentation = idToSegmentations.get(id);
                extractFromSegmentation(cocoImage, segmentation);

            } else if(!annotationsById.isEmpty() && annotationsById.containsKey(id)) {
                extractFromSegmentation(id, cocoImage);
            } else {
                removeEmpty.add(id);

                log.trace("No annotations found for image id " + id + " with file name " + cocoImage.getFileName() + " removing ");

            }
        }

        for(Integer id : removeEmpty) {
            fileNameToImageObject.remove(id);
            idToImage.remove(id);
        }


        initialized = true;
    }

    private void extractFromSegmentation(COCOImage cocoImage, COCOSegmentation segmentation) {
        double xTopLeft = segmentation.getBbox().get(0);
        double yTopLeft = segmentation.getBbox().get(1);
        double height = segmentation.getBbox().get(2);
        double width = segmentation.getBbox().get(3);
        //converts the height and width found in COCO to the equivalent in PASCAL VOC
        /**
         * LOOK AT THIS AND ADAPT: https://stackoverflow.com/questions/64581692/how-to-convert-pascal-voc-to-yolo
         */
        double xMax = xTopLeft + width;
        double yMax = yTopLeft + height;
        ImageObject imageObject = new ImageObject((int)xTopLeft,(int)yTopLeft,(int)xMax,(int)yMax,categoryIdToName.get(segmentation.getCategoryId()));
        if(!fileNameToImageObject.containsKey(cocoImage.getFileName())) {
            fileNameToImageObject.put(cocoImage.getFileName(),new ArrayList<>());
        }

        fileNameToImageObject.get(cocoImage.getFileName()).add(imageObject);
    }

    private void extractFromSegmentation(Integer id, COCOImage cocoImage) {
        COCOAnnotation cocoAnnotation = annotationsById.get(id);
        double xTopLeft = cocoAnnotation.getBbox().get(0);
        double yTopLeft = cocoAnnotation.getBbox().get(1);
        double height = cocoAnnotation.getBbox().get(2);
        double width = cocoAnnotation.getBbox().get(3);
        //converts the height and width found in COCO to the equivalent in PASCAL VOC
        /**
         * LOOK AT THIS AND ADAPT: https://stackoverflow.com/questions/64581692/how-to-convert-pascal-voc-to-yolo
         */
        double xMax = xTopLeft + width;
        double yMax = yTopLeft + height;
        ImageObject imageObject = new ImageObject((int)xTopLeft,(int)yTopLeft,(int)xMax,(int)yMax,categoryIdToName.get(cocoAnnotation.getCategoryId()));
        if(!fileNameToImageObject.containsKey(cocoImage.getFileName())) {
            fileNameToImageObject.put(cocoImage.getFileName(),new ArrayList<>());
        }

        fileNameToImageObject.get(cocoImage.getFileName()).add(imageObject);
    }

    public List<ImageObject> getImageForName(String fileName) {
        if(!initialized) {
            throw new IllegalStateException("Please call init first to retrieve image objects");
        }

        return fileNameToImageObject.get(fileName);
    }

}
