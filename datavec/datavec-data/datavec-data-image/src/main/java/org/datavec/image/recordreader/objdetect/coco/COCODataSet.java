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
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
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
                idToImage.put(cocoImage.getId(),cocoImage);
            }

        if(segmentations != null)
            for(COCOSegmentation segmentation : segmentations) {
                idToSegmentations.put(segmentation.getId(),segmentation);
            }

        if(annotations != null) {
            for(COCOAnnotation annotation : annotations) {
                annotationsById.put(annotation.getImageId(),annotation);
            }
        }

        for(Integer id : idToImage.keySet()) {
            COCOImage cocoImage = idToImage.get(id);
            if(!idToSegmentations.isEmpty() && idToSegmentations.containsKey(id)) {
                COCOSegmentation segmentation = idToSegmentations.get(id);
                double xMin = segmentation.getBbox().get(0);
                double xMax = segmentation.getBbox().get(2);
                double yMin = segmentation.getBbox().get(1);
                double yMax = segmentation.getBbox().get(3);
                ImageObject imageObject = new ImageObject((int)xMin,(int)yMin,(int)xMax,(int)yMax,categoryIdToName.get(segmentation.getCategoryId()));
                if(!fileNameToImageObject.containsKey(cocoImage.getFileName())) {
                    fileNameToImageObject.put(cocoImage.getFileName(),new ArrayList<>());
                }

                fileNameToImageObject.get(cocoImage.getFileName()).add(imageObject);

            } else {
                //test dataset or doesn't contain annotation
                if(!fileNameToImageObject.containsKey(cocoImage.getFileName())) {
                    fileNameToImageObject.put(cocoImage.getFileName(),new ArrayList<>());
                }

                ImageObject imageObject = new ImageObject(0,0,0,0,"");
                fileNameToImageObject.get(cocoImage.getFileName()).add(imageObject);

            }
        }



        initialized = true;
    }

    public List<ImageObject> getImageForName(String fileName) {
        if(!initialized) {
            throw new IllegalStateException("Please call init first to retrieve image objects");
        }

        if(!fileNameToImageObject.containsKey(fileName)) {
            throw new IllegalArgumentException("No image found for file name " + fileName);
        }

        return fileNameToImageObject.get(fileName);
    }

}
