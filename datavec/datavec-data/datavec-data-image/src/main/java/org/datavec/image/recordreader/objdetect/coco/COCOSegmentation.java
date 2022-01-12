package org.datavec.image.recordreader.objdetect.coco;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class COCOSegmentation {

    private List<Double> segmentation;
    private double area;
    private List<Double> bbox;
    @JsonProperty("iscrowd")
    private int isCrowd;
    private int id;
    @JsonProperty("image_id")
    private int imageId;
    @JsonProperty("category_id")
    private int categoryId;
    @JsonProperty("num_keypoints")
    private int numKeyPoints;
    @JsonProperty("keypoints")
    private List<Integer> keyPoints;
}
