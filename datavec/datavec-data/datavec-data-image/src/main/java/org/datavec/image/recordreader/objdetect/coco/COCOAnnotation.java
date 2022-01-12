package org.datavec.image.recordreader.objdetect.coco;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class COCOAnnotation implements Serializable {
    @JsonProperty("image_id")
    private int imageId;
    private int id;
    private String caption;
    private List<List<Float>> segmentation;
    @JsonProperty("iscrowd")
    private int isCrowd;
    private List<Float> bbox;
    private float area;
    @JsonProperty("category_id")
    private int categoryId;
    @JsonProperty("keypoints")
    private List<Integer> keyPoints;
    @JsonProperty("num_keypoints")
    private int numKeyPoints;

}
