package org.deeplearning4j.aws.emr;

import lombok.Getter;

import java.util.Arrays;

public class S3Url {

    @Getter
    private String bucket;
    @Getter
    private String key;

    public S3Url(String url){
        if (!url.startsWith("s3://") || url.length() < 5){
            throw new IllegalArgumentException("S3 URL should start with s3://");
        }
        String payLoad = url.substring(5, url.length() - 1);
        String[] components = payLoad.split("/");
        if (components.length == 0) throw new IllegalArgumentException("Unrecognized S3 URL"+ url);
        bucket = components[0];

        StringBuilder sb = new StringBuilder();
        for(int i = 1; i< components.length; i++) {
            sb.append(components[i]);
            if (i < components.length - 1) sb.append("/");
        }
        key = sb.toString();
    }


    public String toString() {
        return String.format("s3://%s/%s", bucket, key);
    }

}
