package org.datavec.image.transform;

import org.datavec.image.data.ImageWritable;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.Buffer;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by kepricon on 17. 5. 25.
 */
public class JsonYamlTest {
    @Test
    public void testJsonYamlImageTransformProcess() throws IOException {
        int seed = 12345;
        Random random = new Random(seed);

        //from org.bytedeco.javacpp.opencv_imgproc
        int COLOR_BGR2Luv = 50;
        int CV_BGR2GRAY = 6;


        ImageTransformProcess itp = new ImageTransformProcess.Builder()
                .colorConversionTransform(COLOR_BGR2Luv)
                .cropImageTransform(10)
                .equalizeHistTransform(CV_BGR2GRAY)
                .flipImageTransform(0)
                .resizeImageTransform(300, 300)
                .rotateImageTransform(30)
                .scaleImageTransform(3)
                .warpImageTransform((float)0.5)
//              .randomCropTransform(seed, 50, 50)
//              .filterImageTransform("noise=alls=20:allf=t+u,format=rgba", 100, 100, 4)
                .build();

        String asJson = itp.toJson();
        String asYaml = itp.toYaml();

//        System.out.println(asJson);
//        System.out.println("\n\n\n");
//        System.out.println(asYaml);

        ImageWritable img = TestImageTransform.makeRandomImage(0, 0, 3);
        ImageWritable imgJson = new ImageWritable(img.getFrame().clone());
        ImageWritable imgYaml = new ImageWritable(img.getFrame().clone());
        ImageWritable imgAll = new ImageWritable(img.getFrame().clone());

        ImageTransformProcess itpFromJson = ImageTransformProcess.fromJson(asJson);
        ImageTransformProcess itpFromYaml = ImageTransformProcess.fromYaml(asYaml);
//
        List<ImageTransform> transformList = itp.getTransformList();
        List<ImageTransform> transformListJson = itpFromJson.getTransformList();
        List<ImageTransform> transformListYaml = itpFromYaml.getTransformList();

        for (int i = 0; i < transformList.size(); i++) {
            ImageTransform it = transformList.get(i);
            ImageTransform itJson = transformListJson.get(i);
            ImageTransform itYaml = transformListYaml.get(i);

//            System.out.println(i + "\t" + it);

            img = it.transform(img);
            imgJson = itJson.transform(imgJson);
            imgYaml = itYaml.transform(imgYaml);

            if (it instanceof RandomCropTransform) {
                assertTrue(img.getFrame().imageHeight == imgJson.getFrame().imageHeight);
                assertTrue(img.getFrame().imageWidth == imgJson.getFrame().imageWidth);

                assertTrue(img.getFrame().imageHeight == imgYaml.getFrame().imageHeight);
                assertTrue(img.getFrame().imageWidth == imgYaml.getFrame().imageWidth);
            } else if (it instanceof FilterImageTransform) {
                assertEquals(img.getFrame().imageHeight, imgJson.getFrame().imageHeight);
                assertEquals(img.getFrame().imageWidth, imgJson.getFrame().imageWidth);
                assertEquals(img.getFrame().imageChannels, imgJson.getFrame().imageChannels);

                assertEquals(img.getFrame().imageHeight, imgYaml.getFrame().imageHeight);
                assertEquals(img.getFrame().imageWidth, imgYaml.getFrame().imageWidth);
                assertEquals(img.getFrame().imageChannels, imgYaml.getFrame().imageChannels);
            } else {
                assertEquals(img, imgJson);

                assertEquals(img, imgYaml);
            }
        }

        imgAll = itp.execute(imgAll);

        assertEquals(imgAll, img);
    }
}
