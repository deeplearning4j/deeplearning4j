package translate;


import caffe.Caffe.*;
import com.google.protobuf.CodedInputStream;
import org.springframework.core.io.ClassPathResource;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by jeffreytang on 7/9/15.
 */
public class PureProtoBufImplementation {


    public static void main(String args[]) throws IOException {

    }

    public static NetParameter parseCaffeModel(String caffeModelPath) throws IOException {

        String path = new ClassPathResource(caffeModelPath).getURI().getPath();
        InputStream is = new FileInputStream(path);
        NetParameter net = NetParameter.parseFrom(CodedInputStream.newInstance(is));
//        int layersCount = net.getLayersCount();
//        for (int i = 0; i < 2; i++) {
//            V1LayerParameter currentLayer = net.getLayers(i);
////            currentLayer.get
//            System.out.println(i);
//        }
    }
}
