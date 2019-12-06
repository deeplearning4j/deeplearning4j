/*
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.autodiff.samediff;

import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;

public class ConvConfigTests {

    @Test
    public void testDeConv2D(){
        DeConv2DConfig.builder().kH(2).kW(4).build();

        try{
            DeConv2DConfig.builder().kW(4).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            DeConv2DConfig.builder().kH(4).kW(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }
    }

    @Test
    public void testConv2D(){
        Conv2DConfig.builder().kH(2).kW(4).build();

        try{
            Conv2DConfig.builder().kW(4).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            Conv2DConfig.builder().kH(4).kW(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }
    }

    @Test
    public void testPooling2D(){
        Pooling2DConfig.builder().kH(2).kW(4).build();

        try{
            Pooling2DConfig.builder().kW(4).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            Pooling2DConfig.builder().kH(4).kW(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }
    }

    @Test
    public void testDeConv3D(){
        DeConv3DConfig.builder().kH(2).kW(4).kD(3).build();

        try{
            DeConv3DConfig.builder().kW(4).kD(3).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kD(3).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel depth"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).sD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride depth"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).pD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding depth"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }

        try{
            DeConv3DConfig.builder().kH(4).kW(3).kD(3).dD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation depth"));
        }
    }

    @Test
    public void testConv3D(){
        Conv3DConfig.builder().kH(2).kW(4).kD(3).build();

        try{
            Conv3DConfig.builder().kW(4).kD(3).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            Conv3DConfig.builder().kH(4).kD(3).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel depth"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).sD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride depth"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).pD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding depth"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }

        try{
            Conv3DConfig.builder().kH(4).kW(3).kD(3).dD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation depth"));
        }
    }



    @Test
    public void testPooling3D(){
        Pooling3DConfig.builder().kH(2).kW(4).kD(3).build();

        try{
            Pooling3DConfig.builder().kW(4).kD(3).kH(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel height"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kD(3).kW(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel width"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(0).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel depth"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).sH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride height"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).sW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride width"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).sD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride depth"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).pH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding height"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).pW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding width"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).pD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding depth"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).dH(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation height"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).dW(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation width"));
        }

        try{
            Pooling3DConfig.builder().kH(4).kW(3).kD(3).dD(-2).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Dilation depth"));
        }
    }



    @Test
    public void testConv1D(){
        Conv1DConfig.builder().k(2).paddingMode(PaddingMode.SAME).build();

        try{
            Conv1DConfig.builder().k(0).paddingMode(PaddingMode.SAME).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Kernel"));
        }

        try{
            Conv1DConfig.builder().k(4).s(-2).paddingMode(PaddingMode.SAME).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Stride"));
        }

        try{
            Conv1DConfig.builder().k(3).p(-2).paddingMode(PaddingMode.SAME).build();
            fail();
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().contains("Padding"));
        }
    }
}
