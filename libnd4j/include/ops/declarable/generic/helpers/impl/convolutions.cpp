/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
 ******************************************************************************/

//
// @author raver119@gmail.com, created on 07.10.2017.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/generic/helpers/convolutions.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcPadding2D(int& pH, int& pW, int oH, int oW, int iH, int iW, int kH, int kW, int sH, int sW, int dH, int dW) {
            int eKH, eKW;

            if (dH == 1 && dW == 1) {
                eKH = kH;
                eKW = kW;
            } else {
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pH = ((oH - 1) * sH + eKH - iH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pW = ((oW - 1) * sW + eKW - iW) / 2;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW) {

            int eKD, eKH, eKW;
            
            if (dD == 1 && dH == 1 && dW == 1) {
                eKD = kD;
                eKH = kH;
                eKW = kW;
            } else {
                eKD = kD + (kD - 1) * (dD - 1);
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pD = ((oD - 1) * sD + eKD - iD) / 2;       // Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pH = ((oH - 1) * sH + eKH - iH) / 2; 
            pW = ((oW - 1) * sW + eKW - iW) / 2;

        }

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::avgPool3DBP(NDArray<T>& gradO, NDArray<T>& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad) {
    
    T* pO = gradO.getBuffer();
    T* pI = gradI.getBuffer();

    const Nd4jLong bS = gradI.sizeAt(0);
    const Nd4jLong iC = gradI.sizeAt(1);
    const Nd4jLong iD = gradI.sizeAt(2);
    const Nd4jLong iH = gradI.sizeAt(3);
    const Nd4jLong iW = gradI.sizeAt(4);

    const Nd4jLong oD = gradO.sizeAt(2);
    const Nd4jLong oH = gradO.sizeAt(3);
    const Nd4jLong oW = gradO.sizeAt(4);        

    const Nd4jLong iStride1 = iD * iH * iW;
    const Nd4jLong oStride1 = oD * oH * oW;
    const Nd4jLong iStride0 = iC * iStride1;
    const Nd4jLong oStride0 = iC * oStride1;
    const Nd4jLong size0 = bS * iC;
        
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)        
    for (int s = 0; s < bS; ++s) {
        for (int k = 0; k < iC; ++k) {

            /* local pointers */
            T *ip = pI + s*iStride0 + k*iStride1;
            T *op = pO + s*oStride0 + k*oStride1;
            
#pragma omp parallel for simd                
            for (int i = 0; i < iStride1; i++)
                *(ip + i) = 0;

#pragma omp parallel for if(oStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            /* loop over output */
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++) {
                            
                        int cstart = ti * sD - pD;
                        int hstart = i  * sH - pH;
                        int wstart = j  * sW - pW;
                        int cend = nd4j::math::nd4j_min<int>(cstart + kD, iD + pD);
                        int hend = nd4j::math::nd4j_min<int>(hstart + kH, iH + pH);
                        int wend = nd4j::math::nd4j_min<int>(wstart + kW, iW + pW);
                        int pool_size = (cend -cstart) * (hend - hstart) * (wend - wstart);
                        cstart = nd4j::math::nd4j_max<int>(cstart, 0);
                        hstart = nd4j::math::nd4j_max<int>(hstart, 0);
                        wstart = nd4j::math::nd4j_max<int>(wstart, 0);
                        cend = nd4j::math::nd4j_min<int>(cend, iD);
                        hend = nd4j::math::nd4j_min<int>(hend, iH);
                        wend = nd4j::math::nd4j_min<int>(wend, iW);

                        int divide_factor;
                        if (count_include_pad)
                            divide_factor = pool_size;
                        else
                            divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                        /* scatter gradients out to footprint: */
                        T val  = *op++;
                        
                        for (int z = cstart; z < cend; z++)
                            for (int y = hstart; y < hend; y++)
                                for (int x = wstart; x < wend; x++)
                                    *(ip + z * iH * iW + y * iW + x) += val / divide_factor;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::avgPool3D(NDArray<T>& input, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad) {

    T* in  = input.getBuffer();
    T* out = output.getBuffer();
    
    const Nd4jLong bS = input.sizeAt(0);
    const Nd4jLong iC = input.sizeAt(1);
    const Nd4jLong iD = input.sizeAt(2);
    const Nd4jLong iH = input.sizeAt(3);
    const Nd4jLong iW = input.sizeAt(4);

    const Nd4jLong oD = output.sizeAt(2);
    const Nd4jLong oH = output.sizeAt(3);
    const Nd4jLong oW = output.sizeAt(4);    

    const Nd4jLong inStride1  = iD * iH * iW;
    const Nd4jLong outStride1 = oD * oH * oW;
    const Nd4jLong inStride0  = iC * inStride1;
    const Nd4jLong outStride0 = iC * outStride1;
    const Nd4jLong size0 = bS * iC;
        
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
    for(int s = 0; s < bS; ++s)  {            
        for (int k = 0; k < iC; k++) {
                
            /* local pointers. */
            T *ip = in  + s*inStride0  + k*inStride1;
            T *op = out + s*outStride0 + k*outStride1;
#pragma omp parallel for simd
            for (int i = 0; i < outStride1; ++i)
                *(op + i) = static_cast<T>(0.);

            /* loop over output */
#pragma omp parallel for if(outStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++) {

                        /* compute pool range. */
                        int cstart = ti * sD - pD;
                        int hstart = i  * sH - pH;
                        int wstart = j  * sW - pW;
                        int cend = nd4j::math::nd4j_min<int>(cstart + kD, iD + pD);
                        int hend = nd4j::math::nd4j_min<int>(hstart + kH, iH + pH);
                        int wend = nd4j::math::nd4j_min<int>(wstart + kW, iW + pW);
                        int pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);
                        cstart = nd4j::math::nd4j_max<int>(cstart, 0);
                        hstart = nd4j::math::nd4j_max<int>(hstart, 0);
                        wstart = nd4j::math::nd4j_max<int>(wstart, 0);
                        cend = nd4j::math::nd4j_min<int>(cend, iD);
                        hend = nd4j::math::nd4j_min<int>(hend, iH);
                        wend = nd4j::math::nd4j_min<int>(wend, iW);

                        int divide_factor;
                        if (count_include_pad)
                            divide_factor = pool_size;
                        else
                            divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                        /* compute local sum: */
                        T sum = static_cast<T>(0.);

                        for (int z = cstart; z < cend; z++) 
                            for (int y = hstart; y < hend; y++) 
                                for (int x = wstart; x < wend; x++) 
                                    sum +=  *(ip + z * iW * iH + y * iW + x);

                        /* set output to local max */
                        *op++ += sum / divide_factor;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_dilatedMaxPool3D_bp(T *gradInput_p, T *gradOutput_p, T *indBuff, Nd4jLong nslices, Nd4jLong  itime, Nd4jLong  iwidth, Nd4jLong  iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            for (int k = 0; k < nslices; k++)
            {
                T *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
                T *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
                T *indz_p_k = indBuff + k * otime * owidth * oheight;

                /* calculate max points */
                long ti, i, j;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* retrieve position of max */
                            T * indP = &indz_p_k[ti * oheight * owidth + i * owidth + j];
                            long maxti = ((unsigned char*)(indP))[0] * dilationT + ti * dT - pT;
                            long maxi  = ((unsigned char*)(indP))[1] * dilationH + i * dH - pH;
                            long maxj  = ((unsigned char*)(indP))[2] * dilationW + j * dW - pW;

                            if (maxti != -1) {
                                /* update gradient */
                                gradInput_p_k[maxti * iheight * iwidth + maxi * iwidth + maxj] += gradOutput_p_k[ti * oheight * owidth + i * owidth + j];
                            }
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_dilatedMaxPool3D(T *input_p, T *output_p, T *indBuff, Nd4jLong nslices, Nd4jLong itime, Nd4jLong iwidth, Nd4jLong iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int kD, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            Nd4jLong k;
#pragma omp parallel for private(k)
            for (k = 0; k < nslices; k++)
            {
                /* loop over output */
                Nd4jLong i, j, ti;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* local pointers */

                            Nd4jLong start_t = ti * dT - pT;
                            Nd4jLong start_h = i * dH - pH;
                            Nd4jLong start_w = j * dW - pW;

                            Nd4jLong kernel_d = nd4j::math::nd4j_min<Nd4jLong>(kD, kD + start_t);
                            Nd4jLong kernel_h = nd4j::math::nd4j_min<Nd4jLong>(kH, kH + start_h);
                            Nd4jLong kernel_w = nd4j::math::nd4j_min<Nd4jLong>(kW, kW + start_w);

                            while(start_t < 0)
                                start_t += dilationT;
                            while(start_h < 0)
                                start_h += dilationH;
                            while(start_w < 0)
                                start_w += dilationW;

                            T *ip = input_p + k * itime * iwidth * iheight + start_t * iwidth * iheight + start_h * iwidth + start_w;
                            T *op = output_p + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;
                            T *indP = indBuff + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;

                            /* compute local max: */
                            T maxval = -MAX_FLOAT;
                            int x,y,z;
                            int mx, my, mz;
                            mx = my = mz = -1;

                            for (z = 0; z < kernel_d; z++)
                            {
                                for (y = 0; y < kernel_h; y++)
                                {
                                    for (x = 0; x < kernel_w; x++)
                                    {
                                        if ((start_t + z * dilationT < itime) && (start_h + y * dilationH < iheight) && (start_w + x * dilationW < iwidth))
                                        {
                                            T val = *(ip + z * dilationT * iwidth * iheight + y * dilationH * iwidth + x * dilationW);
                                            if (val > maxval)
                                            {
                                                maxval = val;
                                                // Store indices w.r.t the kernel dimension
                                                mz = z + (kD - kernel_d);
                                                my = y + (kH - kernel_h);
                                                mx = x + (kW - kernel_w);
                                            }
                                        }
                                    }
                                }
                            }

                            // set max values
                            ((unsigned char*)(indP))[0] = mz;
                            ((unsigned char*)(indP))[1] = my;
                            ((unsigned char*)(indP))[2] = mx;
                            ((unsigned char*)(indP))[3] = 0;

                            /* set output to local max */
                            *op = maxval;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::validXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tot = (it - kt) / st + 1;
            Nd4jLong tor = (ir - kr) / sr + 1;
            Nd4jLong toc = (ic - kc) / sc + 1;

            Nd4jLong zz, xx, yy;

            for (zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_;
                        T sum = 0;
                        Nd4jLong kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                for(kx = 0; kx < kc; kx++) {
                                    sum += pi_[kx]*pw_[kx];
                                }
                                pi_ += ic; /* next input line */
                                pw_ += kc; /* next mask line */
                            }
                            pi_ += (ir-kr)*ic; /* next input slice */
                        }
                        /* Update output */
                        *r_++ += sum*alpha;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::validConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tot = (it - kt) / st + 1;
            Nd4jLong tor = (ir - kr) / sr + 1;
            Nd4jLong toc = (ic - kc) / sc + 1;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_ + kt*kr*kc - 1;
                        T sum = 0;
                        Nd4jLong kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                for(kx = 0; kx < kc; kx++) {
                                    sum += pi_[kx]*pw_[-kx];
                                }
                                pi_ += ic; /* next input line */
                                pw_ -= kc; /* next mask line */
                            }
                            pi_ += (ir-kr)*ic; /* next input slice */
                        }
                        /* Update output */
                        *r_++ += alpha*sum;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::fullConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tor = (ir - 1) * sr + kr;
            Nd4jLong toc = (ic - 1) * sc + kc;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz*st*tor*toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_;
                        Nd4jLong kz, kx, ky;
                        /* printf("Output Plane : %ld,%ld,%ld, input val=%g\n",zz,yy,xx,*t_); */
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                T z = *t_ * alpha;
                                for(kx = 0; kx < kc; kx++) {
                                    /* printf("o=%g,k=%g," , po_[kx],pw_[kx]); */
                                    po_[kx] += z * pw_[kx];
                                    /* printf("o=%g " , po_[kx]); */
                                }
                                /* printf("\n"); */
                                po_ += toc; /* next input line */
                                pw_ += kc; /* next mask line */
                            }
                            po_ += (tor-kr)*toc; /* next output slice */
                            /* printf("\n"); */
                        }
                        t_++;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::fullXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tor = (ir - 1) * sr + kr;
            Nd4jLong toc = (ic - 1) * sc + kc;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz * st * tor * toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_ + kt*kr*kc -1;
                        Nd4jLong kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                T z = *t_ * alpha;
                                for(kx = 0; kx < kc; kx++) {
                                    po_[kx] += z * pw_[-kx];
                                }
                                po_ += toc; /* next input line */
                                pw_ -= kc; /* next mask line */
                            }
                            po_ += (tor-kr)*toc; /* next output slice */
                        }
                        t_++;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jLong ConvolutionUtils<T>::convsize(Nd4jLong x, Nd4jLong k, Nd4jLong s, const char* vf) {
            if (*vf == 'V')
                return (x-k)/s + 1;
            else
                return (x-1)*s + k;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jStatus ConvolutionUtils<T>::conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_,
                                       Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol, const char *vf, const char *xc) {

            Nd4jLong nInputPlane, nInputDepth, nInputRows, nInputCols;
            Nd4jLong nKernelDepth, nKernelRows, nKernelCols;
            Nd4jLong nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
            Nd4jLong istride0, kstride0, kstride1;
            NDArray<T> *input;
            NDArray<T> *kernel;
            T* input_data;
            T* weight_data;
            T* output_data;
            Nd4jLong nelem;
            Nd4jLong k, i;

            if (t_->rankOf() != 4)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;

            if (k_->rankOf() != 5)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;

            if (sdepth < 1 || srow < 1 || scol < 1)
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            if (!(*vf == 'V' || *vf == 'F'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            if (!(*xc == 'X' || *xc == 'C'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            bool kD = false;
            input = t_->isContiguous() ? t_ : t_->dup(t_->ordering());
            if (!(k_->stridesOf()[4] == 1 || k_->stridesOf()[3] == k_->sizeAt(4))) {
                kernel = k_->isContiguous() ? k_ : k_->dup(k_->ordering());
                kD = true;
            } else {
                kernel = k_;
            }


            nInputPlane = input->sizeAt(0);
            istride0    = input->stridesOf()[0];
            nInputDepth = input->sizeAt(1);
            nInputRows  = input->sizeAt(2);
            nInputCols  = input->sizeAt(3);

            kstride0    = kernel->stridesOf()[0];
            kstride1    = kernel->stridesOf()[1];
            nKernelDepth = kernel->sizeAt(2);
            nKernelRows = kernel->sizeAt(3);
            nKernelCols = kernel->sizeAt(4);
            nOutputPlane = kernel->sizeAt(0);

            if (kernel->sizeAt(1) != nInputPlane)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;


            if (!((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            nOutputDepth = convsize(nInputDepth, nKernelDepth, sdepth, vf);
            nOutputRows = convsize(nInputRows, nKernelRows, srow, vf);
            nOutputCols = convsize(nInputCols, nKernelCols, scol, vf);

            nelem = r_->lengthOf();

            if (r_->sizeAt(0) != nOutputPlane || r_->sizeAt(1) != nOutputDepth || r_->sizeAt(2) != nOutputRows || r_->sizeAt(3)!= nOutputCols) {
                nd4j_printf("Failed at r_ size: {%i, %i, %i, %i} vs {}", r_->sizeAt(0), r_->sizeAt(1), r_->sizeAt(2), r_->sizeAt(3), nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);
                throw "Boom";
                //return ND4J_STATUS_BAD_DIMENSIONS;
            }

            if (nelem == 0 || beta == (T) 0.0f || nelem != r_->lengthOf()) {
                r_->assign((T) 0.0f);
            }
            else if (beta != (T) 1.0f) // stupid comparison
                r_->template applyScalar<simdOps::Multiply<T>>(beta);


            input_data = input->getBuffer();
            weight_data = kernel->getBuffer();
            output_data = r_->getBuffer();

            for(k = 0; k < nOutputPlane; k++) {
                for(i = 0; i < nInputPlane; i++) {
                    /* get kernel */
                    T* ptr_weight = weight_data + k*kstride0 + i*kstride1;
                    /* get input */
                    T* pIn = input_data + i*istride0;

                    /* do image, kernel convolution */
                    ConvolutionUtils<T>::conv3D(output_data,
                           alpha,
                           pIn,  nInputDepth, nInputRows,  nInputCols,
                           ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                           sdepth, srow, scol, vf, xc);
                }
                /* Next output plane */
                output_data += nOutputDepth*nOutputCols*nOutputRows;
            }

            if (kD)
                delete kernel;

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jStatus ConvolutionUtils<T>::conv3D(T* output_data,
                                     T alpha,
                                     T* pIn, Nd4jLong nInputDepth, Nd4jLong nInputRows, Nd4jLong nInputCols,
                                     T* ptr_weight, Nd4jLong nKernelDepth, Nd4jLong nKernelRows, Nd4jLong nKernelCols,
                                     Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol,
                                     const char *vf, const char *xc) {

            if (!(*vf == 'V' || *vf == 'F'))
                return ND4J_STATUS_BAD_PARAMS;

            if (!(*xc == 'X' || *xc == 'C'))
                return ND4J_STATUS_BAD_PARAMS;


            if (*vf == 'F')
                if (*xc == 'X') {
                    ConvolutionUtils<T>::fullXCorr3Dptr(output_data,
                                                 alpha,
                                                 pIn, nInputDepth, nInputRows,  nInputCols,
                                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                 sdepth, srow, scol);
                } else {
                    ConvolutionUtils<T>::fullConv3Dptr(output_data,
                                                alpha,
                                                pIn, nInputDepth, nInputRows,  nInputCols,
                                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                sdepth, srow, scol);
                }
            else
            if (*xc == 'X') {
                ConvolutionUtils<T>::validXCorr3Dptr(output_data,
                                              alpha,
                                              pIn, nInputDepth, nInputRows,  nInputCols,
                                              ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                              sdepth, srow, scol);
            } else {
                ConvolutionUtils<T>::validConv3Dptr(output_data,
                                             alpha,
                                             pIn, nInputDepth, nInputRows,  nInputCols,
                                             ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                             sdepth, srow, scol);
            }

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
// calculation of output height and width in 2D pooling procedure
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
            if(isSameMode > 0) {
                oH = (int) nd4j::math::nd4j_ceil(iH * 1.f / sH);
                oW = (int) nd4j::math::nd4j_ceil(iW * 1.f / sW);
            }
            else {
                oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
                oW = (iW - (kW + (kW-1)*(dW-1)) + 2*pW)/sW + 1;
            }
        }

//////////////////////////////////////////////////////////////////////////
// calculation of output depth, height and width in conv3d procedure        
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode) {

            if(!isSameMode) {                                           // valid
                
                oD = (iD - (kD + (kD - 1) * (dD - 1)) + 2 * pD) / sD + 1;
                oH = (iH - (kH + (kH - 1) * (dH - 1)) + 2 * pH) / sH + 1;
                oW = (iW - (kW + (kW - 1) * (dW - 1)) + 2 * pW) / sW + 1;
            }
            else {                                                      // same
                
                oD = (int) nd4j::math::nd4j_ceil(iD * 1.f / sD);
                oH = (int) nd4j::math::nd4j_ceil(iH * 1.f / sH);
                oW = (int) nd4j::math::nd4j_ceil(iW * 1.f / sW);
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
            
            if (isSameMode) {
                oH = sH * iH;
                oW = sW * iW;
            } 
            else {
                int ekH, ekW;
                if (dH == 1 && dW == 1) {
                    ekH = kH;
                    ekW = kW;
                } else {
                    ekH = kH + (kH - 1) * (dH - 1);
                    ekW = kW + (kW - 1) * (dW - 1);
                }

                oH = sH * (iH - 1) + ekH - 2 * pH;
                oW = sW * (iW - 1) + ekW - 2 * pW;
            }
        }
       
//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizeDeconv3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode) {
            
            if (isSameMode) {
                oD = sD * iD;
                oH = sH * iH;
                oW = sW * iW;
            } 
            else {
                int ekD, ekH, ekW;
                if (dD == 1 && dH == 1 && dW == 1) {
                    ekD = kD;
                    ekH = kH;
                    ekW = kW;
                } else {
                    ekD = kD + (kD - 1) * (dD - 1);
                    ekH = kH + (kH - 1) * (dH - 1);
                    ekW = kW + (kW - 1) * (dW - 1);
                }

                oD = sD * (iD - 1) + ekD - 2 * pD;
                oH = sH * (iH - 1) + ekH - 2 * pH;
                oW = sW * (iW - 1) + ekW - 2 * pW;
            }
        }


//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] always
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    indWiC = 2; indWoC = 3; indWkH = 0;

    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indOoH = 1; 
    }
    else {        
        indIOioC = 1; indIiH = 2; indOoH = 2;
    }    

    bS = inShapeInfo[1];                          // batch size
    iC = inShapeInfo[indIOioC+1];                 // input channels        
    iH = inShapeInfo[indIiH+1];                   // input height
    iW = inShapeInfo[indIiH+2];                   // input width
    oC = outShapeInfo[indIOioC+1];                // output channels
    oH = outShapeInfo[indOoH+1];                  // output height
    oW = outShapeInfo[indOoH+2];                  // output width    
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv2d(const bool isNCHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    getSizesAndIndexesConv2d(isNCHW, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD) {
    
    // input   [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    // weights [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)    
    // output  [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    if(!isNCDHW) {
        indIOioC = 4; indIOioD = 1; indWkD = 0; indWoC = 4; indWiC = 3; 
    }
    else {        
        indIOioC = 1; indIOioD = 2; indWkD = 2; indWoC = 0; indWiC = 1;
    }    

    bS = input.sizeAt(0);                          // batch size
    iC = input.sizeAt(indIOioC);                   // input channels        
    iD = input.sizeAt(indIOioD);                   // input depth
    iH = input.sizeAt(indIOioD+1);                 // input height
    iW = input.sizeAt(indIOioD+2);                 // input width
    oC = output.sizeAt(indIOioC);                  // output channels    
    oD = output.sizeAt(indIOioD);                  // output depth
    oH = output.sizeAt(indIOioD+1);                // output height
    oW = output.sizeAt(indIOioD+2);                // output width    

}
 
//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::conv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, oC] always
    NDArray<T> *bias    = inArrs[2];                                    // [oC]
    
    // output [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
                                         
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
    
    std::vector<int> weightsAxesForDot = {indWiC, indWkH, indWkH+1};                                                        // iC, kH, kW
    
    std::vector<int> permutForOutput;
    if(!isNCHW)
        input = input->permute({0, 3, 1, 2});                                       // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC
    else
        permutForOutput = {0, indOoH, indOoH+1, indIOioC};                          // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
     
    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        

    //----- calculation of output -----//
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, (T)0.f, (T)0.f});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                    // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper<T>::tensorDot(&columns, weights, output, {1,2,3}, weightsAxesForDot, permutForOutput); // [bS, iC, kH, kW, oH, oW] x [kH, kW, iC, oC]/[oC, iC, kH, kW] = [bS, oH, oW, oC]

    //----- add biases if required -----//
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                
}

#ifdef HAVE_MKLDNN
using namespace mkldnn;

template <typename T>
void ConvolutionUtils<T>::mkldnn_conv2d(MKLDNNStream<T> &stream, const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, oC] always
    NDArray<T> *bias    = inArrs[2];                                    // [oC]

    // output [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 1-NCHW,  0-NHWC

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    if (stream.checkAndReset(inArrs, {output}, {}, intArgs)) {
        mkldnn::memory::dims conv_src_tz = { bS, iC, iH, iW };
        mkldnn::memory::dims conv_weights_tz = { oC, iC, kH, kW };
        mkldnn::memory::dims conv_bias_tz = { oC };
        mkldnn::memory::dims conv_dst_tz = { bS, oC, oH, oW };
        mkldnn::memory::dims conv_strides = { sH, sW };
        mkldnn::memory::dims conv_padding = { pH, pW };
        mkldnn::memory::dims conv_padding_r = { (oH - 1) * sH - iH + kH - pH,
                                                (oW - 1) * sW - iW + kW - pW };

        auto type = mkldnn::memory::data_type::f32;
        auto format = isNCHW ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
        auto formatw = mkldnn::memory::format::hwio;
        auto conv_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
        auto conv_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::x);
        auto conv_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
        auto conv_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, format);

        auto conv_desc = bias != nullptr
                ? convolution_forward::desc(prop_kind::forward,
                        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                        conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                : convolution_forward::desc(prop_kind::forward,
                        convolution_direct, conv_src_md, conv_weights_md,
                        conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

        auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, stream.getEngine());
        auto conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc(), input->buffer());
        auto conv_weights_memory = mkldnn::memory(conv_prim_desc.weights_primitive_desc(), weights->buffer());
        auto conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_primitive_desc(), output->buffer());
        if (bias != nullptr) {
            auto conv_bias_memory = mkldnn::memory(conv_prim_desc.bias_primitive_desc(), bias->buffer());
            stream.setMemory({conv_src_memory, conv_weights_memory, conv_bias_memory, conv_dst_memory});
            stream.setOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_bias_memory, conv_dst_memory));
        } else {
            stream.setMemory({conv_src_memory, conv_weights_memory, conv_dst_memory});
            stream.setOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_dst_memory));
        }
    }

    stream.submitAndWait();
}
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::conv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                        // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                        // [kH, kW, iC, oC] always
    NDArray<T> *bias    = inArrs[2];                        // [oC]
    NDArray<T> *gradO   = inArrs[3];                        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    NDArray<T> *gradW = outArrs[1];                         // [kH, kW, iC, oC] always
    NDArray<T> *gradB = outArrs[2];                         // [oC]
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::vector<int> gradOaxesForDot; 

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW        
    }
    else
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
    
    if(isSameMode)                       // SAME        
        calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    
    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());
    
    // ----- calculation of gradW ----- // 
    if(gradW) {
        std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, (T)0.f, (T)0.f});
        input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                    // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]        
        nd4j::MmulHelper<T>::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, {2, 0, 1, 3});       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
    }

    // ----- calculation of gradB ----- // 
    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper<T>::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, {2, 3, 1, 0, 4, 5});       // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                            // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
  
    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::depthwiseConv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, mC] always
    NDArray<T> *bias    = inArrs[2];                                    // [oC] = iC*mC
    
    // output is [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)        
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height 
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier
    
    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
    std::vector<std::vector<Nd4jLong>> modifOutput;
    std::vector<Nd4jLong> outReShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());                
    NDArray<T>* outputReshaped = output->reshape(output->ordering(), outReShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, (T)0.f, (T)0.f});

    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                                 // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::MmulHelper<T>::tensorDot(&columns, weights, outputReshaped, modifColumns, {{2,0,1,3},{iC,kH*kW,mC}}, modifOutput);    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                  
    
    delete outputReshaped;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::depthwiseConv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                            // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = inArrs[1];                            // [kH, kW, iC, mC] always
    NDArray<T> *bias    = inArrs[2];                            // [oC] = [iC*mC]
    NDArray<T> *gradO   = inArrs[3];                            // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                             // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = outArrs[1];                             // [kH, kW, iC, mC] always
    NDArray<T> *gradB = outArrs[2];                             // [oC]        
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier    

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
    std::vector<std::vector<Nd4jLong>> modifGradO1, modifGradO2;
    std::vector<Nd4jLong> gradOreShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        
    NDArray<T>* gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, (T)0.f, (T)0.f});
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    
    // ----- calculation of gradW and gradB ----- //            
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::MmulHelper<T>::tensorDot(&columns, gradOreshaped, gradW, modifColumns, modifGradO1, {{2,0,1,3},{iC,kH*kW,mC}});  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//                
    nd4j::MmulHelper<T>::tensorDot(weights, gradO, &columns, {{2,0,1,3},{iC,kH*kW,mC}}, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]    
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                     // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }

    delete gradOreshaped;      
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::sconv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input        = inArrs[0];                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *weightsDepth = inArrs[1];                                           // [kH, kW, iC, mC]  always
    NDArray<T> *weightsPoint = inArrs[2];                                           // [1, 1, iC*mC, oC] always
    NDArray<T> *bias         = inArrs[3];                                           // [oC], oC = iC*mC if weightsPoint=nullptr
    
    // output is [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    NDArray<T>* outputDepth = output;
    if(weightsPoint)                        // if pointwise convolution is expected
        outputDepth = new NDArray<T>(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}));    

    // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //    
    ConvolutionUtils<T>::depthwiseConv2d({input, weightsDepth, weightsPoint ? nullptr : bias}, outputDepth, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, isNCHW});                                   
    
    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
    if (weightsPoint) {
        ConvolutionUtils<T>::conv2d({outputDepth, weightsPoint, bias}, output, {1,1, 1,1, 0,0, 1,1, isSameMode, isNCHW});             // in this case oH=iH, oW=iW                
        delete outputDepth;
    }
}


//////////////////////////////////////////////////////////////////////////
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
template <typename T>
void ConvolutionUtils<T>::vol2col(NDArray<T>& volume, NDArray<T>& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const Nd4jLong bS = volume.sizeAt(0);
    const Nd4jLong iC = volume.sizeAt(1);
    const Nd4jLong iD = volume.sizeAt(2);
    const Nd4jLong iH = volume.sizeAt(3);
    const Nd4jLong iW = volume.sizeAt(4);
    const Nd4jLong kD = columns.sizeAt(2);
    const Nd4jLong kH = columns.sizeAt(3);
    const Nd4jLong kW = columns.sizeAt(4);
    const Nd4jLong oD = columns.sizeAt(5);
    const Nd4jLong oH = columns.sizeAt(6);
    const Nd4jLong oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];    
    
    T* volBuff = volume.getBuffer();
    T* colBuff = columns.getBuffer();

    T *col, *vol;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo()))

#pragma omp parallel for schedule(static) proc_bind(close) private(col, vol, volDep, volRow, volCol)
    for (int b = 0; b < bS; b++) {
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    
                                
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col = static_cast<T>(0.);
                                    else 
                                        *col = *vol;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol, col, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col = static_cast<T>(0.);
                                    else 
                                        *col = *vol;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to [bS, iC, iD, iH, iW]
template <typename T>
void ConvolutionUtils<T>::col2vol(NDArray<T>& columns, NDArray<T>& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const Nd4jLong bS = volume.sizeAt(0);
    const Nd4jLong iC = volume.sizeAt(1);
    const Nd4jLong iD = volume.sizeAt(2);
    const Nd4jLong iH = volume.sizeAt(3);
    const Nd4jLong iW = volume.sizeAt(4);
    const Nd4jLong kD = columns.sizeAt(2);
    const Nd4jLong kH = columns.sizeAt(3);
    const Nd4jLong kW = columns.sizeAt(4);
    const Nd4jLong oD = columns.sizeAt(5);
    const Nd4jLong oH = columns.sizeAt(6);
    const Nd4jLong oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];    
    
    T* volBuff = volume.getBuffer();
    T* colBuff = columns.getBuffer();

    // initial zeroing of volume content
    volume.assign(0.f);

    T* col, *vol;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) 

#pragma omp parallel for schedule(static) proc_bind(close) private(col, vol, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {        
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    

                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;

                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;

                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        *vol += *col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol, col, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        *vol += *col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling2d(const NDArray<T>& input, NDArray<T>& output, const int factorH, const int factorW, const bool isNCHW) {
    // input  has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC) 
    // output has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = input.sizeAt(dimIH) * input.sizeAt(dimIH+1);
    // const int size1 = factorH * factorW;

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int ih = 0; ih < input.sizeAt(dimIH); ++ih) {
        for(int iw = 0; iw < input.sizeAt(dimIH+1); ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 

// #pragma omp parallel for if(size1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indOut) 
            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {
                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    auto i = input(indIn);
                    auto o = output(indOut);
                    o.assign(i);
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling3d(const NDArray<T>& input, NDArray<T>& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;    
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = input.sizeAt(dimID) * input.sizeAt(dimID+1) * input.sizeAt(dimID+2);
    // const int size1 = factorD * factorH * factorW;

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int id = 0; id < input.sizeAt(dimID); ++id) {
        for(int ih = 0; ih < input.sizeAt(dimID+1); ++ih) {
            for(int iw = 0; iw < input.sizeAt(dimID+2); ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;

// #pragma omp parallel for if(size1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indOut) 
            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            auto i = input(indIn);                    
                            auto o = output(indOut);
                            o.assign(i);
                        }
                    }
                }
            }
        }
    }    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling2dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCHW) {
    // gradO has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    // gradI has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)     
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int factorH = gradO.sizeAt(dimIH)   / gradI.sizeAt(dimIH);
    const int factorW = gradO.sizeAt(dimIH+1) / gradI.sizeAt(dimIH+1);
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = gradI.sizeAt(dimIH) * gradI.sizeAt(dimIH+1);

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int ih = 0; ih < gradI.sizeAt(dimIH); ++ih) {
        for(int iw = 0; iw < gradI.sizeAt(dimIH+1); ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 
            NDArray<T> subGradI = gradI(indIn);

            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    auto o = gradO(indOut);
                    if(!fh && !fw) {                        
                        subGradI.assign(o);
                    }
                    else
                        subGradI += o;
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling3dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;
    const int factorD = gradO.sizeAt(dimID)   / gradI.sizeAt(dimID);
    const int factorH = gradO.sizeAt(dimID+1) / gradI.sizeAt(dimID+1);
    const int factorW = gradO.sizeAt(dimID+2) / gradI.sizeAt(dimID+2);
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = gradI.sizeAt(dimID) * gradI.sizeAt(dimID+1) * gradI.sizeAt(dimID+2);

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3) firstprivate(indOut, indIn) 
    for(int id = 0; id < gradI.sizeAt(dimID); ++id) {
        for(int ih = 0; ih < gradI.sizeAt(dimID+1); ++ih) {
            for(int iw = 0; iw < gradI.sizeAt(dimID+2); ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;
                NDArray<T> subGradI = gradI(indIn);

            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            auto o = gradO(indOut);
                            if(!fd && !fh && !fw)
                                subGradI.assign(o);
                            else
                                subGradI += o;
                        }
                    }
                }
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::maxPool2d(NDArray<T>* input, NDArray<T>* output, const std::vector<int>& params, NDArray<T>* indices) {

    int kH = params[0];
    int kW = params[1];
    int sH = params[2];
    int sW = params[3];
    int pH = params[4];
    int pW = params[5];
    int dH = params[6];
    int dW = params[7];

    const Nd4jLong bS  = input->sizeAt(0);
    const Nd4jLong inD = input->sizeAt(1);
    const Nd4jLong iH = input->sizeAt(2);
    const Nd4jLong iW = input->sizeAt(3);
    const Nd4jLong oH  = output->sizeAt(2);
    const Nd4jLong oW  = output->sizeAt(3);

    const bool isSameMode = params[8];

    if (isSameMode)
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, pH, pW);                    

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; poolingMode; 9 - divisor;
    std::vector<T> argT = {(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T)dW, 0., 1.};

    ConvolutionUtils<T>::pooling2d(*input, *output, argT.data());
    
    if (indices != nullptr) {
        // for max_pool_with_argmax 
        int part = input->lengthOf() / bS;
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
        for (int b = 0; b < input->lengthOf(); b += part) 
            for (int i = 0; i < part; i++)
                (*indices)(b+i) = i;                
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling2d(NDArray<T>& input, NDArray<T>& output, const T* extraParams) {
    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]
    T* out = output.getBuffer();
    T* in  = input.getBuffer();

    const Nd4jLong kH = (int)extraParams[0];
    const Nd4jLong kW = (int)extraParams[1];
    const Nd4jLong sH = (int)extraParams[2];
    const Nd4jLong sW = (int)extraParams[3];
    const Nd4jLong pH = (int)extraParams[4];
    const Nd4jLong pW = (int)extraParams[5];    
    const Nd4jLong dH = (int)extraParams[6];
    const Nd4jLong dW = (int)extraParams[7];
    Nd4jLong poolingMode = (int)extraParams[8];
    T extraParam0 = extraParams[9];

    const Nd4jLong kHEff = kH + (kH-1)*(dH-1);
    const Nd4jLong kWEff = kW + (kW-1)*(dW-1);

    const Nd4jLong bS = input.sizeAt(0);
    const Nd4jLong iC = input.sizeAt(1);    
    const Nd4jLong iH = input.sizeAt(2);
    const Nd4jLong iW = input.sizeAt(3);    
    const Nd4jLong oH = output.sizeAt(2);
    const Nd4jLong oW = output.sizeAt(3);
    const Nd4jLong iStride0 = input.stridesOf()[0];
    const Nd4jLong iStride1 = input.stridesOf()[1];
    const Nd4jLong iStride2 = input.stridesOf()[2];
    const Nd4jLong iStride3 = input.stridesOf()[3];    
    const Nd4jLong oStride0 = output.stridesOf()[0];
    const Nd4jLong oStride1 = output.stridesOf()[1];
    const Nd4jLong oStride2 = output.stridesOf()[2];
    const Nd4jLong oStride3 = output.stridesOf()[3];
    
    const Nd4jLong iStep2   = dH*iStride2;
    const Nd4jLong iStep3   = dW*iStride3;    
    const Nd4jLong kProd   = kH*kW;

    Nd4jLong hstart, wstart, hend, wend;
    T sum, *pIn;

    if(poolingMode == 0) {        // max 
#pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;
                        
                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;                        
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;
                        
                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));                            
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));                            

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = -MAX_FLOAT;
                                                                    
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                T val = pIn[kh + kw];
                                    if (val > sum)
                                        sum = val;
                                    }
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }    
    }
/*************************************************************************/    
    else if(poolingMode == 1) {      // avg
// #pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)        
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.f);
                                            
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += pIn[kh + kw];
                                
                        if ((int) extraParam0 == 0)         //Exclude padding
                            sum /= static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(hend-hstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep3)));   //Accounts for dilation

                        else if ((int) extraParam0 == 1)    //Include padding
                            sum /= kProd;
                
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }
    }    
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
#pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)    
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.f);
                                                                    
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                        sum = nd4j::math::nd4j_pow<T>(sum, static_cast<T>((T)1.f) / extraParam0);
                                                          
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling2d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling3d(NDArray<T>& input, NDArray<T>& output, const T* extraParams) {
    // input is  [bS, iC, iD, iH, iW]
    // output is [bS, iC, oD, oH, oW]
    T* out = output.getBuffer();
    T* in  = input.getBuffer();

    const Nd4jLong kD = (int)extraParams[0];
    const Nd4jLong kH = (int)extraParams[1];
    const Nd4jLong kW = (int)extraParams[2];
    const Nd4jLong sD = (int)extraParams[3];
    const Nd4jLong sH = (int)extraParams[4];
    const Nd4jLong sW = (int)extraParams[5];
    const Nd4jLong pD = (int)extraParams[6];
    const Nd4jLong pH = (int)extraParams[7];
    const Nd4jLong pW = (int)extraParams[8];
    const Nd4jLong dD = (int)extraParams[9]; 
    const Nd4jLong dH = (int)extraParams[10];
    const Nd4jLong dW = (int)extraParams[11];

    int poolingMode = (int)extraParams[12];
    T extraParam0 = extraParams[13];

    const Nd4jLong kDEff = kD + (kD-1)*(dD-1);
    const Nd4jLong kHEff = kH + (kH-1)*(dH-1);
    const Nd4jLong kWEff = kW + (kW-1)*(dW-1);

    const Nd4jLong bS = input.sizeAt(0);
    const Nd4jLong iC = input.sizeAt(1);
    const Nd4jLong iD = input.sizeAt(2);
    const Nd4jLong iH = input.sizeAt(3);
    const Nd4jLong iW = input.sizeAt(4);
    const Nd4jLong oD = output.sizeAt(2);
    const Nd4jLong oH = output.sizeAt(3);
    const Nd4jLong oW = output.sizeAt(4);
    const Nd4jLong iStride0 = input.stridesOf()[0];
    const Nd4jLong iStride1 = input.stridesOf()[1];
    const Nd4jLong iStride2 = input.stridesOf()[2];
    const Nd4jLong iStride3 = input.stridesOf()[3];
    const Nd4jLong iStride4 = input.stridesOf()[4];
    const Nd4jLong oStride0 = output.stridesOf()[0];
    const Nd4jLong oStride1 = output.stridesOf()[1];
    const Nd4jLong oStride2 = output.stridesOf()[2];
    const Nd4jLong oStride3 = output.stridesOf()[3];
    const Nd4jLong oStride4 = output.stridesOf()[4];
    const Nd4jLong iStep2   = dD*iStride2;
    const Nd4jLong iStep3   = dH*iStride3;
    const Nd4jLong iStep4   = dW*iStride4;
    const Nd4jLong kProd    = kD*kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    Nd4jLong dstart, hstart, wstart, dend, hend, wend;
    T sum, *pIn;

    if(poolingMode == 0) {        // max 
#pragma omp parallel for schedule(guided) private(pIn, sum, dstart, hstart, wstart, dend, hend, wend)
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));                            

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = -MAX_FLOAT;
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                        T val = pIn[kd + kh + kw];
                                            if (val > sum)
                                            sum = val;
                                    }
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg
#pragma omp parallel for schedule(guided) private(pIn, sum, dstart, hstart, wstart, dend, hend, wend)        
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += pIn[kd + kh + kw];
                                
                            if ((int) extraParam0 == 0)         //Exclude padding
                                sum /= static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(dend-dstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(hend-hstart) / static_cast<double>(iStep3))) * static_cast<double>(nd4j::math::nd4j_ceil<double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep4)));   //Accounts for dilation
                            else if ((int) extraParam0 == 1)    //Include padding
                                sum /= kProd;
                    
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
#pragma omp parallel for schedule(guided) private(pIn, sum, dstart, hstart, wstart, dend, hend, wend)    
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));                            
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);
                                
                            sum = nd4j::math::nd4j_pow<T>(sum, (T) 1.f / extraParam0);
                                                          
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling3d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling2dBP(NDArray<T>& input, NDArray<T>& gradO, NDArray<T>& gradI, const T* extraParams) {
    // input [bS, iC, iH, iW]
    // gradI [bS, iC, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oH, oW]    
    
    // TO DO: try to optimize initial zeroing using nested loops below
    gradI.assign(0.f);

    T* in = input.getBuffer();
    T* gI = gradI.getBuffer();
    T* gO = gradO.getBuffer();

    const Nd4jLong kH = (int)extraParams[0];
    const Nd4jLong kW = (int)extraParams[1];
    const Nd4jLong sH = (int)extraParams[2];
    const Nd4jLong sW = (int)extraParams[3];
    const Nd4jLong pH = (int)extraParams[4];
    const Nd4jLong pW = (int)extraParams[5];
    const Nd4jLong dH = (int)extraParams[6];
    const Nd4jLong dW = (int)extraParams[7];
    int poolingMode = (int)extraParams[8];
    T extraParam0 = extraParams[9];

    const Nd4jLong kHEff = kH + (kH-1)*(dH-1);
    const Nd4jLong kWEff = kW + (kW-1)*(dW-1);

    const Nd4jLong bS = gradI.sizeAt(0);
    const Nd4jLong iC = gradI.sizeAt(1);
    const Nd4jLong iH = gradI.sizeAt(2);
    const Nd4jLong iW = gradI.sizeAt(3);
    const Nd4jLong oH = gradO.sizeAt(2);
    const Nd4jLong oW = gradO.sizeAt(3);
    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];
    const Nd4jLong iStep2   = dH*iStride2;
    const Nd4jLong iStep3   = dW*iStride3;
    const Nd4jLong kProd    = kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;

    Nd4jLong hstart, wstart,hend, wend, maxKH, maxKW;
    T sum, valO, *pIn, *pgI;

    if(poolingMode == 0) {        // max 
#pragma omp parallel for schedule(guided) private(pIn, valO, sum, hstart, wstart, hend, wend, maxKH, maxKW)
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                    
                        pIn = in + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = -MAX_FLOAT;
                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];
                                                    
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                T valIn = pIn[kh + kw];
                                if (valIn > sum) {
                                    sum = valIn;
                                    maxKH = kh;
                                    maxKW = kw;
                                }
                            }
                        gI[pIn - in + maxKH + maxKW] += valO;
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg        
#pragma omp parallel for schedule(guided) private(pgI, valO, hstart, wstart, hend, wend)        
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pgI  = gI + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];
                                            
                        if ((int) extraParam0 == 0)         //Exclude padding                            
                            valO /= static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(hend-hstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep3)));   //Accounts for dilation
                        else if ((int) extraParam0 == 1)    //Include padding
                            valO /= kProd;

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                pgI[kh + kw] += valO;
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
#pragma omp parallel for schedule(guided) private(pIn, valO, pgI, sum, hstart, wstart, hend, wend)    
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in + b * iStride0 + c * iStride1;
                        pgI  = gI + (pIn - in);

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.f);
                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];
                                            
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                        valO *= nd4j::math::nd4j_pow<T>(sum, ((T)1. - extraParam0) / extraParam0);

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                pgI[kh + kw] += valO * nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0 - 1.f);
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling2dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling3dBP(NDArray<T>& input, NDArray<T>& gradO, NDArray<T>& gradI, const T* extraParams) {
    // input [bS, iC, iD, iH, iW]
    // gradI [bS, iC, iD, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oD, oH, oW]    

    // TO DO: try to optimize initial zeroing using nested loops below
    gradI.assign(0.f);
    
    T* in = input.getBuffer();
    T* gI = gradI.getBuffer();
    T* gO = gradO.getBuffer();

    const Nd4jLong kD = (int)extraParams[0];
    const Nd4jLong kH = (int)extraParams[1];
    const Nd4jLong kW = (int)extraParams[2];
    const Nd4jLong sD = (int)extraParams[3];
    const Nd4jLong sH = (int)extraParams[4];
    const Nd4jLong sW = (int)extraParams[5];
    const Nd4jLong pD = (int)extraParams[6];
    const Nd4jLong pH = (int)extraParams[7];
    const Nd4jLong pW = (int)extraParams[8];
    const Nd4jLong dD = (int)extraParams[9]; 
    const Nd4jLong dH = (int)extraParams[10];
    const Nd4jLong dW = (int)extraParams[11];

    Nd4jLong poolingMode = (int)extraParams[12];
    T extraParam0 = extraParams[13];

    const Nd4jLong kDEff = kD + (kD-1)*(dD-1);
    const Nd4jLong kHEff = kH + (kH-1)*(dH-1);
    const Nd4jLong kWEff = kW + (kW-1)*(dW-1);

    const Nd4jLong bS = gradI.sizeAt(0);
    const Nd4jLong iC = gradI.sizeAt(1);
    const Nd4jLong iD = gradI.sizeAt(2);
    const Nd4jLong iH = gradI.sizeAt(3);
    const Nd4jLong iW = gradI.sizeAt(4);
    const Nd4jLong oD = gradO.sizeAt(2);
    const Nd4jLong oH = gradO.sizeAt(3);
    const Nd4jLong oW = gradO.sizeAt(4);
    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong iStride4 = gradI.stridesOf()[4];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];
    const Nd4jLong oStride4 = gradO.stridesOf()[4];
    const Nd4jLong iStep2   = dD*iStride2;
    const Nd4jLong iStep3   = dH*iStride3;
    const Nd4jLong iStep4   = dW*iStride4;
    const Nd4jLong kProd    = kD*kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    Nd4jLong dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
    T sum, valO, *pIn, *pgI;

    if(poolingMode == 0) {        // max 
#pragma omp parallel for schedule(guided) private(pIn, valO, sum, dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW)
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                                                    
                            pIn = in + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));                            
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));                            

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = -MAX_FLOAT;
                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];
                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                        T valIn = pIn[kd + kh + kw];
                                        if (valIn > sum) {
                                            sum = valIn;
                                            maxKD = kd;
                                            maxKH = kh;
                                            maxKW = kw;
                                        }
                                    }
                            gI[pIn - in + maxKD + maxKH + maxKW] += valO;
                        }
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg        
#pragma omp parallel for schedule(guided) private(pgI, valO, dstart, hstart, wstart, dend, hend, wend)        
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pgI  = gI + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];
                                            
                            if ((int) extraParam0 == 0)         //Exclude padding
                                valO /= static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(dend-dstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double>(static_cast<double>(hend-hstart) / static_cast<double>(iStep3))) * static_cast<double>(nd4j::math::nd4j_ceil<double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep4)));   //Accounts for dilation
                            else if ((int) extraParam0 == 1)    //Include padding
                                valO /= kProd;

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        pgI[kd + kh + kw] += valO;
                        }
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
#pragma omp parallel for schedule(guided) private(pIn, pgI, valO, sum, dstart, hstart, wstart, dend, hend, wend)    
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in + b * iStride0 + c * iStride1;
                            pgI  = gI + (pIn - in);

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff; 
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-dstart) / static_cast<T>(dD));
                            if(hstart < 0)
                                hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                            if(wstart < 0)
                                wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                            if(dend > iD)
                                dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(dend-iD) / static_cast<T>(dD));
                            if(hend > iH)
                                hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                            if(wend > iW)
                                wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);

                            valO *= nd4j::math::nd4j_pow<T>(sum, ((T)1.f - extraParam0) / extraParam0);

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        pgI[kd + kh + kw] += valO * nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0 - 1.f);
                        }
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling3dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

template class ND4J_EXPORT ConvolutionUtils<float>;
template class ND4J_EXPORT ConvolutionUtils<float16>;
template class ND4J_EXPORT ConvolutionUtils<double>;
    
}
}
