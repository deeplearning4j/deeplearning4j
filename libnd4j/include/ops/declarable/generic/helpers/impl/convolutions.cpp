//
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/generic/helpers/convolutions.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_calcPadding2D(int& pH, int& pW, int oH, int oW, int inH, int inW, int kH, int kW, int sH, int sW, int dH, int dW) {
            int eKH, eKW;

            if (dH == 1 && dW == 1) {
                eKH = kH;
                eKW = kW;
            } else {
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pH = ((oH - 1) * sH + eKH - inH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pW = ((oW - 1) * sW + eKW - inW) / 2;
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
        void ConvolutionUtils<T>::col2vol2(NDArray<T>& col, NDArray<T>& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
            
            const T* colBuff = col.getBuffer();            
            T* volBuff       = vol.getBuffer();            

            auto colShapeOnly = shape::shapeOf(col.getShapeInfo());
            auto colStrides   = shape::stride(col.getShapeInfo());
            auto volShapeOnly = shape::shapeOf(vol.getShapeInfo());
            auto volOrder     = shape::order(vol.getShapeInfo());
            auto volStrides   = shape::stride(vol.getShapeInfo());

            int strideBS   = colStrides[0];
            int strideColC = colStrides[1];
            int strideKD   = colStrides[2];
            int strideKH   = colStrides[3];
            int strideKW   = colStrides[4];
            int strideColD = colStrides[5];
            int strideColH = colStrides[6];
            int strideColW = colStrides[7];

            int bS   = volShapeOnly[0];
            int volC = volShapeOnly[1];
            int volD = volShapeOnly[2];
            int volH = volShapeOnly[3];
            int volW = volShapeOnly[4];

            int kD   = colShapeOnly[2];
            int kH   = colShapeOnly[3];
            int kW   = colShapeOnly[4];            
            int colD = colShapeOnly[5];
            int colH = colShapeOnly[6];
            int colW = colShapeOnly[7];            

            //Effective kernel size, accounting for dilation
            int effKD = kD + (kD - 1) * (dD - 1);
            int effKH = kH + (kH - 1) * (dH - 1);
            int effKW = kW + (kW - 1) * (dW - 1);

            int n = bS * volC * volD * volH * volW;                        

#pragma omp parallel for schedule(guided) proc_bind(close)
            for (int i = 0; i < n; i++) {
                
                T val = 0;
                int w_vol = i % volW + pW;
                int h_vol = (i / volW) % volH + pH;
                int d_vol = (i / volW / volH) % volD + pD;
                int c_vol = i / volW / volH / volD;

                int num_vol   = c_vol / volC;
                int depth_vol = c_vol % volC;

                // compute the start and end of the output
                int w_col_start = (w_vol < effKW) ? 0 : (w_vol - effKW) / sW + 1;
                int w_col_end = nd4j::math::nd4j_min<int>(w_vol / sW + 1, colW);

                int h_col_start = (h_vol < effKH) ? 0 : (h_vol - effKH) / sH + 1;
                int h_col_end = nd4j::math::nd4j_min<int>(h_vol / sH + 1, colH);

                int d_col_start = (d_vol < effKD) ? 0 : (d_vol - effKD) / sD + 1;
                int d_col_end = nd4j::math::nd4j_min<int>(d_vol / sD + 1, colD);

                //Iterate over col entries in the 6d array... these are added up
                for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
                    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
                        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {                  

                            int d_k = (d_vol - d_col * sD);
                            int h_k = (h_vol - h_col * sH);
                            int w_k = (w_vol - w_col * sW);
                            
                            if(d_k % dD == 0 && h_k % dH == 0 && w_k % dW == 0) {
                                   
                                   d_k /= dD;
                                   h_k /= dH;
                                   w_k /= dW;
                                   val += colBuff[num_vol * strideBS + depth_vol * strideColC + d_k * strideKD + h_k * strideKH + w_k * strideKW + d_col * strideColD + h_col * strideColH + w_col * strideColW];
                             }
                        }
                    }
                }
                int i_f = 0;
                int i_c = i;
                for (int dim = 4; dim >= 0; --dim)
                {
                    i_f += (i_c % volShapeOnly[dim])  * volStrides[dim];
                    i_c = i_c / volShapeOnly[dim];
                }
                volBuff[i_f] += val;
            }

        }


//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_avgPool3D_bp(T *gradI_p, T *gradO_p, Nd4jLong iC, Nd4jLong iD, Nd4jLong iH, Nd4jLong iW, Nd4jLong oD, Nd4jLong oH, Nd4jLong oW, int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, bool count_include_pad) {
            for (int k = 0; k < iC; k++)
            {
                Nd4jLong i, j, ti;

                /* local pointers */
                T *ip = gradI_p + k * iD * iW * iH;
                T *op = gradO_p + k * oD * oW * oH;
                for (i = 0; i < iD*iW*iH; i++)
                    *(ip + i) = 0;

                /* loop over output */
                for (ti = 0; ti < oD; ti++)
                {
                    for (i = 0; i < oH; i++)
                    {
                        for (j = 0; j < oW; j++)
                        {
                            Nd4jLong cstart = ti * sD - pD;
                            Nd4jLong hstart = i  * sH - pH;
                            Nd4jLong wstart = j  * sW - pW;
                            Nd4jLong cend = nd4j::math::nd4j_min<Nd4jLong>(cstart + kD, iD + pD);
                            Nd4jLong hend = nd4j::math::nd4j_min<Nd4jLong>(hstart + kH, iH + pH);
                            Nd4jLong wend = nd4j::math::nd4j_min<Nd4jLong>(wstart + kW, iW + pW);
                            Nd4jLong pool_size = (cend -cstart) * (hend - hstart) * (wend - wstart);
                            cstart = nd4j::math::nd4j_max<Nd4jLong>(cstart, 0);
                            hstart = nd4j::math::nd4j_max<Nd4jLong>(hstart, 0);
                            wstart = nd4j::math::nd4j_max<Nd4jLong>(wstart, 0);
                            cend = nd4j::math::nd4j_min<Nd4jLong>(cend, iD);
                            hend = nd4j::math::nd4j_min<Nd4jLong>(hend, iH);
                            wend = nd4j::math::nd4j_min<Nd4jLong>(wend, iW);

                            Nd4jLong divide_factor;
                            if (count_include_pad)
                                divide_factor = pool_size;
                            else
                                divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                            /* scatter gradients out to footprint: */
                            T val  = *op++;

                            long x,y,z;
                            for (z = cstart; z < cend; z++)
                            {
                                for (y = hstart; y < hend; y++)
                                {
                                    for (x = wstart; x < wend; x++)
                                    {
                                        *(ip + z * iH * iW + y * iW + x) += val / divide_factor;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_avgPool3D(T *input_p, T *output_p, Nd4jLong iC, Nd4jLong iD, Nd4jLong iH, Nd4jLong iW, Nd4jLong oD, Nd4jLong oH, Nd4jLong oW, int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, bool count_include_pad) {
            for (Nd4jLong k = 0; k < iC; k++)
            {
                long i, j, ti;

                /* local pointers. */
                T *ip = input_p + k * iD * iW * iH;
                T *op = output_p + k * oD * oW * oH;
                for (i = 0; i < oD * oH * oW; ++i)
                    *(op + i) = 0;

                /* loop over output */
                for (ti = 0; ti < oD; ti++)
                {
                    for (i = 0; i < oH; i++)
                    {
                        for (j = 0; j < oW; j++)
                        {
                            /* compute pool range. */
                            Nd4jLong cstart = ti * sD - pD;
                            Nd4jLong hstart = i  * sH - pH;
                            Nd4jLong wstart = j  * sW - pW;
                            Nd4jLong cend = nd4j::math::nd4j_min<Nd4jLong>(cstart + kD, iD + pD);
                            Nd4jLong hend = nd4j::math::nd4j_min<Nd4jLong>(hstart + kH, iH + pH);
                            Nd4jLong wend = nd4j::math::nd4j_min<Nd4jLong>(wstart + kW, iW + pW);
                            Nd4jLong pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);
                            cstart = nd4j::math::nd4j_max<Nd4jLong>(cstart, 0);
                            hstart = nd4j::math::nd4j_max<Nd4jLong>(hstart, 0);
                            wstart = nd4j::math::nd4j_max<Nd4jLong>(wstart, 0);
                            cend = nd4j::math::nd4j_min<Nd4jLong>(cend, iD);
                            hend = nd4j::math::nd4j_min<Nd4jLong>(hend, iH);
                            wend = nd4j::math::nd4j_min<Nd4jLong>(wend, iW);

                            Nd4jLong divide_factor;
                            if (count_include_pad)
                                divide_factor = pool_size;
                            else
                                divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                            /* compute local sum: */
                            T sum = (T) 0.0f;
                            long x, y, z;

                            for (z = cstart; z < cend; z++)
                            {
                                for (y = hstart; y < hend; y++)
                                {
                                    for (x = wstart; x < wend; x++)
                                    {
                                        sum +=  *(ip + z * iW * iH + y * iW + x);
                                    }
                                }
                            }

                            /* set output to local max */
                            *op++ += sum / divide_factor;
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
                    T* ptr_input = input_data + i*istride0;

                    /* do image, kernel convolution */
                    ConvolutionUtils<T>::conv3D(output_data,
                           alpha,
                           ptr_input,  nInputDepth, nInputRows,  nInputCols,
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
                                     T* ptr_input, Nd4jLong nInputDepth, Nd4jLong nInputRows, Nd4jLong nInputCols,
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
                                                 ptr_input, nInputDepth, nInputRows,  nInputCols,
                                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                 sdepth, srow, scol);
                } else {
                    ConvolutionUtils<T>::fullConv3Dptr(output_data,
                                                alpha,
                                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                sdepth, srow, scol);
                }
            else
            if (*xc == 'X') {
                ConvolutionUtils<T>::validXCorr3Dptr(output_data,
                                              alpha,
                                              ptr_input, nInputDepth, nInputRows,  nInputCols,
                                              ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                              sdepth, srow, scol);
            } else {
                ConvolutionUtils<T>::validConv3Dptr(output_data,
                                             alpha,
                                             ptr_input, nInputDepth, nInputRows,  nInputCols,
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
void ConvolutionUtils<T>::maxPool3dFrame(NDArray<T>& input, NDArray<T>& output, const int iStride, const int indStride, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
        
    T* inBuff  = input.getBuffer()  + iStride;
    T* outBuff = output.getBuffer() + indStride;

    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);

    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);    

    int k;

#pragma omp parallel for private(k)
    for (k = 0; k < iC; k++) {
    
        /* loop over output */
        int i, j, ti;
        for (ti = 0; ti < oD; ti++) {
            for (i = 0; i < oH; i++) {
                for (j = 0; j < oW; j++){
          
                    /* local pointers */
                    int start_d = ti * sD - pD;
                    int start_w = j  * sW - pW;
                    int start_h = i  * sH - pH;
                    
                    int kernel_d = math::nd4j_min<int>(kD, kD + start_d);
                    int kernel_h = math::nd4j_min<int>(kH, kH + start_h);
                    int kernel_w = math::nd4j_min<int>(kW, kW + start_w);

                    while(start_d < 0)
                        start_d += dD;
                    while(start_h < 0)
                        start_h += dH;
                    while(start_w < 0)
                        start_w += dW;

                    T* ip = inBuff + k * iD * iW * iH  + start_d * iW * iH + start_h * iW + start_w;
                    T* op = outBuff + k * oD * oW * oH + ti * oW * oH + i * oW + j;          

                    // compute local max
                    T maxval = - DataTypeUtils::max<T>();
                    int x,y,z;                    

                    for (z = 0; z < kernel_d; z++) {
                        for (y = 0; y < kernel_h; y++) {
                            for (x = 0; x < kernel_w; x++) {

                                if ((start_d + z * dD < iD) && (start_h + y * dH < iH) && (start_w + x * dW < iW)) {
                                    
                                    T val = *(ip + z * dD * iW * iH + y * dH * iW + x * dW);
                                    if (val > maxval)                  
                                        maxval = val;                                     
                                }
                            }
                        }
                    }
                    // set output to local max
                    *op = maxval;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::maxPool3dIndicesFrame(NDArray<T>& input, int* indices, const int iStride, const int indStride, const int oD, const int oH, const int oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
        
    T* inBuff    = input.getBuffer()   + iStride;    
    int* indBuff = indices + indStride;

    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);

    int k;

#pragma omp parallel for private(k)
    for (k = 0; k < iC; k++) {
    
        /* loop over output */
        int i, j, ti;
        for (ti = 0; ti < oD; ti++) {
            for (i = 0; i < oH; i++) {
                for (j = 0; j < oW; j++){
          
                    /* local pointers */
                    int start_d = ti * sD - pD;
                    int start_h = i  * sH - pH;
                    int start_w = j  * sW - pW;                    
                    
                    int kernel_d = math::nd4j_min<int>(kD, kD + start_d);
                    int kernel_h = math::nd4j_min<int>(kH, kH + start_h);
                    int kernel_w = math::nd4j_min<int>(kW, kW + start_w);

                    while(start_d < 0)
                        start_d += dD;
                    while(start_h < 0)
                        start_h += dH;
                    while(start_w < 0)
                        start_w += dW;                    

                    T* ip   = inBuff  + k * iD * iH * iW + start_d * iH * iW + start_h * iW + start_w;                    
                    int* indP = indBuff + k * oD * oH * oW + ti * oH * oW + i * oW + j;
                    
                    T maxval = - DataTypeUtils::max<T>();
                    int x,y,z;                    
                    int mx, my, mz;
                    mx = my = mz = -1;

                    for (z = 0; z < kernel_d; z++) {
                        for (y = 0; y < kernel_h; y++) {
                            for (x = 0; x < kernel_w; x++) {

                                if ((start_d + z * dD < iD) && (start_h + y * dH < iH) && (start_w + x * dW < iW)) {   

                                    T val = *(ip + z * dD * iH * iW + y * dH * iW + x * dW);
                                    
                                    if (val > maxval) {                 
                                        
                                        maxval = val;                                  
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
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::maxPool3dFrameBp(NDArray<T>& input, const int* indices, NDArray<T>& output, const int iStride, const int oStride, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    T* inBuff  = input.getBuffer()  + iStride;
    T* outBuff = output.getBuffer() + oStride;    
    const int* indBuff = indices + iStride;

    int iC = input.sizeAt(1);
    int iD = input.sizeAt(2);
    int iH = input.sizeAt(3);
    int iW = input.sizeAt(4);

    int oD = output.sizeAt(2);
    int oH = output.sizeAt(3);
    int oW = output.sizeAt(4);    

    int k;
#pragma omp parallel for private(k)
    for (k = 0; k < iC; k++) {
        
        T* oP = outBuff + k * oD * oW * oH;
        T* iP = inBuff  + k * iD * iW * iH;    
        const int* indP = indBuff + k * iD * iH * iW;

        int ti, i, j;
        for (ti = 0; ti < iD; ti++) {
            for (i = 0; i < iH; i++)  {
                for (j = 0; j < iW; j++) {
                                        
                    const int* indzP = &indP[ti * iH * iW + i * iW + j];

                    int maxti = ((unsigned char*)(indzP))[0] * dD + ti * sD - pD;
                    int maxi  = ((unsigned char*)(indzP))[1] * dH + i  * sH - pH;
                    int maxj  = ((unsigned char*)(indzP))[2] * dW + j  * sW - pW;
                    
                    if (maxti != -1) 
                        oP[maxti * oH * oW + maxi * oW + maxj] += iP[ti * iH * iW + i * iW + j];      
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////

template<typename T>
void ConvolutionUtils<T>::vol2col2(NDArray<T>& vol, NDArray<T>& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    T* colBuff = col.getBuffer();    

    auto colShape  = shape::shapeOf(col.getShapeInfo());
    auto colOrder  = shape::order(col.getShapeInfo());
    auto colStride = shape::stride(col.getShapeInfo());

    auto volShape  = shape::shapeOf(vol.getShapeInfo());
    auto volStride = shape::stride(vol.getShapeInfo());

    int bS   = volShape[0];
    int volC = volShape[1];
    int volD = volShape[2];
    int volH = volShape[3];
    int volW = volShape[4];

    int strideBS   = volStride[0];
    int strideVolC = volStride[1];
    int strideVolD = volStride[2];
    int strideVolH = volStride[3];
    int strideVolW = volStride[4];

    int kD   = colShape[2];
    int kH   = colShape[3];
    int kW   = colShape[4];            
    int colD = colShape[5];
    int colH = colShape[6];
    int colW = colShape[7];

    int kSize = kD * kW * kH;

    int n = bS * volC * colD * colH * colW;

#pragma omp parallel for schedule(guided) proc_bind(close)
    for (int index = 0; index < n; index++) {
                
        int w_col = index % colW;
        int h_col = (index / colW) % colH;
        int d_col = (index / colW / colH) % colD;
    
        int c_vol = index / colW / colH / colD;
        int c_col = c_vol * kSize;
    
        int depth_vol = c_vol % volC;
        int num_vol   = c_vol / volC;
        int d_offset = d_col * sD - pD;
        int h_offset = h_col * sH - pH;
        int w_offset = w_col * sW - pW;

        T* data_col_ptr = col.getBuffer();
        T* data_vol_ptr = vol.getBuffer();

        int i_c = ((c_col * colD + d_col) * colH + h_col) * colW + w_col;
        data_col_ptr += ((c_col * colD + d_col) * colH + h_col) * colW + w_col;
        data_vol_ptr += num_vol * strideBS + depth_vol * strideVolC + d_offset * strideVolD + h_offset * strideVolH + w_offset * strideVolW;

        for (int z = 0; z < kD; ++z) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                            
                    int d_vol = d_offset + z * dD;
                    int h_vol = h_offset + i * dH;
                    int w_vol = w_offset + j * dW;
                            
                    int i_f = 0;
                    int i_c_temp = i_c;
                            
                    for (int dim = 7; dim >= 0; dim--) {
                        i_f += (i_c_temp % colShape[dim])  * colStride[dim];
                        i_c_temp = i_c_temp / colShape[dim];
                    }
                                
                    if (d_vol >= 0 && h_vol >= 0 && w_vol >= 0 && d_vol < volD && h_vol < volH && w_vol < volW)
                        colBuff[i_f] = data_vol_ptr[z * dD * strideVolD + i * dH * strideVolH + j * dW * strideVolW];
                    else 
                        colBuff[i_f] = 0;

                     data_col_ptr += colD * colH * colW;
                     i_c          += colD * colH * colW;
                }
            }
        }
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
void ConvolutionUtils<T>::getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indOoH = 1; indWoC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indOoH = 2; indWoC = 0; indWiC = 1;              
    }    

    bS = inShapeInfo[1];                          // batch size
    iC = inShapeInfo[indIOioC+1];                   // input channels        
    iH = inShapeInfo[indIiH+1];                     // input height
    iW = inShapeInfo[indIiH+2];                   // input width
    oC = outShapeInfo[indIOioC+1];                  // output channels
    oH = outShapeInfo[indOoH+1];                    // output height
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
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
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
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        

    //----- calculation of output -----//
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                    // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    NDArrayFactory<T>::tensorDot(&columns, weights, output, {1,2,3}, weightsAxesForDot, permutForOutput); // [bS, iC, kH, kW, oH, oW] x [kH, kW, iC, oC]/[oC, iC, kH, kW] = [bS, oH, oW, oC]

    //----- add biases if required -----//
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::conv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                        // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                        // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = inArrs[2];                        // [oC]
    NDArray<T> *gradO   = inArrs[3];                        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    NDArray<T> *gradW = outArrs[1];                         // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
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

    std::vector<int> gradOaxesForDot, permutForGradW, permutForColumns;    

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW        
        permutForGradW   = {2, 0, 1, 3};                                        // [kH, kW, iC, oC] -> [iC, kH, kW, oC]        
        permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [kH, kW, iC, bS, oH, oW]
    }
    else {
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
        permutForGradW   = {1, 2, 3, 0};                                        // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
        permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [iC, kH, kW, bS, oH, oW]
    }
    
    if(isSameMode)                       // SAME        
        _calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    // ----- calculation of gradW and gradB ----- // 
    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]        
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, permutForGradW);       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]

    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, permutForColumns);       // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]/[iC, kH, kW, bS, oH, oW]
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
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
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
    std::vector<std::vector<Nd4jLong>> modifWeights, modifOutput;
    std::vector<Nd4jLong> outReShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};                                       // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());                
    NDArray<T>* outputReshaped = output->reshape(output->ordering(), outReShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});

    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                                 // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, weights, outputReshaped, modifColumns, modifWeights, modifOutput);    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    
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
    NDArray<T> *weights = inArrs[1];                            // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = inArrs[2];                            // [oC] = [iC*mC]
    NDArray<T> *gradO   = inArrs[3];                            // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                             // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = outArrs[1];                             // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
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
    std::vector<std::vector<Nd4jLong>> modifGradW, modifGradO1, modifGradO2;
    std::vector<Nd4jLong> gradOreShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{2,0,1,3},{iC,kH*kW,mC}};                                         // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{1,2,3,0},{iC,kH*kW,mC}};                                         // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        
    NDArray<T>* gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    
    // ----- calculation of gradW and gradB ----- //            
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradOreshaped, gradW, modifColumns, modifGradO1, modifGradW);  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//                
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, modifGradW, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]    
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
    NDArray<T> *weightsDepth = inArrs[1];                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = inArrs[2];                                           // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
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

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const int colStride0 = columns.stridesOf()[0];
    const int colStride1 = columns.stridesOf()[1];
    const int colStride2 = columns.stridesOf()[2];
    const int colStride3 = columns.stridesOf()[3];
    const int colStride4 = columns.stridesOf()[4];
    const int colStride5 = columns.stridesOf()[5];
    const int colStride6 = columns.stridesOf()[6];
    const int colStride7 = columns.stridesOf()[7];  
    const int volStride0 = volume.stridesOf()[0];
    const int volStride1 = volume.stridesOf()[1];
    const int volStride2 = volume.stridesOf()[2];
    const int volStride3 = volume.stridesOf()[3];
    const int volStride4 = volume.stridesOf()[4];    
    
    T* vol = volume.getBuffer();
    T* col = columns.getBuffer();

    const T* vol0End = vol + volStride1 * iC;
    const int kDepEnd = -pD + kD * dD;
    const int kRowEnd = -pH + kH * dH;
    const int kColEnd = -pW + kW * dW;
    const int oHW = oH * oW;
    const int volDepEnd = oD * sD;
    const int volRowEnd = oH * sH;
    const int volColEnd = oW * sW;

    T *vol1, *vol2, *col0;
    int volDepStart, volRowStart, volColStart, volDep, volRow, volCol;

    if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) {

#pragma omp parallel for if(bS > Environment::getInstance()->elementwiseThreshold()) schedule(static) proc_bind(close) private(vol1, vol2, col0, volDepStart, volRowStart, volColStart, volDep, volRow, volCol)
        for (int b = 0; b < bS; b++) {            
            col0 = col + (b * colStride0);                        
            T *vol0 = vol + (b * volStride0);

            for (int channel = 0; channel < iC; ++channel, vol0 += volStride1) { 

                for (int kDep = 0; kDep < kD; ++kDep) { 
                    volDepStart = -pD + kDep * dD;

                    for (int kRow = 0; kRow < kH; ++kRow) {
                        volRowStart = -pH + kRow * dH;

                        for (int kCol = 0; kCol < kW; ++kCol) {
                            volDep = volDepStart;
                            volColStart = -pW + kCol * dW;
                            
                            for (int colDep = 0; colDep < oD; ++colDep, volDep += sD) {

                                if(static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD)) {                                
                                    for (int colHW = 0; colHW < oHW; ++colHW, ++col0)
                                            *col0 = 0.;
                                }
                                else {
                                    volRow = volRowStart;
                                    vol1 = vol0 + volDep * volStride2;

                                    for (int colRow = 0; colRow < oH; ++colRow, volRow+=sH) {

                                        if (static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH)) {                                        
                                            for (int colW = 0; colW < oW; ++colW, ++col0) 
                                                *col0 = 0.;
                                        }
                                        else {
                                            volCol = volColStart;
                                            vol2 = vol1 + volRow * volStride3;

                                            for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW, ++col0) 
                                                if (static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))                                                
                                                    *col0 = 0.;                  
                                                else 
                                                    *col0 = *(vol2 + volCol * volStride4);                
                                        }        
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        const int col5End = oH * colStride6;
        const int col6End = oW * colStride7;
        T *col1, *col2, *col3, *col4, *col5, *col6;
#pragma omp parallel for if(bS > Environment::getInstance()->elementwiseThreshold()) schedule(static) proc_bind(close) private(vol1, vol2, col0, col1, col2, col3, col4, col5, col6, volDepStart, volRowStart, volColStart, volDep, volRow, volCol)
          for (int b = 0; b < bS; b++) {            
            col0 = col + (b * colStride0);     
            T *vol0 = vol + (b * volStride0);                   
            
            for (int channel = 0; channel < iC; ++channel, vol0+=volStride1, col0+=colStride1) {            
                col1 = col0;

                for (int kDep = 0; kDep < kD; ++kDep, col1+=colStride2) { 
                    col2 = col1;
                    volDepStart = -pD + kDep * dD;                   
                    
                    for (int kRow = 0; kRow < kH; ++kRow, col2+=colStride3) {
                        col3 = col2;
                        volRowStart = -pH + kRow * dH;
                    
                        for (int kCol = 0; kCol < kW; ++kCol, col3+=colStride4) {
                            col4 = col3;
                            volDep = volDepStart;
                            volColStart = -pW + kCol * dW;
                            
                            for (int colDep = 0; colDep < oD; ++colDep, volDep+=sD, col4+=colStride5) {                            
                                col5 = col4;

                                if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD)) {
                                    for (int colH = 0; colH < oH; ++colH, col5+=colStride6) {
                                        col6 = col5;                                        
                                        for (int colW = 0; colW < oW; ++colW, col6+=colStride7)
                                            *col6 = 0.;
                                    }                                    
                                }
                                else {
                                    volRow = volRowStart;                                    
                                    vol1 = vol0 + volDep * volStride2;

                                    for (int colRow = 0; colRow < oH; ++colRow, volRow+=sH, col5+=colStride6) {                                    
                                        col6 = col5;                                        

                                        if (static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH)) {
                                            for (int colW = 0; colW < oW; ++colW, col6+=colStride7)
                                                *col6 = 0.;
                                        }
                                        else {
                                            volCol = volColStart;                            
                                            vol2 = vol1 + volRow * volStride3;

                                            for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW, col6+=colStride7) 
                                                if (static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))                                                
                                                    *col6 = 0.;                  
                                                else 
                                                    *col6 = *(vol2 + volCol * volStride4);
                                        }
                                    }
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

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const int colStride0 = columns.stridesOf()[0];
    const int colStride1 = columns.stridesOf()[1];
    const int colStride2 = columns.stridesOf()[2];
    const int colStride3 = columns.stridesOf()[3];
    const int colStride4 = columns.stridesOf()[4];
    const int colStride5 = columns.stridesOf()[5];
    const int colStride6 = columns.stridesOf()[6];
    const int colStride7 = columns.stridesOf()[7];  
    const int volStride0 = volume.stridesOf()[0];
    const int volStride1 = volume.stridesOf()[1];
    const int volStride2 = volume.stridesOf()[2];
    const int volStride3 = volume.stridesOf()[3];
    const int volStride4 = volume.stridesOf()[4];    
    
    T* vol = volume.getBuffer();
    T* col = columns.getBuffer();

    const T* vol0End = vol + volStride1 * iC;
    const int kDepEnd = -pD + kD * dD;
    const int kRowEnd = -pH + kH * dH;
    const int kColEnd = -pW + kW * dW;
    const int colStepOH = oH * colStride6;
    const int colStepOW = oW * colStride7;
    const int volDepEnd = oD * sD;
    const int volRowEnd = oH * sH;
    const int volColEnd = oW * sW;

    T *vol1, *vol2, *vol3, *col0;
    int volDepStart, volRowStart, volColStart, volDep, volRow, volCol;

    if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) {

#pragma omp parallel for if(bS > Environment::getInstance()->elementwiseThreshold()) schedule(static) proc_bind(close) private(vol1, vol2, vol3, col0, volDepStart, volRowStart, volColStart, volDep, volRow, volCol)
        for (int b = 0; b < bS; b++) {            
            col0 = col + (b * colStride0);                        
            T *vol0 = vol + (b * volStride0);

            for (int channel = 0; channel < iC; ++channel, vol0 += volStride1) { 

                for (int kDep = 0; kDep < kD; ++kDep) { 
                    volDepStart = -pD + kDep * dD;

                    for (int kRow = 0; kRow < kH; ++kRow) {
                        volRowStart = -pH + kRow * dH;

                        for (int kCol = 0; kCol < kW; ++kCol) {
                            volDep = volDepStart;
                            volColStart = -pW + kCol * dW;
                                   
                            for (int colDep = 0; colDep < oD; ++colDep, volDep += sD) {                            

                                if(static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD)) {                            
                                    col0 += colStepOH;
                                }
                                else {
                                    volRow = volRowStart;
                                    vol1 = vol0 + volDep * volStride2;

                                    for (int colRow = 0; colRow < oH; ++colRow, volRow+=sH) {                                    

                                        if (static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH)) {                                        
                                            col0 += colStepOW;           
                                        }
                                        else {                                            
                                            volCol = volColStart;
                                            vol2 = vol1 + volRow * volStride3;

                                            if(kDep == -pD &&  kRow == -pH && kCol == -pW) {        // first pass, nullify all columns elemets
                                                for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW,  col0+=colStride7)
                                                    if (static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) 
                                                        *(vol2 + volCol * volStride4) = *col0;                                                        
                                            }
                                            else {
                                                for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW,  col0+=colStride7) {
                                                    if (static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) {
                                                        vol3 = vol2 + volCol * volStride4;
                                                        *vol3 += *col0;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        const int col5End = oH * colStride6;
        const int col6End = oW * colStride7;
        T *vol0, *col1, *col2, *col3, *col4, *col5, *col6;
#pragma omp parallel for if(bS > Environment::getInstance()->elementwiseThreshold()) schedule(static) proc_bind(close) private(vol1, vol2, col0, col1, col2, col3, col4, col5, col6, volDepStart, volRowStart, volColStart, volDep, volRow, volCol)
          for (int b = 0; b < bS; b++) {            
            col0 = col + (b * colStride0);     
            T *vol0 = vol + (b * volStride0);                   
            
            for (int channel = 0; channel < iC; ++channel, vol0+=volStride1, col0+=colStride1) {            
                col1 = col0;

                    for (int kDep = 0; kDep < kD; ++kDep, col1+=colStride2) { 
                    col2 = col1;
                    volDepStart = -pD + kDep * dD;                   
                    
                    for (int kRow = 0; kRow < kH; ++kRow, col2+=colStride3) {
                        col3 = col2;
                        volRowStart = -pH + kRow * dH;
                    
                        for (int kCol = 0; kCol < kW; ++kCol, col3+=colStride4) {
                            col4 = col3;
                            volDep = volDepStart;
                            volColStart = -pW + kCol * dW;                    

                            for (int colDep = 0; colDep < oD; ++colDep, volDep+=sD, col4+=colStride5) {                            
                                col5 = col4;

                                if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD)) {
                                    col5 += colStepOH;
                                }
                                else {
                                    volRow = volRowStart;                                    
                                    vol1 = vol0 + volDep * volStride2;

                                    for (int colRow = 0; colRow < oH; ++colRow, volRow+=sH, col5+=colStride6) {                                    
                                        col6 = col5;         

                                        if (static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH)) {
                                            col6 += colStepOW;
                                        }
                                        else {                                            
                                            volCol = volColStart;    
                                            vol2 = vol1 + volRow * volStride3;

                                            if(kDep == -pD &&  kRow == -pH && kCol == -pW) {        // first pass, nullify all columns elemets
                                                for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW, col6+=colStride7)
                                                    if (static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) 
                                                        *(vol2 + volCol * volStride4) = *col6;
                                            }
                                            else {
                                                for (int colCol = 0; colCol < oW; ++colCol, volCol+=sW, col6+=colStride7) {
                                                    if (static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) {
                                                        vol3 = vol2 + volCol * volStride4;
                                                        *vol3 += *col6;   
                                                    }
                                                }
                                            }
                                        }        
                                    }
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
    
    int indIn[8]  = {0,0,  0,0,  0,0,  0,0};
    int indOut[8] = {0,0,  0,0,  0,0,  0,0};
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
                    output(indOut).assign(input(indIn));
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
    int indIn[10]  = {0,0,  0,0,  0,0,  0,0,  0,0};
    int indOut[10] = {0,0,  0,0,  0,0,  0,0,  0,0};
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
                            output(indOut).assign(input(indIn));
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
    int indIn[8]  = {0,0,  0,0,  0,0,  0,0};
    int indOut[8] = {0,0,  0,0,  0,0,  0,0};
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
                    if(!fh && !fw)
                        subGradI.assign(gradO(indOut));
                    else
                        subGradI += gradO(indOut);
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
    int indIn[10]  = {0,0,  0,0,  0,0,  0,0,  0,0};
    int indOut[10] = {0,0,  0,0,  0,0,  0,0,  0,0};
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
                            if(!fd && !fh && !fw)
                                subGradI.assign(gradO(indOut));
                            else
                                subGradI += gradO(indOut);
                        }
                    }
                }
            }
        }
    }    
}
template class ND4J_EXPORT ConvolutionUtils<float>;
template class ND4J_EXPORT ConvolutionUtils<float16>;
template class ND4J_EXPORT ConvolutionUtils<double>;
    
}
}

