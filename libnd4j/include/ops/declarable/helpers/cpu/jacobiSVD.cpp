//
// Created by Yurii Shyrma on 11.01.2018
//

#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/hhColPivQR.h>


namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
JacobiSVD<T>::JacobiSVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw std::runtime_error("ops::helpers::JacobiSVD constructor: input array must be 2D matrix !");

    _rows = static_cast<int>(matrix.sizeAt(0));
    _cols = static_cast<int>(matrix.sizeAt(1));
    _diagSize = math::nd4j_min<int>(_rows, _cols);    

    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;

    _s = NDArray<T>(matrix.ordering(), {_diagSize, 1}, matrix.getWorkspace());

    if(_calcU) {
        if(_fullUV)
            _u = NDArray<T>(matrix.ordering(), {_rows, _rows}, matrix.getWorkspace());   
        else
            _u = NDArray<T>(matrix.ordering(), {_rows, _diagSize}, matrix.getWorkspace());   
    }
    else 
        _u = NDArray<T>(matrix.ordering(), {_rows, 1}, matrix.getWorkspace());   

    if(_calcV) {
        if(_fullUV)
            _v = NDArray<T>(matrix.ordering(), {_cols, _cols}, matrix.getWorkspace());   
        else
            _v = NDArray<T>(matrix.ordering(), {_cols, _diagSize}, matrix.getWorkspace());   
    }
    else 
        _v = NDArray<T>(matrix.ordering(), {_cols, 1}, matrix.getWorkspace());   
    
    _m = NDArray<T>(matrix.ordering(), {_diagSize, _diagSize}, matrix.getWorkspace());
    
    evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnLeft(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: second arguments is out of array row range !");
        
        IndicesList indices({NDIndex::interval(i, j+1, j-i), NDIndex::all()});
        NDArray<T>* pTemp = block.subarray(indices);
        NDArray<T> temp = *pTemp;
        pTemp->assign(mmul(rotation, temp));
        delete pTemp;
    }
    else {

        if(j+1 > block.sizeAt(0) || i+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: some or both integer arguments are out of array row range !");
        
        NDArray<T> temp(block.ordering(), {2, block.sizeAt(1)}, block.getWorkspace());
        NDArray<T>* row1 = block.subarray({{i, i+1}, {}});
        NDArray<T>* row2 = block.subarray({{j, j+1}, {}});
        NDArray<T>* rowTemp1 = temp.subarray({{0, 1}, {}});
        NDArray<T>* rowTemp2 = temp.subarray({{1, 2}, {}});
        rowTemp1->assign(row1);
        rowTemp2->assign(row2);
        temp.assign(mmul(rotation, temp));
        row1->assign(rowTemp1);
        row2->assign(rowTemp2);
        
        delete row1;
        delete row2;
        delete rowTemp1;
        delete rowTemp2;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnRight(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: second argument is out of array column range !");
        
        IndicesList indices({NDIndex::all(), NDIndex::interval(i, j+1, j-i)});
        NDArray<T>* pTemp = block.subarray(indices);
        NDArray<T> temp = *pTemp;
        pTemp->assign(mmul(temp, rotation));
        delete pTemp;
    }
    else {

        if(j+1 > block.sizeAt(1) || i+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: some or both integer arguments are out of array column range !");
        
        NDArray<T> temp(block.ordering(), {block.sizeAt(0), 2}, block.getWorkspace());
        NDArray<T>* col1 = block.subarray({{}, {i, i+1}});
        NDArray<T>* col2 = block.subarray({{}, {j, j+1}});
        NDArray<T>* colTemp1 = temp.subarray({{}, {0, 1}});
        NDArray<T>* colTemp2 = temp.subarray({{}, {1, 2}});
        colTemp1->assign(col1);
        colTemp2->assign(col2);
        temp.assign(mmul(temp, rotation));
        col1->assign(colTemp1);
        col2->assign(colTemp2);
        
        delete col1;
        delete col2;
        delete colTemp1;
        delete colTemp2;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::isBlock2x2NotDiag(NDArray<T>& block, int p, int q, T& maxElem) {
        
    NDArray<T> rotation(_m.ordering(), {2, 2}, _m.getWorkspace());    
    T n = math::nd4j_sqrt<T>(block(p,p)*block(p,p) + block(q,p)*block(q,p));

    const T almostZero = DataTypeUtils::min<T>();
    const T precision = DataTypeUtils::eps<T>();

    if(n == (T)0.)      
        block(p,p) = block(q,p) = 0.;
    else {

        rotation(0,0) =  rotation(1,1) = block(p,p) / n;
        rotation(0,1) = block(q,p) / n;
        rotation(1,0) = -rotation(0,1);
        
        mulRotationOnLeft(p, q, block, rotation);        

        if(_calcU) {
            NDArray<T>* temp2 = rotation.transpose();
            mulRotationOnRight(p, q, _u, *temp2);
            delete temp2;
        }
    }
    
    maxElem = math::nd4j_max<T>(maxElem, math::nd4j_max<T>(math::nd4j_abs<T>(block(p,p)), math::nd4j_abs<T>(block(q,q))));
    T threshold = math::nd4j_max<T>(almostZero, precision * maxElem);
    const bool condition1 = math::nd4j_abs<T>(block(p,q)) > threshold;
    const bool condition2 = math::nd4j_abs<T>(block(q,p)) > threshold;

    return condition1 || condition2;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::createJacobiRotation(const T& x, const T& y, const T& z, NDArray<T>& rotation) {
  
    T denom = 2.* math::nd4j_abs<T>(y);

    if(denom < DataTypeUtils::min<T>()) {
        
        rotation(0,0) = rotation(1,1) = 1.;
        rotation(0,1) = rotation(1,0) = 0.;
        return false;
    } 
    else {
        
        T tau = (x-z)/denom;
        T w = math::nd4j_sqrt<T>(tau*tau + 1.);
        T t;
  
        if(tau > (T)0.)
            t = 1. / (tau + w);
        else
            t = 1. / (tau - w);
  
        T sign = t > (T)0. ? 1. : -1.;
        T n = 1. / math::nd4j_sqrt<T>(t*t + 1.);
        rotation(0,0) = rotation(1,1) = n;
        rotation(0,1) = -sign * (y / math::nd4j_abs<T>(y)) * math::nd4j_abs<T>(t) * n;
        rotation(1,0) = -rotation(0,1);

        return true;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::svd2x2(const NDArray<T>& block, int p, int q, NDArray<T>& left, NDArray<T>& right) {
        
    NDArray<T> m(block.ordering(), {2, 2}, block.getWorkspace());
    m(0,0) = block(p,p);
    m(0,1) = block(p,q);
    m(1,0) = block(q,p);
    m(1,1) = block(q,q);
  
    NDArray<T> rotation(block.ordering(), {2, 2}, block.getWorkspace());
    T t = m(0,0) + m(1,1);
    T d = m(1,0) - m(0,1);

    if(math::nd4j_abs<T>(d) < DataTypeUtils::min<T>()) {
    
        rotation(0,0) = rotation(1,1) = 1.;
        rotation(0,1) = rotation(1,0) = 0.;
    }
    else {    
    
        T u = t / d;
        T tmp = math::nd4j_sqrt<T>(1. + u*u);
        rotation(0,0) = rotation(1,1) = u / tmp;
        rotation(0,1) = 1./tmp;
        rotation(1,0) = -rotation(0,1);        
    }
              
    m.assign(mmul(rotation, m));
    createJacobiRotation(m(0,0), m(0,1), m(1,1), right);

    NDArray<T> *temp = right.transpose();
    left.assign(mmul(rotation, *temp));
    delete temp;
    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::evalData(const NDArray<T>& matrix) {

    const T precision  = (T)2. * DataTypeUtils::eps<T>();  
    const T almostZero = DataTypeUtils::min<T>();

    T scale = matrix.template reduceNumber<simdOps::AMax<T>>();
    if(scale== (T)0.) 
        scale = 1.;

    if(_rows > _cols) {

        HHcolPivQR<T> qr(matrix / scale);
        _m.assign(qr._qr({{0, _cols},{0, _cols}}));
        _m.setValueInDiagMatrix(0., -1, 'l');
            
        HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');

        if(_fullUV)
            hhSeg.applyTo(_u);             
        else if(_calcU) {            
            _u.setIdentity();
            hhSeg.mulLeft(_u);
        }
        
        if(_calcV)
            _v.assign(qr._permut);
    }    
    else if(_rows < _cols) {

        NDArray<T>* matrixT = matrix.transpose();
        HHcolPivQR<T> qr(*matrixT / scale);
        _m.assign(qr._qr({{0, _rows},{0, _rows}}));
        _m.setValueInDiagMatrix(0., -1, 'l');
        _m.transposei();
    
        HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');          // type = 'u' is not mistake here !

        if(_fullUV)
            hhSeg.applyTo(_v);             
        else if(_calcV) {            
            _v.setIdentity();
            hhSeg.mulLeft(_v);        
        }
                        
        if(_calcU)
            _u.assign(qr._permut);
        
        delete matrixT;      
    }
    else {

        _m.assign(matrix({{0, _diagSize}, {0,_diagSize}}) / scale);

        if(_calcU) 
            _u.setIdentity();

        if(_calcV) 
            _v.setIdentity();
    }

    T maxDiagElem = 0.;
    for(int i = 0; i < _diagSize; ++i) {
        T current = math::nd4j_abs<T>(_m(i,i));
        if(maxDiagElem < current )
            maxDiagElem = current;
    }    

    bool stop = false;

    while(!stop) {        

        stop = true;            

        for(int p = 1; p < _diagSize; ++p) {
            
            for(int q = 0; q < p; ++q) {
        
                T threshold = math::nd4j_max<T>(almostZero, precision * maxDiagElem);                
                
                if(math::nd4j_abs<T>(_m(p,q)) > threshold || math::nd4j_abs<T>(_m(q,p)) > threshold){          
                    
                    stop = false;
                    
                    // if(isBlock2x2NotDiag(_m, p, q, maxDiagElem)) 
                    {                                                                       
                        NDArray<T> rotLeft (_m.ordering(), {2, 2}, _m.getWorkspace());
                        NDArray<T> rotRight(_m.ordering(), {2, 2}, _m.getWorkspace());
                        svd2x2(_m, p, q, rotLeft, rotRight);

                        mulRotationOnLeft(p, q, _m, rotLeft);
                                                    
                        if(_calcU) {                            
                            NDArray<T>* temp = rotLeft.transpose();
                            mulRotationOnRight(p, q, _u, *temp);
                            delete temp;
                        }                        
                        
                        mulRotationOnRight(p, q, _m, rotRight);                        

                        if(_calcV)
                            mulRotationOnRight(p, q, _v, rotRight);
            
                        maxDiagElem = math::nd4j_max<T>(maxDiagElem, math::nd4j_max<T>(math::nd4j_abs<T>(_m(p,p)), math::nd4j_abs<T>(_m(q,q))));
                    }
                }
            }
        }
    }
    
    for(int i = 0; i < _diagSize; ++i) {                
        _s(i) = math::nd4j_abs<T>(_m(i,i));
        if(_calcU && _m(i,i) < (T)0.) {
            NDArray<T>* temp = _u.subarray({{},{i, i+1}});
            temp->template applyTransform<simdOps::Neg<T>>();            
            delete temp;
        }
    }
  
    _s *= scale;
    
    for(int i = 0; i < _diagSize; i++) {
                
        int pos = (int)(_s({{i, -1}, {}}).template indexReduceNumber<simdOps::IndexMax<T>>());
        T maxSingVal =  _s({{i, -1}, {}}).template reduceNumber<simdOps::Max<T>>();

        if(maxSingVal == (T)0.)   
            break;

        if(pos) {
            
            pos += i;
            math::nd4j_swap<T>(_s(i), _s(pos));
            
            if(_calcU) {
                NDArray<T>* temp1 = _u.subarray({{}, {pos, pos+1}});
                NDArray<T>* temp2 = _u.subarray({{}, {i, i+1}});
                NDArray<T>  temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);
                delete temp1;
                delete temp2;                
            }
            
            if(_calcV) { 
                NDArray<T>* temp1 = _v.subarray({{}, {pos, pos+1}});
                NDArray<T>* temp2 = _v.subarray({{}, {i, i+1}});
                NDArray<T>  temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);
                delete temp1;
                delete temp2;                                
            }
        }
    }  
}




template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

