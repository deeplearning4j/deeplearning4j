//
//  @author raver119@gmail.com
//


#include <pointercast.h>
#include <op_boilerplate.h>
#include <NDArray.h>
#include <NDArrayFactory.h>


namespace nd4j {
    namespace ops {

        template <typename T>
        class ScatterHelper {
        public:
            template <typename OpClass>
            static FORCEINLINE Nd4jStatus scatter_apply(NDArray<T>* output, NDArray<T>* indices, NDArray<T>* updates) {
                NDArray<T>* input = output;
                int indicesLength = (int) indices->lengthOf();

            if ((indices->isVector() && input->isVector() && updates->isVector()) ||
                (input->isScalar() && input->isScalar() && updates->isScalar()) ||
                (input->isVector() && indices->isScalar() && updates->isScalar()) ) {
                
                for (int e = 0; e < indicesLength; e++) {
                    int idx = (int) indices->getScalar(e);
                    
                    T t0 = input->getScalar(idx);
                    T t1 = updates->getScalar(e);
                    
                    output->putScalar(idx, OpClass::op(t0, t1, nullptr));
                }

                return Status::OK();
            } else if (indices->isVector() || indices->isScalar()) {
                std::vector<int> idc;
                std::vector<int> idcU;

                for (int e = 0; e < indicesLength; e++) {
                    idc.push_back((int) indices->getScalar(e));
                    idcU.push_back(e);
                }

                std::vector<int> tadDimension = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {0});
                auto tadsOperand = NDArrayFactory<T>::multipleTensorsAlongDimension(output, idc, tadDimension);
                auto tadsUpdate = NDArrayFactory<T>::multipleTensorsAlongDimension(updates, idcU, tadDimension);

                auto z0 = tadsOperand->at(0);
                auto z1 = tadsUpdate->at(0);

                REQUIRE_TRUE(z0->isSameShape(z1), 0, "scatter_add: updates shapes should match");

                for (int e = 0; e < tadsOperand->size(); e++) {
                    auto t0 = tadsOperand->at(e);
                    auto t1 = tadsUpdate->at(e);
                    
                    t0->template applyPairwiseTransform<OpClass>(t1, nullptr);
                }

                delete tadsOperand;
                delete tadsUpdate;

                return Status::OK();
            }  else if (indices->isMatrix() || indices->rankOf() >= 2) {
                auto _input = input->reshape(input->ordering(), {input->sizeAt(0), -1});
                auto _updates = updates->reshape(updates->ordering(), {indicesLength, (int) updates->lengthOf() / indicesLength});

                auto tadsOperand = NDArrayFactory<T>::allTensorsAlongDimension(_input, {1});
                auto tadsUpdates = NDArrayFactory<T>::allTensorsAlongDimension(_updates, {1});

                for (int e = 0; e < indicesLength; e++) {
                    int idx = indices->getScalar(e);
                    
                    auto t0 = tadsOperand->at(idx);
                    auto t1 = tadsUpdates->at(e);

                    t0->template applyPairwiseTransform<OpClass>(t1, nullptr);
                }

                delete _input;
                delete _updates;

                delete tadsOperand;
                delete tadsUpdates;
                return Status::OK();
            }

                return Status::THROW("ScatterHelper failed");
            }
        };
    }
}