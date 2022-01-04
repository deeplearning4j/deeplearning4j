/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Created by raver119 on 20.11.17.
//
#include <build_info.h>
#include <graph/Graph.h>
#include <graph/Node.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <helpers/BenchmarkHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/GradCheck.h>
#include <helpers/Loops.h>
#include <helpers/LoopsCoordsHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/RandomLauncher.h>
#include <helpers/threshold.h>
#include <loops/type_conversions.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/reductions.h>
#include <ops/declarable/helpers/scatter.h>
#include <ops/ops.h>
#include <build_info.h>
#include <array>
#include <chrono>
#include <random>

#include "testlayers.h"
using namespace sd;
using namespace sd::graph;

class PlaygroundTests : public testing::Test {
 public:
  int numIterations = 3;
  int poolSize = 10;

  PlaygroundTests() {}
};

TEST_F(PlaygroundTests, test_avx) {
  sd_printf("Optimal level: %i; Binary level: %i;\n", ::optimalLevel(), ::binaryLevel());
}

TEST_F(PlaygroundTests, buildver) { sd_printf("%s\n", buildInfo()); }

TEST_F(PlaygroundTests, test_biasAdd_1) {
  auto x = NDArrayFactory::create<float>('c', {512, 3072});
  auto y = NDArrayFactory::create<float>('c', {3072});

  std::vector<sd::LongType> values;

  sd::ops::biasadd op;

  for (int e = 0; e < 100; e++) {
    auto timeStart = std::chrono::system_clock::now();

    op.execute({&x, &y}, {&x});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
    values.emplace_back(outerTime);
  }

  std::sort(values.begin(), values.end());

  sd_printf("Time: %lld us;\n", values[values.size() / 2]);
}

 
void bench_conv(int outter_loop, const char *msg,  const std::vector<NDArray *> &inList,  const std::vector<NDArray *> &outList,     const std::vector<sd::LongType> &iArgs ){
   std::vector<sd::LongType> values;
   if(outter_loop<1) outter_loop=1;
   if(msg==nullptr ) msg="bench:";
    sd::ops::conv2d op;
   for (int e = 0; e < outter_loop; e++) {
    auto timeStart = std::chrono::system_clock::now();

    op.execute(inList, outList, {} , iArgs);

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
    values.emplace_back(outerTime);
  }

  std::sort(values.begin(), values.end());
  double sum = 0.0;
  for(auto ss: values){
    sum+=ss;
  }
  sd_printf("%s:\nTime: %lld us; avg %f   Loop count %d\n", msg, values[values.size() / 2], sum/(double)values.size(), outter_loop);
}


TEST_F(PlaygroundTests, test_conv2d_bench) {
  // sd::Environment::getInstance().setDebug(true);
  // sd::Environment::getInstance().setVerbose(true);
  auto rr=buildInfo();
  sd_printf("info:\n%s\n",rr);
// shapeInfo input: [4,  1,27,27,512,  373248,13824,512,1,  8192,1,99]
// shapeInfo weights: [4,  3,3,512,1024,  1572864,524288,1024,1,  8192,1,99]
// shapeInfo output: [4,  1,13,13,1024,  173056,13312,1024,1,  8192,1,99]
  auto in0 = NDArrayFactory::create<float>('c', {1,27,27,512});
  auto w0 = NDArrayFactory::create<float>('c', {3,3,512,1024});
  auto out0 = NDArrayFactory::create<float>('c', {1,13,13,1024});
  in0.linspace(0);
  w0.linspace(0);
//   shapeInfo input: [4,  1,417,417,32,  5564448,13344,32,1,  8192,1,99]
//  shapeInfo weights: [4,  3,3,32,64,  6144,2048,64,1,  8192,1,99]
// shapeInfo output: [4,  1,208,208,64,  2768896,13312,64,1,  8192, z1,99]
  auto in1 = NDArrayFactory::create<float>('c', {1,417,417,32});
  auto w1 = NDArrayFactory::create<float>('c', {3,3,32,64});
  in1.linspace(0);
  w1.linspace(0);
  auto out1 = NDArrayFactory::create<float>('c', {1,208,208,64});

  auto in0_2 =in0.permute( {0,3, 1, 2} ).dup('c');
  auto w0_2 = w0.permute( {3, 2, 0, 1} ).dup('c');
  auto out0_2 = out0.permute( {0,3, 1, 2} ).dup('c');
  in0_2.printShapeInfo("in0_2");
  w0_2.printShapeInfo("w0_2");
  out0_2.printShapeInfo("out0_2");

  auto in1_2 =in1.permute( {0,3, 1, 2} ).dup('c');
  auto w1_2 = w1.permute( {3, 2, 0, 1} ).dup('c');
  auto out1_2 = out1.permute( {0,3, 1, 2} ).dup('c');
  in1_2.printShapeInfo("in1_2");
  w1_2.printShapeInfo("w1_2");
  out1_2.printShapeInfo("out1_2");
  //Running conv2d onednn with kernels: 3 3 strides: 2 2 padding: 0 0 dilation: 1 1 paddingMode 0 weightFormat 0  isNCHW 0
  int   kH = 3, kW = 3, sH = 2, sW = 2, pH = 0, pW = 0, dH = 1, dW = 1, paddingMode=0, weightFormat=0;
 
  //op.execute( std::vector<NDArray *>{&in0, &w0},  std::vector<NDArray *>{&out0},{kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, weightFormat});
  std::vector<sd::LongType> iArgs = {kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, 1, weightFormat} ;
  std::vector<sd::LongType> iArgs2 = {kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, 0, 1} ;
 

  bench_conv(20, "first ", {&in0, &w0},  {&out0}, iArgs );
  bench_conv(20, "second", {&in1, &w1},  {&out1}, iArgs  );
  bench_conv(20, "first_nchw_wformat1",{&in0_2, &w0_2},  {&out0_2},  iArgs2   );
  bench_conv(20, "second_nchw_wformat1", {&in1_2, &w1_2},  {&out1_2}, iArgs2  );

  auto out_revert0= out0_2.permute({0,2,3,1});
  auto out_revert1= out1_2.permute({0,2,3,1});
  ASSERT_TRUE(out0.equalsTo(&out_revert0));
  ASSERT_TRUE(out1.equalsTo(&out_revert1)); 
   

}

TEST_F(PlaygroundTests, tt_conv2) {
    auto in0 = NDArrayFactory::create<float>('c', { 1,2,5,4 });
    auto w0 = NDArrayFactory::create<float>('c', { 2, 2, 2,2 }); 
    in0.linspace(0);
    w0.linspace(0);
    //w0 = 1;
    int kH =2;
    int kW =2;
    int sH =1;
    int sW =1;
    int pH =0;
    int pW =0;
    int dH = 1;
    int dW= 1;
    int paddingMode =0;
    //int isNCHW = 1;
    int weightFormat= 1 ;
    //w0.permutei({1,0,3,2});
    in0.printIndexedBuffer("input");
    w0.printIndexedBuffer("weights");
    std::vector<sd::LongType> iArgs = {kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, 0, weightFormat} ;
    sd::ops::conv2d op; 
    auto res= op.evaluate({&in0, &w0},   {} , iArgs);
    auto out=res.at(0);
    out->printShapeInfo("out");
    out->printIndexedBuffer("out");

}

TEST_F(PlaygroundTests, test_bert_full_1) {
#ifdef _RELEASE

  // this test will run ONLY if this model exists
  if (sd::graph::getFileSize("/home/raver119/Downloads/BertFull/model.fb") < 0) return;

  auto graph = GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/BertFull/model.fb");

  auto t = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in0_IteratorGetNext.npy");
  auto u = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in1_IteratorGetNext_1.npy");
  auto v = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in2_IteratorGetNext_4.npy");
  auto z = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/out_loss-Softmax.npy");

  // graph->printOut();

  graph->tagInplaceNodes();

  graph->getVariableSpace()->putVariable(658, 0, t);
  graph->getVariableSpace()->putVariable(659, 0, u);
  graph->getVariableSpace()->putVariable(660, 0, v);

  /*
      // validating graph now
      auto status = GraphExecutioner::execute(graph);
      ASSERT_EQ(Status::OK, status);
      ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1620));

      auto array = graph->getVariableSpace()->getVariable(1620)->getNDArray();
      ASSERT_EQ(z, *array);

  */

  sd::Environment::getInstance().setProfiling(true);
  auto profile = GraphProfilingHelper::profile(graph, 1);

  profile->printOut();

  sd::Environment::getInstance().setProfiling(false);
  delete profile;

  /*
      std::vector<sd::LongType> values;

      for (int e = 0; e < 1; e++) {
          auto timeStart = std::chrono::system_clock::now();

          GraphExecutioner::execute(graph);

          auto timeEnd = std::chrono::system_clock::now();
          auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
          values.emplace_back(outerTime);
      }

      std::sort(values.begin(), values.end());

      sd_printf("Time: %lld us;\n", values[values.size() / 2]);
  */
  delete graph;
#endif
}

TEST_F(PlaygroundTests, test_bert_1) {
#ifdef _RELEASE
  // this test will run ONLY if this model exists
  if (sd::graph::getFileSize("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model.fb") < 0) return;

  auto graph =
      GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model.fb");

  auto t = NDArrayFactory::fromNpyFile(
      "/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext.numpy");
  auto u = NDArrayFactory::fromNpyFile(
      "/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext_1.numpy");
  auto v = NDArrayFactory::fromNpyFile(
      "/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext_4.numpy");
  auto z = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model_output.numpy");

  // graph->printOut();

  graph->tagInplaceNodes();

  graph->getVariableSpace()->putVariable(85, 0, t);
  graph->getVariableSpace()->putVariable(86, 0, u);
  graph->getVariableSpace()->putVariable(87, 0, v);

  /*
      // validating graph now
      auto status = GraphExecutioner::execute(graph);
      ASSERT_EQ(Status::OK, status);
      ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

      auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
      ASSERT_EQ(z, *array);

  */
  sd::Environment::getInstance().setProfiling(true);
  auto profile = GraphProfilingHelper::profile(graph, 1);

  profile->printOut();

  sd::Environment::getInstance().setProfiling(false);
  delete profile;

  /*
      std::vector<sd::LongType> values;

      for (int e = 0; e < 1; e++) {
          auto timeStart = std::chrono::system_clock::now();

          GraphExecutioner::execute(graph);

          auto timeEnd = std::chrono::system_clock::now();
          auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
          values.emplace_back(outerTime);
      }

      std::sort(values.begin(), values.end());

      sd_printf("Time: %lld us;\n", values[values.size() / 2]);
  */
  delete graph;
#endif
}

TEST_F(PlaygroundTests, test_bert_2) {
#ifdef _RELEASE
  // this test will run ONLY if this model exists
  if (sd::graph::getFileSize("/home/raver119/Downloads/Bert_minimal_model/bert_like_ops.fb") < 0) return;

  auto graph = GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/Bert_minimal_model/bert_like_ops.fb");

  // graph->printOut();

  graph->tagInplaceNodes();

  /*
      // validating graph now
      auto status = GraphExecutioner::execute(graph);
      ASSERT_EQ(Status::OK, status);
      ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

      auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
      ASSERT_EQ(z, *array);
  */

  sd::Environment::getInstance().setProfiling(true);
  auto profile = GraphProfilingHelper::profile(graph, 1);

  profile->printOut();

  sd::Environment::getInstance().setProfiling(false);
  delete profile;

  /*
      std::vector<sd::LongType> values;

      for (int e = 0; e < 1; e++) {
          auto timeStart = std::chrono::system_clock::now();

          GraphExecutioner::execute(graph);

          auto timeEnd = std::chrono::system_clock::now();
          auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
          values.emplace_back(outerTime);
      }

      std::sort(values.begin(), values.end());

      sd_printf("Time: %lld us;\n", values[values.size() / 2]);
  */
  delete graph;
#endif
}

TEST_F(PlaygroundTests, test_one_off_ops_1) {
  auto x = NDArrayFactory::create<float>('c', {4, 128, 768});
  auto y = NDArrayFactory::create<float>('c', {4, 128, 1});
  auto z = x.ulike();

  sd::ops::squaredsubtract op;
  op.execute({&x, &y}, {&z});
}

#if defined(INDEX_REDUCTIONS_BENCH_TESTS)
// temporarly, testing against the original one
void original_argmax(const NDArray& input, std::vector<int>& axis, NDArray& output) {
  sd::ops::helpers::adjustAxis(input.rankOf(), axis);
  input.applyIndexReduce(sd::indexreduce::IndexMax, output, axis);
}

template <typename T>
void fill_random(sd::NDArray& arr) {
  sd::LongType coords[SD_MAX_RANK] = {};
  std::random_device rd;
  std::mt19937 gen(rd());
  // for floats
  std::uniform_real_distribution<T> dis((T)-10.0, (T)22.9);
  T* x = arr.bufferAsT<T>();
  auto shapeInfo = arr.shapeInfo();
  auto strides = arr.stridesOf();
  sd::LongType rank = shapeInfo[0];
  auto bases = &(shapeInfo[1]);
  size_t t = 1;
  for (size_t i = 0; i < rank; i++) {
    t *= bases[i];
  }
  size_t offset = 0;
  if (arr.ordering() == 'c') {
    for (size_t i = 0; i < t; i++) {
      x[offset] = dis(gen);
      offset = sd::inc_coords(bases, strides, coords, offset, rank);
    }

  } else {
    for (size_t i = 0; i < t; i++) {
      x[offset] = dis(gen);
      offset = sd::inc_coords<false>(bases, strides, coords, offset, rank);
    }
  }
}

void testLegacy(bool random) {
#if 0
    int bases[] = { 3, 2, 4, 5, 7 };
    constexpr int Loop = 1;
#else
  int bases[] = {8, 32, 64, 32, 64};
  constexpr int Loop = 10;
#endif
  constexpr int N = 5;

  auto x = NDArrayFactory::create<float>('c', {bases[0], bases[1], bases[2], bases[3], bases[4]});
  if (!random) {
    x.linspace(1);
  } else {
    fill_random<float>(x);
  }

#define COMBINATIONS 1
#if COMBINATIONS
  // https://www.rosettacode.org/wiki/Combinations#C.2B.2B
  for (int k = N; k >= 1; k--) {
    std::string bitmask(k, 1);  // K leading 1's
    bitmask.resize(N, 0);       // N-K trailing 0's

    do {
      std::vector<int> dimension;
      std::vector<sd::LongType> output_bases;

      for (int i = 0; i < N; ++i)  // [0..N-1] integers
      {
        if (bitmask[i])
          dimension.push_back(i);
        else {
          output_bases.push_back(bases[i]);
        }
      }
#else
  std::vector<int> dimension = {0, 1, 2, 3};
  int k = 4;
#endif
      auto dim = NDArrayFactory::create<int>(dimension);

#if 1
      sd_printf("C(N:%d K:%d) \n", N, k);
      dim.printIndexedBuffer("Dimension");
      for (int xind : dimension) {
        sd_printf(" %d ,", bases[xind]);
      }
      sd_printf("%s", "\n");
#endif

      std::vector<sd::LongType> values;
      sd::ResultSet result;
      for (int e = 0; e < Loop; e++) {
        auto timeStart = std::chrono::system_clock::now();
        NDArray exp = output_bases.size() > 0 ? NDArrayFactory::create<sd::LongType>('c', output_bases)
                                              : NDArrayFactory::create<sd::LongType>(0);
        original_argmax(x, dimension, exp);
        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
      }

      std::sort(values.begin(), values.end());

      sd_printf("Time: %lld us;\n", values[values.size() / 2]);
#if COMBINATIONS

    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  }
#endif
}

#define DEBUG 1

void testNewReduction(bool random, bool checkCorrectness = false, char order = 'c') {
  std::vector<sd::LongType> arr_dimensions;
#if defined(DEBUG)
  int bases[] = {3, 2, 3, 3, 5, 4, 7, 4, 7, 7};
  constexpr int Loop = 1;
  constexpr int N = 10;
#else
  int bases[] = {8, 32, 64, 32, 64};
  constexpr int Loop = 10;
  constexpr int N = 5;

#endif

  for (int i = 0; i < N; i++) {
    arr_dimensions.push_back(bases[i]);
  }
  auto x = NDArrayFactory::create<float>(order, arr_dimensions);
  if (!random) {
    x.linspace(1);
  } else {
    fill_random<float>(x);
  }

#define COMBINATIONS 1
#if COMBINATIONS
  // https://www.rosettacode.org/wiki/Combinations#C.2B.2B
  for (int k = N; k >= 1; k--) {
    std::string bitmask(k, 1);  // K leading 1's
    bitmask.resize(N, 0);       // N-K trailing 0's

    do {
      std::vector<int> dimension;

      std::vector<sd::LongType> output_bases;

      for (int i = 0; i < N; ++i)  // [0..N-1] integers
      {
        if (bitmask[i])
          dimension.push_back(i);
        else {
          output_bases.push_back(bases[i]);
        }
      }
#else
  std::vector<int> dimension = {0, 1, 2, 3};
  int k = 4;
#endif
      auto dim = NDArrayFactory::create<int>(dimension);

#if 1
      sd_printf("C(N:%d K:%d) \n", N, k);
      dim.printIndexedBuffer("Dimension");
      for (int xind : dimension) {
        sd_printf(" %d ,", bases[xind]);
      }
      sd_printf("%s", "\n");
#endif

      sd::ops::argmax op;
      std::vector<sd::LongType> values;
      sd::ResultSet result;
      for (int e = 0; e < Loop; e++) {
        auto timeStart = std::chrono::system_clock::now();
        result = op.evaluate({&x, &dim}, {}, {});
        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
      }
      auto z = result.at(0);

      if (checkCorrectness) {
        // check for the correctness
        NDArray exp = output_bases.size() > 0 ? NDArrayFactory::create<sd::LongType>('c', output_bases)
                                              : NDArrayFactory::create<sd::LongType>(0);
        original_argmax(x, dimension, exp);

#if 0  // defined(DEBUG)
     x.printIndexedBuffer("X");
    exp.printIndexedBuffer("Expected");
    z->printIndexedBuffer("Z");
#endif

        ASSERT_TRUE(exp.isSameShape(z));
        ASSERT_TRUE(exp.equalsTo(z));
      }
      std::sort(values.begin(), values.end());

      sd_printf("Time: %lld us;\n", values[values.size() / 2]);
#if COMBINATIONS

    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  }
#endif
}

constexpr bool test_corr = true;
#if !defined(DEBUG)
TEST_F(PlaygroundTests, ArgMaxPerfLinspace) { testNewReduction(false, test_corr); }
#endif

TEST_F(PlaygroundTests, ArgMaxPerfRandom) { testNewReduction(true, test_corr); }

TEST_F(PlaygroundTests, ArgMaxPerfRandomOrderF) { testNewReduction(true, test_corr, 'f'); }

#if !defined(DEBUG)
TEST_F(PlaygroundTests, ArgMaxPerfLegacyLinspace) { testLegacy(false); }

TEST_F(PlaygroundTests, ArgMaxPerfLegacyRandom) { testLegacy(true); }

#endif

#endif

/*

TEST_F(PlaygroundTests, test_broadcast_1) {
    int pool = 1000;
    std::vector<NDArray*> aX(pool);
    std::vector<NDArray*> aY(pool);
    std::vector<NDArray*> aZ(pool);

    for (int e = 0; e < pool; e++) {
        aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
        aY[e] = NDArrayFactory::create_<float>('c', {3072});
        aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

        aX[e]->assign(119 * (e+1));
        aY[e]->assign(119 * (e+3));
    }

    std::vector<sd::LongType> values;
    Context ctx(1);

    sd::ops::biasadd op;

    for (int e = 0; e < 1000; e++) {
        auto x = aX[e < pool ? e : e % pool];
        auto y = aY[e < pool ? e : e % pool];
        auto z = aZ[e < pool ? e : e % pool];

        auto timeStart = std::chrono::system_clock::now();

        //op.execute({x, y}, {z});
        sd::ops::helpers::addBias(ctx, *x, *y, *z, false);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    sd_printf("Time: %lld us;\n", values[values.size() / 2]);

    for (int e = 0; e < pool; e++) {
        delete aX[e];
        delete aY[e];
        delete aZ[e];
    }
}


/*
TEST_F(PlaygroundTests, test_broadcast_1) {
    int pool = 500;
    std::vector<NDArray*> aX(pool);
    std::vector<NDArray*> aY(pool);
    std::vector<NDArray*> aZ(pool);

    for (int e = 0; e < pool; e++) {
        aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
        aY[e] = NDArrayFactory::create_<float>('c', {768});
        aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

        aX[e]->assign( (e+1) / 119);
        aY[e]->assign( (e+3) / 119);
    }


    std::vector<sd::LongType> values;

    for (int e = 0; e < 1000; e++) {
        auto x = aX[e < pool ? e : e % pool];
        auto y = aY[e < pool ? e : e % pool];
        auto z = aZ[e < pool ? e : e % pool];

        auto timeStart = std::chrono::system_clock::now();

        //x->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), *y, *z);
        x->applyTransform(transform::Tanh, *z, nullptr);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    sd_printf("Time: %lld us;\n", values[values.size() / 2]);

    for (int e = 0; e < pool; e++) {
        delete aX[e];
        delete aY[e];
        delete aZ[e];
    }
}

*/
/*

TEST_F(PlaygroundTests, test_s_0) {
    std::vector<std::vector<sd::LongType>> shapes = {{32, 224, 224, 3}, {32, 56, 56, 64}, {32, 7, 7, 512}};
    std::vector<int> threads = {1, 2, 4, 8, 16};

    for (auto shape: shapes) {
        for (auto t: threads) {
            sd::Environment::getInstance().setMaxMasterThreads(t);

            auto x = NDArrayFactory::create<float>('c', shape);
            auto y = NDArrayFactory::create<float>('c', {shape[3]});
            auto z = x.ulike();

            std::vector<sd::LongType> values;
            Context ctx(1);
            ctx.setInputArray(0, &x);
            ctx.setInputArray(1, &y);
            ctx.setOutputArray(0, &z);

            sd::ops::biasadd op;


            for (int e = 0; e < 10000; e++) {
                auto timeStart = std::chrono::system_clock::now();

                op.execute(&ctx);
                sd::ops::helpers::addBias(ctx, x, y, z, false);

                auto timeEnd = std::chrono::system_clock::now();
                auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
                values.emplace_back(outerTime);
            }

            std::sort(values.begin(), values.end());

            sd_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2],
shape[3], t, values[values.size() / 2]);
        }
    }
}

TEST_F(PlaygroundTests, test_s_1) {
    std::vector<std::vector<sd::LongType>> shapes = {{32, 3, 224, 224}, {32, 64, 56, 56}, {32, 512, 7, 7}};
    std::vector<int> threads = {1, 2, 4, 8, 16};

    for (auto shape: shapes) {
        for (auto t: threads) {
            sd::Environment::getInstance().setMaxMasterThreads(t);

            auto x = NDArrayFactory::create<float>('c', shape);
            auto y = NDArrayFactory::create<float>('c', {shape[1]});
            auto z = x.ulike();

            std::vector<sd::LongType> values;
            Context ctx(1);
            ctx.setInputArray(0, &x);
            ctx.setInputArray(1, &y);
            ctx.setOutputArray(0, &z);

            sd::ops::biasadd op;


            for (int e = 0; e < 10000; e++) {
                auto timeStart = std::chrono::system_clock::now();

                //op.execute({&x, &y}, {&z}, {true});
                sd::ops::helpers::addBias(ctx, x, y, z, true);

                auto timeEnd = std::chrono::system_clock::now();
                auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
                values.emplace_back(outerTime);
            }

            std::sort(values.begin(), values.end());

            sd_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2],
shape[3], t, values[values.size() / 2]);
        }
    }
}
*/

/*
TEST_F(PlaygroundTests, test_s_0) {
    auto x = NDArrayFactory::create<float>('c', {32, 112, 112, 16});
    auto y = NDArrayFactory::create<float>('c', {16});
    auto z = x.ulike();

    std::vector<sd::LongType> values;
    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setInputArray(1, &y);
    ctx.setOutputArray(0, &z);

    sd::ops::biasadd op;


    for (int e = 0; e < 10000; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    sd_printf("Time: %lld us;\n", values[values.size() / 2]);
}
*/
/*
TEST_F(PlaygroundTests, test_s_1) {
    auto x0 = NDArrayFactory::create<float>('c', {32, 7, 7, 176});
    auto x1 = x0.ulike();
    auto x2 = x0.ulike();
    auto x3 = x0.ulike();
    auto x4 = x0.ulike();
    auto x5 = x0.ulike();

    auto y = NDArrayFactory::create<int >(3);
    auto z = NDArrayFactory::create<float>('c', {32, 7, 7, 1056});

    Context ctx(1);
    ctx.setInputArray(0, &x0);
    ctx.setInputArray(1, &x1);
    ctx.setInputArray(2, &x2);
    ctx.setInputArray(3, &x3);
    ctx.setInputArray(4, &x4);
    ctx.setInputArray(5, &x5);

    ctx.setInputArray(6, &y);
    ctx.setOutputArray(0, &z);
    ctx.setBArguments({true});

    std::vector<sd::LongType> values;

    sd::ops::concat op;
    op.execute(&ctx);

    for (int e = 0; e < 1000; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }


    std::sort(values.begin(), values.end());

    sd_printf("Time: %lld us;\n", values[values.size() / 2]);
}
*/

/*
TEST_F(PlaygroundTests, test_s_1) {
    auto t = ::runLightBenchmarkSuit(true);
    delete[] t;
}

TEST_F(PlaygroundTests, test_s_2) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    std::vector<sd::LongType> values;

    for (int e = 0; e < 100000; e++) {
        s = 0;

        auto timeStart = std::chrono::system_clock::now();
        //samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
        PRAGMA_OMP_PARALLEL_THREADS(4) {
            s++;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    };
    std::sort(values.begin(), values.end());

    sd_printf("Time: %lld;\n", values[values.size() / 2]);
}
 */
/*
TEST_F(PlaygroundTests, test_s_4) {
    std::atomic<float> f;
    std::atomic<int> s;
    std::vector<sd::LongType> valuesX, valuesY;
    int iterations = 1000;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);

    ////////

    auto x = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    auto z = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    x.linspace(1.0);

    auto xs0 = x.sizeAt(0);
    auto xs1 = x.sizeAt(1);
    auto xs2 = x.sizeAt(2);
    auto xs3 = x.sizeAt(3);

    auto buffer = x.bufferAsT<float>();
    auto zbuffer = z.bufferAsT<float>();

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int i = 0; i < xs0; i++) {
            for (int j = 0; j < xs1; j++) {
                auto thread_id = omp_get_thread_num();
                for (int k = 0; k < xs2; k++) {
                    for (int l = 0; l < xs3; l++) {
                        zbuffer[thread_id] += buffer[i * j + (k*l)] * 2.5f;
                    }
                }
            }
        }
        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesX.emplace_back(outerTime);
    }


    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        auto f2d = PRAGMA_THREADS_FOR_2D {
            for (auto i = start_x; i < stop_x; i++) {
                for (auto j = start_y; j < stop_y; j++) {

                    for (auto k = 0; k < xs2; k++) {
                        for (auto l = 0; l < xs3; l++) {
                            zbuffer[thread_id] += buffer[i * j + (k * l)] * 2.5f;
                        }
                    }
                }
            }
        };
        samediff::Threads::parallel_for(f2d, 0, xs0, 1, 0, xs1, 1);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesY.emplace_back(outerTime);
    }

    if (valuesX.size() > 0) {
        std::sort(valuesX.begin(), valuesX.end());
        sd_printf("OpenMP time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0],
valuesX[valuesX.size() - 1]);
    }

    if (valuesY.size() > 0) {
        std::sort(valuesY.begin(), valuesY.end());
        sd_printf("Threads time: %lld; Min: %lld; Max: %lld;\n", valuesY[valuesY.size() / 2], valuesY[0],
valuesY[valuesY.size() - 1]);
    }

    sd_printf("Sum: %f\n", z.sumNumber().e<float>(0));
}


TEST_F(PlaygroundTests, test_s_5) {
    auto x = NDArrayFactory::create<float>('c', {32, 1, 28, 28});

    std::vector<sd::LongType> values;
    auto iterations = 100;

    auto startX = 0;
    auto stopX = x.sizeAt(0);
    auto incX = 1;
    auto startY = 0;
    auto stopY = x.sizeAt(1);
    auto incY = 1;
    auto numThreads = 4;

    // number of elements per loop
    auto delta_x = (stopX - startX);
    auto delta_y = (stopY - startY);

    // number of iterations per loop
    auto itersX = delta_x / incX;
    auto itersY = delta_y / incY;

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        auto splitLoop = samediff::ThreadsHelper::pickLoop2d(numThreads, itersX, itersY);
        auto span = samediff::Span2::build(splitLoop, 0, numThreads, startX, stopX, incX, startY, stopY, incY);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    sd_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0],
values[values.size()-1]);
}


TEST_F(PlaygroundTests, test_s_6) {
    auto x = NDArrayFactory::create<float>('c', {1024 * 1024 * 64});
    auto buffer = x.bufferAsT<float>();
    auto len = x.lengthOf();
    std::vector<sd::LongType> values;
    auto iterations = 1000;

    for (int i = 0; i < iterations; i++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        for (int e = 0; e < len; e++) {
            buffer[e] = (buffer[e] + 1.72f) * 3.17f - 0.0012f;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    sd_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0],
values[values.size()-1]);
}


TEST_F(PlaygroundTests, test_s_3) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    for (int e = 0; e < 10000; e++) {

        samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    }
}
 */

/*
TEST_F(PlaygroundTests, test_relubp_1) {
    auto x = NDArrayFactory::create<float>('c', {128, 64, 224, 224});
    auto y = x.ulike();
    auto z = x.ulike();
    RandomGenerator rng(119, 120);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &x, -1.0, 1.0);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &y, -1.0, 1.0);

    int iterations = 10;

    auto timeStart = std::chrono::system_clock::now();
    for (int e = 0; e < iterations; e++)
        ops::helpers::reluDerivative(LaunchContext::defaultContext(), &x, &y, &z);
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    auto time = (sd::LongType) outerTime / iterations;
    auto bw = (1000000L * (float) (x.lengthOf() * x.sizeOfT()) / time) / 1024 / 1024 / 1024;

    sd_printf("Time: %lld; BW: %f GB/s\n", time, bw);
}

//////////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, my) {

    int bS=8, iD=32,iH=32,iW=32,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
    int       oD,oH,oW;

    sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
iW, 0);

    printf("!!%i, %i, %i\n", oD,oH,oW);

    NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
    NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);

    col = 3.77;
    vol = -10.33;

    auto variableSpace = new VariableSpace();
    auto block = new Context(1, variableSpace, false);  // not-in-place

    auto timeStart = std::chrono::system_clock::now();
    sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    printf("time: %i \n", time);

    delete block;
    delete variableSpace;
}

TEST_F(PlaygroundTests, my) {

    int bS=32, iD=32,iH=64,iW=64,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
    int       oD,oH,oW;

    // sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD,
iH, iW, 0); sd::ops::ConvolutionUtils::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW,dH, dW, iH, iW, 0);

    printf("!!%i, %i, %i\n", oD,oH,oW);

    // NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
    // NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);
    NDArray col('c', {bS, iC, kH, kW, iH, iW}, sd::DataType::DOUBLE);
    NDArray im('c', {bS, iC, oH, oW}, sd::DataType::DOUBLE);

    col = 3.77;
    // vol = -10.33;
    im = -10.33;

    auto variableSpace = new VariableSpace();
    auto block = new Context(1, variableSpace, false);  // not-in-place

    auto timeStart = std::chrono::system_clock::now();
    // sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
    sd::ops::helpers::col2im(*col.getContext(), col, im, sH, sW, pH, pW, iH, iW, dH, dW);
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    printf("time: %i \n", time);

    delete block;
    delete variableSpace;
}

///////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, lstmLayerCellBp_1) {

    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;
    // const int nIn  = 8;
    // const int nOut = 6;

    const float cellClip = 1.1;       // clipping value
    const sd::LongType gateAct = 2;        // sigmoid activation for input (i), forget (f) and output (o) gates
    const float gateAlpha = 0;      // alpha value for activation for gates, not required for sigmoid
    const float gateBeta = 0;       // beta value for activation for gates, not required for sigmoid
    const sd::LongType cellAct = 0;        // tanh activation for cell state
    const float cellAlpha = 0;      // alpha value for cell state activation, not required for tanh
    const float cellBeta = 0;       // beta value for cell state activation, not required for tanh
    const sd::LongType outAct = 0;         // tanh activation for output
    const float outAlpha = 0;       // alpha value for output activation, not required for tanh
    const float outBeta = 0;        // beta value for output activation, not required for tanh

    NDArray x ('c',   {bS, nIn}, sd::DataType::DOUBLE);
    NDArray hI('c',   {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c',   {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdc('c', {bS, nOut}, sd::DataType::DOUBLE);

    // NDArray x ('c',   {nIn}, sd::DataType::DOUBLE);
    // NDArray hI('c',   {nOut}, sd::DataType::DOUBLE);
    // NDArray cI('c',   {nOut}, sd::DataType::DOUBLE);
    // NDArray dLdh('c', {nOut}, sd::DataType::DOUBLE);
    // NDArray dLdc('c', {nOut}, sd::DataType::DOUBLE);

    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b ('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);

    x.linspace(-4,1);
    hI.linspace(-2.5,0.5);
    cI.linspace(-3,0.5);
    Wx.linspace(0,0.1);
    Wr.linspace(3,-0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    // x.assign(1.);
    // hI.assign(2.);
    // cI.assign(3.);
    // Wx.assign(0.5);
    // Wr.assign(0.5);
    // Wp.assign(0.75);
    // b.assign(0.7);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {gateAct, cellAct, outAct};

    // std::vector<bool>     bArgs = {false, false};
    // const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &hI, &cI}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &hI, &cI, &dLdh}, tArgs, iArgs, bArgs);

    std::vector<bool>     bArgs = {true, true};
    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &hI, &cI, &Wp, &dLdh}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayerCell opFF;
    sd::ops::lstmLayerCellBp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
true, true, true});
}

TEST_F(PlaygroundTests, my) {

    const int N = 40;

    NDArray x('c', {256,256,128,128}, sd::DataType::FLOAT32);
    NDArray z1('c', {256,2,128}, sd::DataType::DOUBLE);
    NDArray z = z1({0,0,0,1,0,0});
    z.printShapeInfo();

    auto timeStart = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        // x.reduceAlongDimension(sd::reduce::Mean, z, {1,3});
        x.applyBroadcast(sd::broadcast::Ops::Add, {1,3}, z, x);
    }
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart) / N).count();

    printf("old %ld\n", time);
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_1) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 3;

    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;  // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {sL, bS, nIn}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {sL, bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_2) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 3;

    const int dataFormat = 1;       // [bS,sL,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = false;  // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {bS, sL, nIn}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, sL, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &hI, &cI, &Wp, &dLdh, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, std::vector<bool>(),
{0., 1.}, GradCheck::LossFunc::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_3) {

    const int sL   = 4;
    const int bS   = 3;
    const int nIn  = 3;
    const int nOut = 2;

    const int dataFormat = 2;       // [bS, nIn, sL]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       //  clipping

    NDArray x('c', {bS, nIn, sL}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray seqLen('c', {bS}, {2,0,4}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, nOut, sL}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs,
bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
false, true, true, true});

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_4) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 3;

    const int dataFormat = 1;       // [bS,sL,nIn]
    const int directionMode = 1;    // backward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {bS, sL, nIn}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, sL, nOut}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_5) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 2;

    const int dataFormat = 2;       // [bS, nIn, sL]
    const int directionMode = 1;    // backward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {bS, nIn, sL}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {4*nOut}, sd::DataType::DOUBLE);
    NDArray seqLen('c', {bS}, {0,2}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, nOut, sL}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs,
bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
false, true, true, true});

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_6) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 2;

    const int dataFormat = 2;       // [bS, nIn, sL]
    const int directionMode = 2;    // bidirectional sum
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {bS, nIn, sL}, sd::DataType::DOUBLE);
    NDArray Wx('c', {2, nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {2, nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {2, 4*nOut}, sd::DataType::DOUBLE);
    NDArray seqLen('c', {bS}, {0,2}, sd::DataType::DOUBLE);
    NDArray hI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {2, 3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS, nOut, sL}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {2, bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs,
bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
false, true, true, true});

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_7) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 2;

    const int dataFormat = 1;       // [bS,sL,nIn]
    const int directionMode = 3;    // bidirectional concat
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {bS,sL,nIn}, sd::DataType::DOUBLE);
    NDArray Wx('c', {2, nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {2, nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {2, 4*nOut}, sd::DataType::DOUBLE);
    NDArray seqLen('c', {bS}, {0,2}, sd::DataType::DOUBLE);
    NDArray hI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {2, 3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {bS,sL,2*nOut}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {2, bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs,
bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
false, true, true, true});

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_bp_8) {

    const int sL   = 3;
    const int bS   = 2;
    const int nIn  = 2;
    const int nOut = 2;

    const int dataFormat = 3;       // [sL, bS, nIn]
    const int directionMode = 4;    // bidirectional extra output dim
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // dLdh per each time step
    const auto retLastH   = true;   // output at last time step
    const auto retLastC   = true;   // cells state at last time step

    const double cellClip = 0.5;       // clipping

    NDArray x('c', {sL, bS, nIn}, sd::DataType::DOUBLE);
    NDArray Wx('c', {2, nIn, 4*nOut}, sd::DataType::DOUBLE);
    NDArray Wr('c', {2, nOut, 4*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {2, 4*nOut}, sd::DataType::DOUBLE);
    NDArray seqLen('c', {bS}, {0,2}, sd::DataType::DOUBLE);
    NDArray hI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray cI('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray Wp('c', {2, 3*nOut}, sd::DataType::DOUBLE);
    NDArray dLdh('c', {sL, 2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdhL('c', {2, bS, nOut}, sd::DataType::DOUBLE);
    NDArray dLdcL('c', {2, bS, nOut}, sd::DataType::DOUBLE);

    x.linspace(-2,0.1);
    hI.linspace(-1.5,0.1);
    cI.linspace(0.7,-0.1);
    Wx.linspace(1,-0.1);
    Wr.linspace(-1,0.1);
    Wp.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    std::vector<double>   tArgs = {cellClip};
    std::vector<sd::LongType> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::vector<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    const OpArgsHolder argsHolderFF({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);
    const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL, &dLdcL}, tArgs, iArgs,
bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL, &dLdcL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdh}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdhL}, tArgs, iArgs, bArgs);
    // const OpArgsHolder argsHolderBP({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp, &dLdcL}, tArgs, iArgs, bArgs);

    sd::ops::lstmLayer opFF;
    sd::ops::lstmLayer_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {true, true, true, true,
false, true, true, true});

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, gru_bp_1) {

    const int sL = 3;
    const int bS = 2;
    const int nIn = 5;
    const int nOut = 4;


    NDArray x('c', {sL, bS, nIn}, {0.5,  1. ,  1.5,  2. ,  2.5, 3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7.
,  7.5, 8. ,  8.5,  9. ,  9.5, 10. ,  10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5, 15.}, sd::DataType::DOUBLE);
    NDArray hI('c', {bS, nOut}, {-3,-2,-1,0,1,2,3,4}, sd::DataType::DOUBLE);
    NDArray Wx('c', {nIn, 3*nOut}, sd::DataType::DOUBLE);
    NDArray Wh('c', {nOut, 3*nOut}, sd::DataType::DOUBLE);
    NDArray b('c', {3*nOut}, sd::DataType::DOUBLE);

    NDArray dLdh('c', {sL, bS, nOut}, sd::DataType::DOUBLE);

    Wx.linspace(1,-0.1);
    Wh.linspace(0.2,0.2);
    b.linspace(1,-0.15);

    const OpArgsHolder argsHolderFF({&x, &hI, &Wx, &Wh, &b}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &hI, &Wx, &Wh, &b, &dLdh}, {}, {});

    sd::ops::gru opFF;
    sd::ops::gru_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
}

*/
