// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <benchmark/benchmark.h>

#include "arrow/array/concatenate.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/util/key_value_metadata.h"

namespace arrow {
namespace compute {

const int64_t elems = 1024 * 1024;

template <typename Type>
static void IfElseBench(benchmark::State& state) {
  using CType = typename Type::c_type;
  auto type = TypeTraits<Type>::type_singleton();
  using ArrayType = typename TypeTraits<Type>::ArrayType;

  int64_t len = state.range(0);
  int64_t offset = state.range(1);

  random::RandomArrayGenerator rand(/*seed=*/0);

  auto cond = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto left = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  for (auto _ : state) {
    ABORT_NOT_OK(IfElse(cond->Slice(offset), left->Slice(offset), right->Slice(offset)));
  }

  state.SetBytesProcessed(state.iterations() *
                          ((len - offset) / 8 + 2 * (len - offset) * sizeof(CType)));
}

template <typename Type>
static void IfElseBenchContiguous(benchmark::State& state) {
  using CType = typename Type::c_type;
  auto type = TypeTraits<Type>::type_singleton();
  using ArrayType = typename TypeTraits<Type>::ArrayType;

  int64_t len = state.range(0);
  int64_t offset = state.range(1);

  ASSERT_OK_AND_ASSIGN(auto temp1, MakeArrayFromScalar(BooleanScalar(true), len / 2));
  ASSERT_OK_AND_ASSIGN(auto temp2,
                       MakeArrayFromScalar(BooleanScalar(false), len - len / 2));
  ASSERT_OK_AND_ASSIGN(auto concat, Concatenate({temp1, temp2}));
  auto cond = std::static_pointer_cast<BooleanArray>(concat);

  random::RandomArrayGenerator rand(/*seed=*/0);
  auto left = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  for (auto _ : state) {
    ABORT_NOT_OK(IfElse(cond->Slice(offset), left->Slice(offset), right->Slice(offset)));
  }

  state.SetBytesProcessed(state.iterations() *
                          ((len - offset) / 8 + 2 * (len - offset) * sizeof(CType)));
}

static void IfElseBench64(benchmark::State& state) {
  return IfElseBench<UInt64Type>(state);
}

static void IfElseBench32(benchmark::State& state) {
  return IfElseBench<UInt32Type>(state);
}

static void IfElseBench64Contiguous(benchmark::State& state) {
  return IfElseBenchContiguous<UInt64Type>(state);
}

static void IfElseBench32Contiguous(benchmark::State& state) {
  return IfElseBenchContiguous<UInt32Type>(state);
}

template <typename Type>
static void CaseWhenBench(benchmark::State& state) {
  using CType = typename Type::c_type;
  auto type = TypeTraits<Type>::type_singleton();
  using ArrayType = typename TypeTraits<Type>::ArrayType;

  int64_t len = state.range(0);
  int64_t offset = state.range(1);

  random::RandomArrayGenerator rand(/*seed=*/0);

  auto cond1 = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto cond2 = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto cond3 = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto cond_field =
      field("cond", boolean(), key_value_metadata({{"null_probability", "0.01"}}));
  auto cond = rand.ArrayOf(*field("", struct_({cond_field, cond_field, cond_field}),
                                  key_value_metadata({{"null_probability", "0.0"}})),
                           len);
  auto val1 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto val2 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto val3 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto val4 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  for (auto _ : state) {
    ABORT_NOT_OK(
        CaseWhen(cond->Slice(offset), {val1->Slice(offset), val2->Slice(offset),
                                       val3->Slice(offset), val4->Slice(offset)}));
  }

  state.SetBytesProcessed(state.iterations() * (len - offset) * sizeof(CType));
}

template <typename Type>
static void CaseWhenBenchContiguous(benchmark::State& state) {
  using CType = typename Type::c_type;
  auto type = TypeTraits<Type>::type_singleton();
  using ArrayType = typename TypeTraits<Type>::ArrayType;

  int64_t len = state.range(0);
  int64_t offset = state.range(1);

  ASSERT_OK_AND_ASSIGN(auto trues, MakeArrayFromScalar(BooleanScalar(true), len / 3));
  ASSERT_OK_AND_ASSIGN(auto falses, MakeArrayFromScalar(BooleanScalar(false), len / 3));
  ASSERT_OK_AND_ASSIGN(auto nulls, MakeArrayOfNull(boolean(), len - 2 * (len / 3)));
  ASSERT_OK_AND_ASSIGN(auto concat, Concatenate({trues, falses, nulls}));
  auto cond1 = std::static_pointer_cast<BooleanArray>(concat);

  random::RandomArrayGenerator rand(/*seed=*/0);
  auto cond2 = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto val1 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto val2 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto val3 = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  ASSERT_OK_AND_ASSIGN(
      auto cond, StructArray::Make({cond1, cond2}, std::vector<std::string>{"a", "b"},
                                   nullptr, /*null_count=*/0));

  for (auto _ : state) {
    ABORT_NOT_OK(CaseWhen(cond->Slice(offset), {val1->Slice(offset), val2->Slice(offset),
                                                val3->Slice(offset)}));
  }

  state.SetBytesProcessed(state.iterations() * (len - offset) * sizeof(CType));
}

static void CaseWhenBench64(benchmark::State& state) {
  return CaseWhenBench<UInt64Type>(state);
}

static void CaseWhenBench64Contiguous(benchmark::State& state) {
  return CaseWhenBenchContiguous<UInt64Type>(state);
}

BENCHMARK(IfElseBench32)->Args({elems, 0});
BENCHMARK(IfElseBench64)->Args({elems, 0});

BENCHMARK(IfElseBench32)->Args({elems, 99});
BENCHMARK(IfElseBench64)->Args({elems, 99});

BENCHMARK(IfElseBench32Contiguous)->Args({elems, 0});
BENCHMARK(IfElseBench64Contiguous)->Args({elems, 0});

BENCHMARK(IfElseBench32Contiguous)->Args({elems, 99});
BENCHMARK(IfElseBench64Contiguous)->Args({elems, 99});

BENCHMARK(CaseWhenBench64)->Args({elems, 0});
BENCHMARK(CaseWhenBench64)->Args({elems, 99});

BENCHMARK(CaseWhenBench64Contiguous)->Args({elems, 0});
BENCHMARK(CaseWhenBench64Contiguous)->Args({elems, 99});

}  // namespace compute
}  // namespace arrow
