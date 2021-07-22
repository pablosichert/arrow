#include <emscripten.h>

#include <iostream>

#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "gandiva/decimal_type_util.h"
#include "gandiva/engine.h"
#include "gandiva/filter.h"
#include "gandiva/llvm_types.h"
#include "gandiva/projector.h"
#include "gandiva/tree_expr_builder.h"

using namespace arrow;
using namespace gandiva;

// ArrayFromVector: construct an Array from vectors of C values

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::shared_ptr<DataType>& type,
                     const std::vector<bool>& is_valid, const std::vector<C_TYPE>& values,
                     std::shared_ptr<Array>* out) {
  auto type_id = TYPE::type_id;

  std::cout << "ArrayFromVector 1" << std::endl;
  std::unique_ptr<ArrayBuilder> builder_ptr;
  std::cout << "ArrayFromVector 2" << std::endl;
  MakeBuilder(default_memory_pool(), type, &builder_ptr).ok();
  std::cout << "ArrayFromVector 3" << std::endl;
  std::cout << reinterpret_cast<uintptr_t>(builder_ptr.get()) << std::endl;
  // Get the concrete builder class to access its Append() specializations
  auto& builder = dynamic_cast<typename TypeTraits<TYPE>::BuilderType&>(*builder_ptr);
  std::cout << "ArrayFromVector 4" << std::endl;

  for (size_t i = 0; i < values.size(); ++i) {
    std::cout << "ArrayFromVector 5, " << i << std::endl;
    if (is_valid[i]) {
      builder.Append(values[i]).ok();
    } else {
      builder.AppendNull().ok();
    }
  }
  builder.Finish(out).ok();
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::shared_ptr<DataType>& type,
                     const std::vector<C_TYPE>& values, std::shared_ptr<Array>* out) {
  auto type_id = TYPE::type_id;

  std::cout << "ArrayFromVector 1" << std::endl;
  std::unique_ptr<ArrayBuilder> builder_ptr;
  std::cout << "ArrayFromVector 2" << std::endl;
  MakeBuilder(default_memory_pool(), type, &builder_ptr).ok();
  std::cout << "ArrayFromVector 3" << std::endl;
  // Get the concrete builder class to access its Append() specializations
  auto& builder = dynamic_cast<typename TypeTraits<TYPE>::BuilderType&>(*builder_ptr);
  std::cout << "ArrayFromVector 4" << std::endl;

  for (size_t i = 0; i < values.size(); ++i) {
    std::cout << "ArrayFromVector 5, " << i << std::endl;
    builder.Append(values[i]).ok();
  }
  builder.Finish(out).ok();
}

// Overloads without a DataType argument, for parameterless types

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::vector<bool>& is_valid, const std::vector<C_TYPE>& values,
                     std::shared_ptr<Array>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  std::cout << "MakeArrowArray[0] 1" << std::endl;
  ArrayFromVector<TYPE, C_TYPE>(type, is_valid, values, out);
  std::cout << "MakeArrowArray[0] 2" << std::endl;
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::vector<C_TYPE>& values, std::shared_ptr<Array>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  std::cout << "MakeArrowArray[1] 1" << std::endl;
  ArrayFromVector<TYPE, C_TYPE>(type, values, out);
  std::cout << "MakeArrowArray[1] 2" << std::endl;
}

template <typename TYPE, typename C_TYPE>
static inline ArrayPtr MakeArrowArray(std::vector<C_TYPE> values,
                                      std::vector<bool> validity) {
  ArrayPtr out;
  std::cout << "MakeArrowArray[2] 1" << std::endl;
  ArrayFromVector<TYPE, C_TYPE>(validity, values, &out);
  std::cout << "MakeArrowArray[2] 2" << std::endl;
  return out;
}

template <typename TYPE, typename C_TYPE>
static inline ArrayPtr MakeArrowArray(std::vector<C_TYPE> values) {
  ArrayPtr out;
  std::cout << "MakeArrowArray[3] 1" << std::endl;
  ArrayFromVector<TYPE, C_TYPE>(values, &out);
  std::cout << "MakeArrowArray[3] 2" << std::endl;
  return out;
}

template <typename TYPE, typename C_TYPE>
static inline ArrayPtr MakeArrowArray(const std::shared_ptr<arrow::DataType>& type,
                                      std::vector<C_TYPE> values,
                                      std::vector<bool> validity) {
  ArrayPtr out;
  std::cout << "MakeArrowArray[4] 1" << std::endl;
  ArrayFromVector<TYPE, C_TYPE>(type, validity, values, &out);
  std::cout << "MakeArrowArray[4] 2" << std::endl;
  return out;
}

#define MakeArrowArrayBool MakeArrowArray<arrow::BooleanType, bool>
#define MakeArrowArrayInt8 MakeArrowArray<arrow::Int8Type, int8_t>
#define MakeArrowArrayInt16 MakeArrowArray<arrow::Int16Type, int16_t>
#define MakeArrowArrayInt32 MakeArrowArray<arrow::Int32Type, int32_t>
#define MakeArrowArrayInt64 MakeArrowArray<arrow::Int64Type, int64_t>
#define MakeArrowArrayUint8 MakeArrowArray<arrow::UInt8Type, uint8_t>
#define MakeArrowArrayUint16 MakeArrowArray<arrow::UInt16Type, uint16_t>
#define MakeArrowArrayUint32 MakeArrowArray<arrow::UInt32Type, uint32_t>
#define MakeArrowArrayUint64 MakeArrowArray<arrow::UInt64Type, uint64_t>
#define MakeArrowArrayFloat32 MakeArrowArray<arrow::FloatType, float>
#define MakeArrowArrayFloat64 MakeArrowArray<arrow::DoubleType, double>
#define MakeArrowArrayDate64 MakeArrowArray<arrow::Date64Type, int64_t>
#define MakeArrowArrayUtf8 MakeArrowArray<arrow::StringType, std::string>
#define MakeArrowArrayBinary MakeArrowArray<arrow::BinaryType, std::string>
#define MakeArrowArrayDecimal MakeArrowArray<arrow::Decimal128Type, arrow::Decimal128>

static inline std::shared_ptr<Configuration> TestConfiguration() {
  auto builder = ConfigurationBuilder();
  auto configuration = builder.DefaultConfiguration();
  configuration->set_optimize(false);
  configuration->target_host_cpu(false);
  return configuration;
}

/*
int main() {
  EM_ASM({ window.jitFunctions = []; });
  std::cout << "Hello!" << std::endl;

  auto configuration = TestConfiguration();

  auto pool_ = arrow::default_memory_pool();
  // schema for input fields
  auto field0 = field("f0", int32());
  auto field1 = field("f1", int32());
  auto schema = arrow::schema({field0, field1});

  std::cout << "1" << std::endl;

  // Build condition f0 + f1 < 10
  auto node_f0 = TreeExprBuilder::MakeField(field0);
  auto node_f1 = TreeExprBuilder::MakeField(field1);
  auto sum_func =
      TreeExprBuilder::MakeFunction("add", {node_f0, node_f1}, arrow::int32());
  // auto literal_10 = TreeExprBuilder::MakeLiteral((int32_t)10);
  // auto less_than_10 = TreeExprBuilder::MakeFunction("less_than", {sum_func,
  // literal_10},
  //                                                   arrow::boolean());
  // auto condition = TreeExprBuilder::MakeCondition(less_than_10);

  std::cout << "2" << std::endl;

  // std::shared_ptr<Filter> filter;
  // auto status = Filter::Make(schema, condition, configuration, &filter);
  // status.ok();

  std::shared_ptr<Projector> projector;
  auto status = Projector::Make(schema, sum_func, configuration, &projector);
  status.ok();

  std::cout << "3" << std::endl;

  // Create a row-batch with some sample data
  int num_records = 5;
  auto array0 = MakeArrowArrayInt32({1, 2, 3, 4, 6}, {true, true, true, false, true});
  std::cout << "3.1" << std::endl;
  auto array1 = MakeArrowArrayInt32({5, 9, 6, 17, 3}, {true, true, false, true, true});
  std::cout << "3.2" << std::endl;
  // expected output (indices for which condition matches)
  auto exp = MakeArrowArrayUint16({0, 4});
  std::cout << "3.3" << std::endl;

  std::cout << "4" << std::endl;

  // prepare input record batch
  auto in_batch = arrow::RecordBatch::Make(schema, num_records, {array0, array1});

  std::cout << "5" << std::endl;

  std::shared_ptr<SelectionVector> selection_vector;
  status = SelectionVector::MakeInt16(num_records, pool_, &selection_vector);
  status.ok();

  std::cout << "6" << std::endl;

  // Evaluate expression
  status = sum_func->Evaluate(*in_batch, selection_vector);
  status.ok();

  std::cout << "7" << std::endl;

  std::cout << selection_vector->ToArray()->ToString() << std::endl;

  std::cout << "8" << std::endl;

  return 0;
}
*/

// void BuildEngine(std::unique_ptr<Engine>& engine) {
//   auto status = Engine::Make(TestConfiguration(), &engine);

//   std::cout << "BuildEngine" << status << std::endl;
// }

// llvm::Function* BuildVecAdd(Engine* engine) {
//   auto types = engine->types();
//   llvm::IRBuilder<>* builder = engine->ir_builder();
//   llvm::LLVMContext* context = engine->context();

//   // Create fn prototype :
//   //   int64_t add_longs(int64_t *elements, int32_t nelements)
//   std::vector<llvm::Type*> arguments;
//   arguments.push_back(types->i64_ptr_type());
//   arguments.push_back(types->i32_type());
//   llvm::FunctionType* prototype =
//       llvm::FunctionType::get(types->i64_type(), arguments, false /*isVarArg*/);

//   // Create fn
//   std::string func_name = "add_longs";
//   engine->AddFunctionToCompile(func_name);
//   llvm::Function* fn = llvm::Function::Create(
//       prototype, llvm::GlobalValue::ExternalLinkage, func_name, engine->module());
//   fn != nullptr;

//   // Name the arguments
//   llvm::Function::arg_iterator args = fn->arg_begin();
//   llvm::Value* arg_elements = &*args;
//   arg_elements->setName("elements");
//   ++args;
//   llvm::Value* arg_nelements = &*args;
//   arg_nelements->setName("nelements");
//   ++args;

//   llvm::BasicBlock* loop_entry = llvm::BasicBlock::Create(*context, "entry", fn);
//   llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(*context, "loop", fn);
//   llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(*context, "exit", fn);

//   // Loop entry
//   builder->SetInsertPoint(loop_entry);
//   builder->CreateBr(loop_body);

//   // Loop body
//   builder->SetInsertPoint(loop_body);

//   llvm::PHINode* loop_var = builder->CreatePHI(types->i32_type(), 2, "loop_var");
//   llvm::PHINode* sum = builder->CreatePHI(types->i64_type(), 2, "sum");

//   loop_var->addIncoming(types->i32_constant(0), loop_entry);
//   sum->addIncoming(types->i64_constant(0), loop_entry);

//   // setup loop PHI
//   llvm::Value* loop_update =
//       builder->CreateAdd(loop_var, types->i32_constant(1), "loop_var+1");
//   loop_var->addIncoming(loop_update, loop_body);

//   // get the current value
//   llvm::Value* offset = builder->CreateGEP(arg_elements, loop_var, "offset");
//   llvm::Value* current_value = builder->CreateLoad(offset, "value");

//   // setup sum PHI
//   llvm::Value* sum_update = builder->CreateAdd(sum, current_value, "sum+ith");
//   sum->addIncoming(sum_update, loop_body);

//   // check loop_var
//   llvm::Value* loop_var_check =
//       builder->CreateICmpSLT(loop_update, arg_nelements, "loop_var < nrec");
//   builder->CreateCondBr(loop_var_check, loop_body, loop_exit);

//   // Loop exit
//   builder->SetInsertPoint(loop_exit);
//   builder->CreateRet(sum_update);
//   return fn;
// }

// int main() {
//   std::cout << "main 1" << std::endl;
//   std::unique_ptr<Engine> engine;
//   std::cout << "main 2" << std::endl;
//   BuildEngine(engine);
//   std::cout << "main 3" << std::endl;
//   BuildVecAdd(engine.get());
//   std::cout << "main 4" << std::endl;

//   llvm::Function* ir_func = BuildVecAdd(engine.get());
//   std::cout << "main 5" << std::endl;
//   engine->FinalizeModule().ok();
//   std::cout << "main 6" << std::endl;
//   auto add_func =
//   reinterpret_cast<add_vector_func_t>(engine->CompiledFunction(ir_func));

//   int64_t my_array[] = {1, 3, -5, 8, 10};
//   EXPECT_EQ(add_func(my_array, 5), 17);
// }

// std::vector<Decimal128> MakeDecimalVector(std::vector<std::string> values,
//                                           int32_t scale) {
//   std::vector<arrow::Decimal128> ret;
//   for (auto str : values) {
//     Decimal128 str_value;
//     int32_t str_precision;
//     int32_t str_scale;

//     DCHECK_OK(Decimal128::FromString(str, &str_value, &str_precision, &str_scale));

//     Decimal128 scaled_value;
//     if (str_scale == scale) {
//       scaled_value = str_value;
//     } else {
//       scaled_value = str_value.Rescale(str_scale, scale).ValueOrDie();
//     }
//     ret.push_back(scaled_value);
//   }
//   return ret;
// }

// int main() {
//   auto pool_ = arrow::default_memory_pool();
//   // schema for input fields
//   constexpr int32_t precision = 36;
//   constexpr int32_t scale = 18;
//   auto decimal_type = std::make_shared<arrow::Decimal128Type>(precision, scale);
//   auto field_a = field("a", decimal_type);
//   auto field_b = field("b", decimal_type);
//   auto schema = arrow::schema({field_a, field_b});

//   // build expressions
//   auto exprs = std::vector<ExpressionPtr>{
//       TreeExprBuilder::MakeExpression("equal", {field_a, field_b},
//                                       field("res_eq", boolean())),
//       TreeExprBuilder::MakeExpression("not_equal", {field_a, field_b},
//                                       field("res_ne", boolean())),
//       TreeExprBuilder::MakeExpression("less_than", {field_a, field_b},
//                                       field("res_lt", boolean())),
//       TreeExprBuilder::MakeExpression("less_than_or_equal_to", {field_a, field_b},
//                                       field("res_le", boolean())),
//       TreeExprBuilder::MakeExpression("greater_than", {field_a, field_b},
//                                       field("res_gt", boolean())),
//       TreeExprBuilder::MakeExpression("greater_than_or_equal_to", {field_a, field_b},
//                                       field("res_ge", boolean())),
//   };

//   // Build a projector for the expression.
//   std::shared_ptr<Projector> projector;
//   auto status = Projector::Make(schema, exprs, TestConfiguration(), &projector);
//   DCHECK_OK(status);

//   // Create a row-batch with some sample data
//   int num_records = 4;
//   auto array_a =
//       MakeArrowArrayDecimal(decimal_type, MakeDecimalVector({"1", "2", "3", "-4"},
//       scale),
//                             {true, true, true, true});
//   auto array_b =
//       MakeArrowArrayDecimal(decimal_type, MakeDecimalVector({"1", "3", "2", "-3"},
//       scale),
//                             {true, true, true, true});

//   // prepare input record batch
//   auto in_batch = arrow::RecordBatch::Make(schema, num_records, {array_a, array_b});

//   // Evaluate expression
//   arrow::ArrayVector outputs;
//   status = projector->Evaluate(*in_batch, pool_, &outputs);
//   DCHECK_OK(status);

//   for (auto& output : outputs) {
//     std::cout << "Output: " << output->ToString() << std::endl;
//   }

//   return 0;
// }

#define EXPECT_EQ(expected, actual) \
  if (&actual != &expected) std::cout

#define EXPECT_TRUE(expr) \
  if (!expr) std::cout

#define ARROW_EXPECT_OK(expr)                                           \
  do {                                                                  \
    auto _res = (expr);                                                 \
    ::arrow::Status _st = ::arrow::internal::GenericToStatus(_res);     \
    EXPECT_TRUE(_st.ok()) << "'" ARROW_STRINGIFY(expr) "' failed with " \
                          << _st.ToString();                            \
  } while (false)

using arrow::Decimal128;

#define EXPECT_DECIMAL_RESULT(op, x, y, expected, actual)                                \
  EXPECT_EQ(expected, actual) << op << " (" << (x).ToString() << "),(" << (y).ToString() \
                              << ")"                                                     \
                              << " expected : " << (expected).ToString()                 \
                              << " actual : " << (actual).ToString();

DecimalScalar128 decimal_literal(const char* value, int precision, int scale) {
  std::string value_string = std::string(value);
  return DecimalScalar128(value_string, precision, scale);
}

class TestDecimalOps {
 public:
  void SetUp() { pool_ = arrow::default_memory_pool(); }

  ArrayPtr MakeDecimalVector(const DecimalScalar128& in);

  void Verify(DecimalTypeUtil::Op, const std::string& function, const DecimalScalar128& x,
              const DecimalScalar128& y, const DecimalScalar128& expected);

  void AddAndVerify(const DecimalScalar128& x, const DecimalScalar128& y,
                    const DecimalScalar128& expected) {
    Verify(DecimalTypeUtil::kOpAdd, "add", x, y, expected);
  }

  void SubtractAndVerify(const DecimalScalar128& x, const DecimalScalar128& y,
                         const DecimalScalar128& expected) {
    Verify(DecimalTypeUtil::kOpSubtract, "subtract", x, y, expected);
  }

  void MultiplyAndVerify(const DecimalScalar128& x, const DecimalScalar128& y,
                         const DecimalScalar128& expected) {
    Verify(DecimalTypeUtil::kOpMultiply, "multiply", x, y, expected);
  }

  void DivideAndVerify(const DecimalScalar128& x, const DecimalScalar128& y,
                       const DecimalScalar128& expected) {
    Verify(DecimalTypeUtil::kOpDivide, "divide", x, y, expected);
  }

  void ModAndVerify(const DecimalScalar128& x, const DecimalScalar128& y,
                    const DecimalScalar128& expected) {
    Verify(DecimalTypeUtil::kOpMod, "mod", x, y, expected);
  }

  void Test();

 protected:
  arrow::MemoryPool* pool_;
};

ArrayPtr TestDecimalOps::MakeDecimalVector(const DecimalScalar128& in) {
  std::vector<arrow::Decimal128> ret;

  Decimal128 decimal_value = in.value();

  auto decimal_type = std::make_shared<arrow::Decimal128Type>(in.precision(), in.scale());
  return MakeArrowArrayDecimal(decimal_type, {decimal_value}, {true});
}

void TestDecimalOps::Verify(DecimalTypeUtil::Op op, const std::string& function,
                            const DecimalScalar128& x, const DecimalScalar128& y,
                            const DecimalScalar128& expected) {
  auto x_type = std::make_shared<arrow::Decimal128Type>(x.precision(), x.scale());
  auto y_type = std::make_shared<arrow::Decimal128Type>(y.precision(), y.scale());
  auto field_x = field("x", x_type);
  auto field_y = field("y", y_type);
  auto schema = arrow::schema({field_x, field_y});

  Decimal128TypePtr output_type;
  auto status = DecimalTypeUtil::GetResultType(op, {x_type, y_type}, &output_type);
  ARROW_EXPECT_OK(status);

  // output fields
  auto res = field("res", output_type);

  // build expression : x op y
  auto expr = TreeExprBuilder::MakeExpression(function, {field_x, field_y}, res);

  // Build a projector for the expression.
  std::shared_ptr<Projector> projector;
  status = Projector::Make(schema, {expr}, TestConfiguration(), &projector);
  ARROW_EXPECT_OK(status);

  // Create a row-batch with some sample data
  auto array_a = MakeDecimalVector(x);
  auto array_b = MakeDecimalVector(y);

  // prepare input record batch
  auto in_batch = arrow::RecordBatch::Make(schema, 1 /*num_records*/, {array_a, array_b});

  // Evaluate expression
  arrow::ArrayVector outputs;
  status = projector->Evaluate(*in_batch, pool_, &outputs);
  ARROW_EXPECT_OK(status);

  // Validate results
  auto out_array = dynamic_cast<arrow::Decimal128Array*>(outputs[0].get());
  const Decimal128 out_value(out_array->GetValue(0));

  auto dtype = dynamic_cast<arrow::Decimal128Type*>(out_array->type().get());
  std::string value_string = out_value.ToString(0);
  DecimalScalar128 actual{value_string, dtype->precision(), dtype->scale()};

  EXPECT_DECIMAL_RESULT(function, x, y, expected, actual);
}

void TestDecimalOps::Test() {
  std::cout << "TestDecimalOps, TestAdd" << std::endl;
  // fast-path
  AddAndVerify(decimal_literal("201", 30, 3),   // x
               decimal_literal("301", 30, 3),   // y
               decimal_literal("502", 31, 3));  // expected

  AddAndVerify(decimal_literal("201", 30, 3),    // x
               decimal_literal("301", 30, 2),    // y
               decimal_literal("3211", 32, 3));  // expected

  AddAndVerify(decimal_literal("201", 30, 3),    // x
               decimal_literal("301", 30, 4),    // y
               decimal_literal("2311", 32, 4));  // expected

  // max precision, but no overflow
  AddAndVerify(decimal_literal("201", 38, 3),   // x
               decimal_literal("301", 38, 3),   // y
               decimal_literal("502", 38, 3));  // expected

  AddAndVerify(decimal_literal("201", 38, 3),    // x
               decimal_literal("301", 38, 2),    // y
               decimal_literal("3211", 38, 3));  // expected

  AddAndVerify(decimal_literal("201", 38, 3),    // x
               decimal_literal("301", 38, 4),    // y
               decimal_literal("2311", 38, 4));  // expected

  AddAndVerify(decimal_literal("201", 38, 3),      // x
               decimal_literal("301", 38, 7),      // y
               decimal_literal("201030", 38, 6));  // expected

  AddAndVerify(decimal_literal("1201", 38, 3),   // x
               decimal_literal("1801", 38, 3),   // y
               decimal_literal("3002", 38, 3));  // carry-over from fractional

  // max precision
  AddAndVerify(decimal_literal("09999999999999999999999999999999000000", 38, 5),  // x
               decimal_literal("100", 38, 7),                                     // y
               decimal_literal("99999999999999999999999999999990000010", 38, 6));

  AddAndVerify(decimal_literal("-09999999999999999999999999999999000000", 38, 5),  // x
               decimal_literal("100", 38, 7),                                      // y
               decimal_literal("-99999999999999999999999999999989999990", 38, 6));

  AddAndVerify(decimal_literal("09999999999999999999999999999999000000", 38, 5),  // x
               decimal_literal("-100", 38, 7),                                    // y
               decimal_literal("99999999999999999999999999999989999990", 38, 6));

  AddAndVerify(decimal_literal("-09999999999999999999999999999999000000", 38, 5),  // x
               decimal_literal("-100", 38, 7),                                     // y
               decimal_literal("-99999999999999999999999999999990000010", 38, 6));

  AddAndVerify(decimal_literal("09999999999999999999999999999999999999", 38, 6),  // x
               decimal_literal("89999999999999999999999999999999999999", 38, 7),  // y
               decimal_literal("18999999999999999999999999999999999999", 38, 6));

  // Both -ve
  AddAndVerify(decimal_literal("-201", 30, 3),    // x
               decimal_literal("-301", 30, 2),    // y
               decimal_literal("-3211", 32, 3));  // expected

  AddAndVerify(decimal_literal("-201", 38, 3),    // x
               decimal_literal("-301", 38, 4),    // y
               decimal_literal("-2311", 38, 4));  // expected

  // Mix of +ve and -ve
  AddAndVerify(decimal_literal("-201", 30, 3),   // x
               decimal_literal("301", 30, 2),    // y
               decimal_literal("2809", 32, 3));  // expected

  AddAndVerify(decimal_literal("-201", 38, 3),    // x
               decimal_literal("301", 38, 4),     // y
               decimal_literal("-1709", 38, 4));  // expected

  AddAndVerify(decimal_literal("201", 38, 3),      // x
               decimal_literal("-301", 38, 7),     // y
               decimal_literal("200970", 38, 6));  // expected

  AddAndVerify(decimal_literal("-1901", 38, 4),  // x
               decimal_literal("1801", 38, 4),   // y
               decimal_literal("-100", 38, 4));  // expected

  AddAndVerify(decimal_literal("1801", 38, 4),   // x
               decimal_literal("-1901", 38, 4),  // y
               decimal_literal("-100", 38, 4));  // expected

  // rounding +ve
  AddAndVerify(decimal_literal("1000999", 38, 6),   // x
               decimal_literal("10000999", 38, 7),  // y
               decimal_literal("2001099", 38, 6));

  AddAndVerify(decimal_literal("1000999", 38, 6),   // x
               decimal_literal("10000995", 38, 7),  // y
               decimal_literal("2001099", 38, 6));

  AddAndVerify(decimal_literal("1000999", 38, 6),   // x
               decimal_literal("10000992", 38, 7),  // y
               decimal_literal("2001098", 38, 6));

  // rounding -ve
  AddAndVerify(decimal_literal("-1000999", 38, 6),   // x
               decimal_literal("-10000999", 38, 7),  // y
               decimal_literal("-2001099", 38, 6));

  AddAndVerify(decimal_literal("-1000999", 38, 6),   // x
               decimal_literal("-10000995", 38, 7),  // y
               decimal_literal("-2001099", 38, 6));

  AddAndVerify(decimal_literal("-1000999", 38, 6),   // x
               decimal_literal("-10000992", 38, 7),  // y
               decimal_literal("-2001098", 38, 6));

  // subtract is a wrapper over add. so, minimal tests are sufficient.
  std::cout << "TestDecimalOps, TestSubtract" << std::endl;
  // fast-path
  SubtractAndVerify(decimal_literal("201", 30, 3),    // x
                    decimal_literal("301", 30, 3),    // y
                    decimal_literal("-100", 31, 3));  // expected

  // max precision
  SubtractAndVerify(
      decimal_literal("09999999999999999999999999999999000000", 38, 5),  // x
      decimal_literal("100", 38, 7),                                     // y
      decimal_literal("99999999999999999999999999999989999990", 38, 6));

  // Mix of +ve and -ve
  SubtractAndVerify(decimal_literal("-201", 30, 3),    // x
                    decimal_literal("301", 30, 2),     // y
                    decimal_literal("-3211", 32, 3));  // expected

  // Lots of unit tests for multiply/divide/mod in decimal_ops_test.cc. So, keeping these
  // basic.
  std::cout << "TestDecimalOps, TestMultiply" << std::endl;
  // fast-path
  MultiplyAndVerify(decimal_literal("201", 10, 3),     // x
                    decimal_literal("301", 10, 2),     // y
                    decimal_literal("60501", 21, 5));  // expected

  // max precision
  MultiplyAndVerify(DecimalScalar128(std::string(35, '9'), 38, 20),  // x
                    DecimalScalar128(std::string(36, '9'), 38, 20),  // x
                    DecimalScalar128("9999999999999999999999999999999999890", 38, 6));

  std::cout << "TestDecimalOps, TestDivide" << std::endl;
  DivideAndVerify(decimal_literal("201", 10, 3),              // x
                  decimal_literal("301", 10, 2),              // y
                  decimal_literal("6677740863787", 23, 14));  // expected

  DivideAndVerify(DecimalScalar128(std::string(38, '9'), 38, 20),  // x
                  DecimalScalar128(std::string(35, '9'), 38, 20),  // x
                  DecimalScalar128("1000000000", 38, 6));

  std::cout << "TestDecimalOps, TestMod" << std::endl;
  ModAndVerify(decimal_literal("201", 20, 2),   // x
               decimal_literal("301", 20, 3),   // y
               decimal_literal("204", 20, 3));  // expected

  ModAndVerify(DecimalScalar128(std::string(38, '9'), 38, 20),  // x
               DecimalScalar128(std::string(35, '9'), 38, 21),  // x
               DecimalScalar128("9990", 38, 21));
}

EMSCRIPTEN_KEEPALIVE
extern "C" void add(int x, int y) {
  auto pool_ = arrow::default_memory_pool();
  auto field_x = field("x", int32());
  auto field_y = field("y", int32());
  auto schema = arrow::schema({field_x, field_y});

  // output fields
  auto res = field("res", int32());

  // build expression : x op y
  auto expr = TreeExprBuilder::MakeExpression("add", {field_x, field_y}, res);

  // Build a projector for the expression.
  std::shared_ptr<Projector> projector;
  auto status = Projector::Make(schema, {expr}, TestConfiguration(), &projector);
  ARROW_EXPECT_OK(status);

  // Create a row-batch with some sample data
  auto array_a = MakeArrowArrayInt32({x}, {true});
  auto array_b = MakeArrowArrayInt32({y}, {true});

  // prepare input record batch
  auto in_batch = arrow::RecordBatch::Make(schema, 1 /*num_records*/, {array_a, array_b});

  std::cout << "input: " << in_batch->ToString() << std::endl;

  // Evaluate expression
  arrow::ArrayVector outputs;

  status = projector->Evaluate(*in_batch, pool_, &outputs);

  for (auto& output : outputs) {
    std::cout << "output: " << output->ToString() << std::endl;
  }
}

int main() {
  EM_ASM({ window.jitFunctions = []; });
  // auto test = TestDecimalOps();
  // test.SetUp();
  // test.Test();

  add(1337, 42);

  return 0;
}