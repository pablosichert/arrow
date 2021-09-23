#include <emscripten.h>
#include <emscripten/bind.h>
#include <gandiva/selection_vector.h>

#include <iostream>
#include <memory>

#include "arrow/io/memory.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/memory_pool.h"
#include "arrow/table.h"
#include "arrow/type_fwd.h"
#include "gandiva/filter.h"
#include "gandiva/gandiva_aliases.h"
#include "gandiva/node.h"
#include "gandiva/projector.h"
#include "gandiva/tree_expr_builder.h"

using namespace gandiva;

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

class Reader {
 private:
  std::string buffer;
  std::shared_ptr<arrow::io::RandomAccessFile> file;

 public:
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;

  Reader(std::string buffer, std::shared_ptr<arrow::io::RandomAccessFile> file,
         std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader)
      : buffer(std::move(buffer)), file(file), reader(reader) {}
};

using ArrayPtr = std::shared_ptr<arrow::Array>;
using BufferPtr = std::shared_ptr<arrow::Buffer>;
using FieldPtr = std::shared_ptr<arrow::Field>;
using FilterPtr = std::shared_ptr<Filter>;
using ProjectorPtr = std::shared_ptr<Projector>;
using ReaderPtr = std::shared_ptr<Reader>;
using RecordBatchFileReaderPtr = std::shared_ptr<arrow::ipc::RecordBatchFileReader>;
using RecordBatchPtr = std::shared_ptr<arrow::RecordBatch>;
using SelectionVectorPtr = std::shared_ptr<SelectionVector>;

DataTypePtr type_null() { return arrow::null(); }

DataTypePtr type_boolean() { return arrow::boolean(); }

DataTypePtr type_int8() { return arrow::int8(); }

DataTypePtr type_int16() { return arrow::int16(); }

DataTypePtr type_int32() { return arrow::int32(); }

DataTypePtr type_int64() { return arrow::int64(); }

DataTypePtr type_uint8() { return arrow::uint8(); }

DataTypePtr type_uint16() { return arrow::uint16(); }

DataTypePtr type_uint32() { return arrow::uint32(); }

DataTypePtr type_uint64() { return arrow::uint64(); }

DataTypePtr type_float16() { return arrow::float16(); }

DataTypePtr type_float32() { return arrow::float32(); }

DataTypePtr type_float64() { return arrow::float64(); }

DataTypePtr type_utf8() { return arrow::utf8(); }

DataTypePtr type_large_utf8() { return arrow::large_utf8(); }

DataTypePtr type_binary() { return arrow::binary(); }

DataTypePtr type_large_binary() { return arrow::large_binary(); }

DataTypePtr type_date32() { return arrow::date32(); }

DataTypePtr type_date64() { return arrow::date64(); }

NodePtr make_literal_bool(bool value) { return TreeExprBuilder::MakeLiteral(value); }

NodePtr make_literal_uint8_t(uint8_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_uint16_t(uint16_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_uint32_t(uint32_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_uint64_t(uint64_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_int8_t(int8_t value) { return TreeExprBuilder::MakeLiteral(value); }

NodePtr make_literal_int16_t(int16_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_int32_t(int32_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_int64_t(int64_t value) {
  return TreeExprBuilder::MakeLiteral(value);
}

NodePtr make_literal_float(float value) { return TreeExprBuilder::MakeLiteral(value); }

NodePtr make_literal_double(double value) { return TreeExprBuilder::MakeLiteral(value); }

NodePtr make_string_literal(const std::string& value) {
  return TreeExprBuilder::MakeStringLiteral(value);
}

NodePtr make_binary_literal(const std::string& value) {
  return TreeExprBuilder::MakeBinaryLiteral(value);
}

NodePtr make_decimal_literal(const DecimalScalar128& value) {
  return TreeExprBuilder::MakeDecimalLiteral(value);
}

NodePtr make_null(DataTypePtr data_type) { return TreeExprBuilder::MakeNull(data_type); }

NodePtr make_field(FieldPtr field) { return TreeExprBuilder::MakeField(field); }

NodePtr make_function(const std::string& name, const NodeVector& params,
                      DataTypePtr return_type) {
  return TreeExprBuilder::MakeFunction(name, params, return_type);
}

NodePtr make_if(NodePtr condition, NodePtr then_node, NodePtr else_node,
                DataTypePtr result_type) {
  return TreeExprBuilder::MakeIf(condition, then_node, else_node, result_type);
}

NodePtr make_and(const NodeVector& children) {
  return TreeExprBuilder::MakeAnd(children);
}

NodePtr make_or(const NodeVector& children) { return TreeExprBuilder::MakeOr(children); }

ExpressionPtr make_expression(NodePtr root_node, FieldPtr result_field) {
  return TreeExprBuilder::MakeExpression(root_node, result_field);
}

ExpressionPtr make_function_expression(const std::string& function,
                                       const FieldVector& in_fields, FieldPtr out_field) {
  return TreeExprBuilder::MakeExpression(function, in_fields, out_field);
}

ConditionPtr make_condition(NodePtr root_node) {
  return TreeExprBuilder::MakeCondition(root_node);
}

ConditionPtr make_function_condition(const std::string& function,
                                     const FieldVector& in_fields) {
  return TreeExprBuilder::MakeCondition(function, in_fields);
}

NodePtr make_in_expression_int32(NodePtr node,
                                 const std::unordered_set<int32_t>& constants) {
  return TreeExprBuilder::MakeInExpressionInt32(node, constants);
}

NodePtr make_in_expression_int64(NodePtr node,
                                 const std::unordered_set<int64_t>& constants) {
  return TreeExprBuilder::MakeInExpressionInt64(node, constants);
}

NodePtr make_in_expression_decimal(
    NodePtr node, std::unordered_set<gandiva::DecimalScalar128>& constants) {
  return TreeExprBuilder::MakeInExpressionDecimal(node, constants);
}

NodePtr make_in_expression_string(NodePtr node,
                                  const std::unordered_set<std::string>& constants) {
  return TreeExprBuilder::MakeInExpressionString(node, constants);
}

NodePtr make_in_expression_binary(NodePtr node,
                                  const std::unordered_set<std::string>& constants) {
  return TreeExprBuilder::MakeInExpressionBinary(node, constants);
}

NodePtr make_in_expression_float(NodePtr node,
                                 const std::unordered_set<float>& constants) {
  return TreeExprBuilder::MakeInExpressionFloat(node, constants);
}

NodePtr make_in_expression_double(NodePtr node,
                                  const std::unordered_set<double>& constants) {
  return TreeExprBuilder::MakeInExpressionDouble(node, constants);
}

NodePtr make_in_expression_date32(NodePtr node,
                                  const std::unordered_set<int32_t>& constants) {
  return TreeExprBuilder::MakeInExpressionDate32(node, constants);
}

NodePtr make_in_expression_date64(NodePtr node,
                                  const std::unordered_set<int64_t>& constants) {
  return TreeExprBuilder::MakeInExpressionDate64(node, constants);
}

NodePtr make_in_expression_time32(NodePtr node,
                                  const std::unordered_set<int32_t>& constants) {
  return TreeExprBuilder::MakeInExpressionTime32(node, constants);
}

NodePtr make_in_expression_time64(NodePtr node,
                                  const std::unordered_set<int64_t>& constants) {
  return TreeExprBuilder::MakeInExpressionTime64(node, constants);
}

NodePtr make_in_expression_timestamp(NodePtr node,
                                     const std::unordered_set<int64_t>& constants) {
  return TreeExprBuilder::MakeInExpressionTimeStamp(node, constants);
}

std::shared_ptr<Configuration> make_configuration() {
  auto builder = ConfigurationBuilder();
  auto configuration = builder.DefaultConfiguration();
  configuration->set_optimize(false);
  configuration->target_host_cpu(false);
  return configuration;
}

ProjectorPtr make_projector(SchemaPtr schema, const ExpressionVector& exprs) {
  auto configuration = make_configuration();
  ProjectorPtr projector;
  ARROW_EXPECT_OK(Projector::Make(schema, exprs, configuration, &projector));
  return projector;
}

ProjectorPtr make_projector_with_selection_vector_mode(
    SchemaPtr schema, const ExpressionVector& exprs,
    SelectionVector::Mode selection_vector_mode) {
  auto configuration = make_configuration();
  ProjectorPtr projector;
  ARROW_EXPECT_OK(
      Projector::Make(schema, exprs, selection_vector_mode, configuration, &projector));
  return projector;
}

FilterPtr make_filter(SchemaPtr schema, ConditionPtr condition) {
  auto configuration = make_configuration();
  FilterPtr filter;
  ARROW_EXPECT_OK(Filter::Make(schema, condition, configuration, &filter));
  return filter;
}

SchemaPtr make_schema(FieldVector fields) {
  return std::make_shared<arrow::Schema>(fields);
}

arrow::ArrayVector projector_evaluate(ProjectorPtr projector, RecordBatchPtr batch) {
  arrow::ArrayVector outputs;
  ARROW_EXPECT_OK(projector->Evaluate(*batch, arrow::default_memory_pool(), &outputs));
  return outputs;
}

arrow::ArrayVector projector_evaluate_with_selection_vector(
    ProjectorPtr projector, SelectionVectorPtr selection_vector, RecordBatchPtr batch) {
  arrow::ArrayVector outputs;
  ARROW_EXPECT_OK(projector->Evaluate(*batch, &*selection_vector,
                                      arrow::default_memory_pool(), &outputs));
  return outputs;
}

void filter_evaluate(FilterPtr filter, SelectionVectorPtr selection_vector,
                     RecordBatchPtr batch) {
  ARROW_EXPECT_OK(filter->Evaluate(*batch, selection_vector));
}

emscripten::val buffer_view(BufferPtr buffer) {
  return emscripten::val(emscripten::typed_memory_view(buffer->size(), buffer->data()));
}

ReaderPtr make_reader(std::string buffer) {
  auto data = reinterpret_cast<const uint8_t*>(buffer.data());
  int64_t size = buffer.length();
  auto file = std::make_shared<arrow::io::BufferReader>(data, size);
  auto reader = arrow::ipc::RecordBatchFileReader::Open(file.get()).ValueOrDie();
  return std::make_shared<Reader>(std::move(buffer), file, reader);
}

int schema_num_fields(SchemaPtr schema) { return schema->num_fields(); }

FieldPtr schema_field(SchemaPtr schema, int index) { return schema->field(index); }

FieldPtr schema_field_by_name(SchemaPtr schema, const std::string& name) {
  return schema->GetFieldByName(name);
}

FieldVector schema_fields(SchemaPtr schema) { return schema->fields(); }

SchemaPtr reader_schema(ReaderPtr reader) { return reader->reader->schema(); }

int reader_num_record_batches(ReaderPtr reader) {
  return reader->reader->num_record_batches();
}

RecordBatchPtr reader_read_record_batch(ReaderPtr reader, int index) {
  return reader->reader->ReadRecordBatch(index).ValueOrDie();
}

int batch_num_columns(RecordBatchPtr batch) { return batch->num_columns(); }

int batch_num_rows(RecordBatchPtr batch) { return batch->num_rows(); }

SelectionVectorPtr selection_vector_make_int16(int max_slots) {
  std::shared_ptr<SelectionVector> selection_vector;
  ARROW_EXPECT_OK(SelectionVector::MakeInt16(max_slots, arrow::default_memory_pool(),
                                             &selection_vector));
  return selection_vector;
}

SelectionVectorPtr selection_vector_make_int32(int max_slots) {
  std::shared_ptr<SelectionVector> selection_vector;
  ARROW_EXPECT_OK(SelectionVector::MakeInt32(max_slots, arrow::default_memory_pool(),
                                             &selection_vector));
  return selection_vector;
}

SelectionVectorPtr selection_vector_make_int64(int max_slots) {
  std::shared_ptr<SelectionVector> selection_vector;
  ARROW_EXPECT_OK(SelectionVector::MakeInt64(max_slots, arrow::default_memory_pool(),
                                             &selection_vector));
  return selection_vector;
}

BufferPtr selection_vector_to_buffer(SelectionVectorPtr selection_vector,
                                     SchemaPtr schema) {
  auto vector = selection_vector->ToArray();
  auto array_vector = arrow::ArrayVector{vector};
  auto table = arrow::Table::Make(schema, array_vector);
  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeFileWriter(sink, schema).ValueOrDie();
  ARROW_EXPECT_OK(writer->WriteTable(*table));
  ARROW_EXPECT_OK(writer->Close());
  return sink->Finish().ValueOrDie();
}

BufferPtr array_vector_to_buffer(const arrow::ArrayVector& array_vector,
                                 SchemaPtr schema) {
  auto table = arrow::Table::Make(schema, array_vector);
  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeFileWriter(sink, schema).ValueOrDie();
  ARROW_EXPECT_OK(writer->WriteTable(*table));
  ARROW_EXPECT_OK(writer->Close());
  return sink->Finish().ValueOrDie();
}

EMSCRIPTEN_BINDINGS() {
  using namespace emscripten;

  class_<arrow::Array>("Array").smart_ptr<ArrayPtr>("ArrayPtr");
  class_<arrow::Buffer>("Buffer").smart_ptr<BufferPtr>("BufferPtr");
  class_<arrow::DataType>("DataType").smart_ptr<DataTypePtr>("DataTypePtr");
  class_<arrow::Field>("Field").smart_ptr<FieldPtr>("FieldPtr");
  class_<arrow::RecordBatch>("RecordBatch").smart_ptr<RecordBatchPtr>("RecordBatchPtr");
  class_<arrow::Schema>("Schema").smart_ptr<SchemaPtr>("SchemaPtr");
  class_<Condition>("Condition").smart_ptr<ConditionPtr>("ConditionPtr");
  class_<Expression>("Expression").smart_ptr<ExpressionPtr>("ExpressionPtr");
  class_<Filter>("Filter").smart_ptr<FilterPtr>("FilterPtr");
  class_<Node>("Node").smart_ptr<NodePtr>("NodePtr");
  class_<Projector>("Projector").smart_ptr<ProjectorPtr>("ProjectorPtr");
  class_<Reader>("Reader").smart_ptr<ReaderPtr>("ReaderPtr");
  class_<SelectionVector>("SelectionVector")
      .smart_ptr<SelectionVectorPtr>("SelectionVectorPtr");

  register_vector<ArrayPtr>("ArrayVector");
  register_vector<ExpressionPtr>("ExpressionVector");
  register_vector<FieldPtr>("FieldVector");
  register_vector<NodePtr>("NodeVector");

  enum_<SelectionVector::Mode>("SelectionVectorMode")
      .value("NONE", SelectionVector::MODE_NONE)
      .value("UINT16", SelectionVector::MODE_UINT16)
      .value("UINT32", SelectionVector::MODE_UINT32)
      .value("UINT64", SelectionVector::MODE_UINT64)
      .value("MAX", SelectionVector::MODE_MAX);

  function("arrayVectorToBuffer", &array_vector_to_buffer);
  function("batchNumColumns", &batch_num_columns);
  function("batchNumRows", &batch_num_rows);
  function("bufferView", &buffer_view);
  function("filterEvaluate", &filter_evaluate);
  function("makeAnd", &make_and);
  function("makeBinaryLiteral", &make_binary_literal);
  function("makeCondition", &make_condition);
  function("makeDecimalLiteral", &make_decimal_literal);
  function("makeExpression", &make_expression);
  function("makeField", &make_field);
  function("makeFilter", &make_filter);
  function("makeFunction", &make_function);
  function("makeFunctionCondition", &make_function_condition);
  function("makeFunctionExpression", &make_function_expression);
  function("makeIf", &make_if);
  function("makeInExpressionBinary", &make_in_expression_binary);
  function("makeInExpressionDate32", &make_in_expression_date32);
  function("makeInExpressionDate64", &make_in_expression_date64);
  function("makeInExpressionDecimal", &make_in_expression_decimal);
  function("makeInExpressionDouble", &make_in_expression_double);
  function("makeInExpressionFloat", &make_in_expression_float);
  function("makeInExpressionInt32", &make_in_expression_int32);
  function("makeInExpressionInt64", &make_in_expression_int64);
  function("makeInExpressionString", &make_in_expression_string);
  function("makeInExpressionTime32", &make_in_expression_time32);
  function("makeInExpressionTime64", &make_in_expression_time64);
  function("makeInExpressionTimestamp", &make_in_expression_timestamp);
  function("makeLiteralBool", &make_literal_bool);
  function("makeLiteralDouble", &make_literal_double);
  function("makeLiteralFloat", &make_literal_float);
  function("makeLiteralInt16", &make_literal_int16_t);
  function("makeLiteralInt32", &make_literal_int32_t);
  function("makeLiteralInt64", &make_literal_int64_t);
  function("makeLiteralInt8", &make_literal_int8_t);
  function("makeLiteralUInt16", &make_literal_uint16_t);
  function("makeLiteralUInt32", &make_literal_uint32_t);
  function("makeLiteralUInt64", &make_literal_uint64_t);
  function("makeLiteralUInt8", &make_literal_uint8_t);
  function("makeNull", &make_null);
  function("makeOr", &make_or);
  function("makeProjector", &make_projector);
  function("makeProjectorWithSelectionVectorMode",
           &make_projector_with_selection_vector_mode);
  function("makeReader", &make_reader);
  function("makeSchema", &make_schema);
  function("makeStringLiteral", &make_string_literal);
  function("projectorEvaluate", &projector_evaluate);
  function("projectorEvaluateWithSelectionVector",
           &projector_evaluate_with_selection_vector);
  function("readerNumRecordBatches", &reader_num_record_batches);
  function("readerReadRecordBatch", &reader_read_record_batch);
  function("readerSchema", &reader_schema);
  function("schemaField", &schema_field);
  function("schemaFieldByName", &schema_field_by_name);
  function("schemaFields", &schema_fields);
  function("schemaNumFields", &schema_num_fields);
  function("selectionVectorMakeInt16", &selection_vector_make_int16);
  function("selectionVectorMakeInt32", &selection_vector_make_int32);
  function("selectionVectorMakeInt64", &selection_vector_make_int64);
  function("selectionVectorToBuffer", &selection_vector_to_buffer);
  function("typeBinary", &type_binary);
  function("typeBoolean", &type_boolean);
  function("typeDate32", &type_date32);
  function("typeDate64", &type_date64);
  function("typeFloat16", &type_float16);
  function("typeFloat32", &type_float32);
  function("typeFloat64", &type_float64);
  function("typeInt16", &type_int16);
  function("typeInt32", &type_int32);
  function("typeInt64", &type_int64);
  function("typeInt8", &type_int8);
  function("typeLargeBinary", &type_large_binary);
  function("typeLargeUTF8", &type_large_utf8);
  function("typeNull", &type_null);
  function("typeUInt16", &type_uint16);
  function("typeUInt32", &type_uint32);
  function("typeUInt64", &type_uint64);
  function("typeUInt8", &type_uint8);
  function("typeUTF8", &type_utf8);
}
