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

// Eager evaluation convenience APIs for invoking common functions, including
// necessary memory allocations

#pragma once

#include <string>
#include <utility>

#include "arrow/compute/exec.h"  // IWYU pragma: keep
#include "arrow/compute/function.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

/// \addtogroup compute-concrete-options
///
/// @{

class ARROW_EXPORT ArithmeticOptions : public FunctionOptions {
 public:
  explicit ArithmeticOptions(bool check_overflow = false);
  constexpr static char const kTypeName[] = "ArithmeticOptions";
  bool check_overflow;
};

class ARROW_EXPORT ElementWiseAggregateOptions : public FunctionOptions {
 public:
  explicit ElementWiseAggregateOptions(bool skip_nulls = true);
  constexpr static char const kTypeName[] = "ElementWiseAggregateOptions";
  static ElementWiseAggregateOptions Defaults() { return ElementWiseAggregateOptions{}; }

  bool skip_nulls;
};

/// Options for var_args_join.
class ARROW_EXPORT JoinOptions : public FunctionOptions {
 public:
  /// How to handle null values. (A null separator always results in a null output.)
  enum NullHandlingBehavior {
    /// A null in any input results in a null in the output.
    EMIT_NULL,
    /// Nulls in inputs are skipped.
    SKIP,
    /// Nulls in inputs are replaced with the replacement string.
    REPLACE,
  };
  explicit JoinOptions(NullHandlingBehavior null_handling = EMIT_NULL,
                       std::string null_replacement = "");
  constexpr static char const kTypeName[] = "JoinOptions";
  static JoinOptions Defaults() { return JoinOptions(); }
  NullHandlingBehavior null_handling;
  std::string null_replacement;
};

class ARROW_EXPORT MatchSubstringOptions : public FunctionOptions {
 public:
  explicit MatchSubstringOptions(std::string pattern, bool ignore_case = false);
  MatchSubstringOptions();
  constexpr static char const kTypeName[] = "MatchSubstringOptions";

  /// The exact substring (or regex, depending on kernel) to look for inside input values.
  std::string pattern;
  /// Whether to perform a case-insensitive match.
  bool ignore_case = false;
};

class ARROW_EXPORT SplitOptions : public FunctionOptions {
 public:
  explicit SplitOptions(int64_t max_splits = -1, bool reverse = false);
  constexpr static char const kTypeName[] = "SplitOptions";

  /// Maximum number of splits allowed, or unlimited when -1
  int64_t max_splits;
  /// Start splitting from the end of the string (only relevant when max_splits != -1)
  bool reverse;
};

class ARROW_EXPORT SplitPatternOptions : public FunctionOptions {
 public:
  explicit SplitPatternOptions(std::string pattern, int64_t max_splits = -1,
                               bool reverse = false);
  SplitPatternOptions();
  constexpr static char const kTypeName[] = "SplitPatternOptions";

  /// The exact substring to split on.
  std::string pattern;
  /// Maximum number of splits allowed, or unlimited when -1
  int64_t max_splits;
  /// Start splitting from the end of the string (only relevant when max_splits != -1)
  bool reverse;
};

class ARROW_EXPORT ReplaceSliceOptions : public FunctionOptions {
 public:
  explicit ReplaceSliceOptions(int64_t start, int64_t stop, std::string replacement);
  ReplaceSliceOptions();
  constexpr static char const kTypeName[] = "ReplaceSliceOptions";

  /// Index to start slicing at
  int64_t start;
  /// Index to stop slicing at
  int64_t stop;
  /// String to replace the slice with
  std::string replacement;
};

class ARROW_EXPORT ReplaceSubstringOptions : public FunctionOptions {
 public:
  explicit ReplaceSubstringOptions(std::string pattern, std::string replacement,
                                   int64_t max_replacements = -1);
  ReplaceSubstringOptions();
  constexpr static char const kTypeName[] = "ReplaceSubstringOptions";

  /// Pattern to match, literal, or regular expression depending on which kernel is used
  std::string pattern;
  /// String to replace the pattern with
  std::string replacement;
  /// Max number of substrings to replace (-1 means unbounded)
  int64_t max_replacements;
};

class ARROW_EXPORT ExtractRegexOptions : public FunctionOptions {
 public:
  explicit ExtractRegexOptions(std::string pattern);
  ExtractRegexOptions();
  constexpr static char const kTypeName[] = "ExtractRegexOptions";

  /// Regular expression with named capture fields
  std::string pattern;
};

/// Options for IsIn and IndexIn functions
class ARROW_EXPORT SetLookupOptions : public FunctionOptions {
 public:
  explicit SetLookupOptions(Datum value_set, bool skip_nulls = false);
  SetLookupOptions();
  constexpr static char const kTypeName[] = "SetLookupOptions";

  /// The set of values to look up input values into.
  Datum value_set;
  /// Whether nulls in `value_set` count for lookup.
  ///
  /// If true, any null in `value_set` is ignored and nulls in the input
  /// produce null (IndexIn) or false (IsIn) values in the output.
  /// If false, any null in `value_set` is successfully matched in
  /// the input.
  bool skip_nulls;
};

class ARROW_EXPORT StrptimeOptions : public FunctionOptions {
 public:
  explicit StrptimeOptions(std::string format, TimeUnit::type unit);
  StrptimeOptions();
  constexpr static char const kTypeName[] = "StrptimeOptions";

  std::string format;
  TimeUnit::type unit;
};

class ARROW_EXPORT PadOptions : public FunctionOptions {
 public:
  explicit PadOptions(int64_t width, std::string padding = " ");
  PadOptions();
  constexpr static char const kTypeName[] = "PadOptions";

  /// The desired string length.
  int64_t width;
  /// What to pad the string with. Should be one codepoint (Unicode)/byte (ASCII).
  std::string padding;
};

class ARROW_EXPORT TrimOptions : public FunctionOptions {
 public:
  explicit TrimOptions(std::string characters);
  TrimOptions();
  constexpr static char const kTypeName[] = "TrimOptions";

  /// The individual characters that can be trimmed from the string.
  std::string characters;
};

class ARROW_EXPORT SliceOptions : public FunctionOptions {
 public:
  explicit SliceOptions(int64_t start, int64_t stop = std::numeric_limits<int64_t>::max(),
                        int64_t step = 1);
  SliceOptions();
  constexpr static char const kTypeName[] = "SliceOptions";
  int64_t start, stop, step;
};

enum CompareOperator : int8_t {
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
};

class ARROW_EXPORT CompareOptions : public FunctionOptions {
 public:
  explicit CompareOptions(CompareOperator op);
  CompareOptions();
  constexpr static char const kTypeName[] = "CompareOptions";
  enum CompareOperator op;
};

class ARROW_EXPORT ProjectOptions : public FunctionOptions {
 public:
  ProjectOptions(std::vector<std::string> n, std::vector<bool> r,
                 std::vector<std::shared_ptr<const KeyValueMetadata>> m);
  explicit ProjectOptions(std::vector<std::string> n);
  ProjectOptions();
  constexpr static char const kTypeName[] = "ProjectOptions";

  /// Names for wrapped columns
  std::vector<std::string> field_names;

  /// Nullability bits for wrapped columns
  std::vector<bool> field_nullability;

  /// Metadata attached to wrapped columns
  std::vector<std::shared_ptr<const KeyValueMetadata>> field_metadata;
};

struct ARROW_EXPORT DayOfWeekOptions : public FunctionOptions {
 public:
  explicit DayOfWeekOptions(bool one_based_numbering = false, uint32_t week_start = 1);
  constexpr static char const kTypeName[] = "DayOfWeekOptions";
  static DayOfWeekOptions Defaults() { return DayOfWeekOptions{}; }

  /// Number days from 1 if true and from 0 if false
  bool one_based_numbering;
  /// What day does the week start with (Monday=1, Sunday=7)
  uint32_t week_start;
};

/// @}

/// \brief Get the absolute value of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value transformed
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise absolute value
ARROW_EXPORT
Result<Datum> AbsoluteValue(const Datum& arg,
                            ArithmeticOptions options = ArithmeticOptions(),
                            ExecContext* ctx = NULLPTR);

/// \brief Add two values together. Array values must be the same length. If
/// either addend is null the result will be null.
///
/// \param[in] left the first addend
/// \param[in] right the second addend
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise sum
ARROW_EXPORT
Result<Datum> Add(const Datum& left, const Datum& right,
                  ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Subtract two values. Array values must be the same length. If the
/// minuend or subtrahend is null the result will be null.
///
/// \param[in] left the value subtracted from (minuend)
/// \param[in] right the value by which the minuend is reduced (subtrahend)
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise difference
ARROW_EXPORT
Result<Datum> Subtract(const Datum& left, const Datum& right,
                       ArithmeticOptions options = ArithmeticOptions(),
                       ExecContext* ctx = NULLPTR);

/// \brief Multiply two values. Array values must be the same length. If either
/// factor is null the result will be null.
///
/// \param[in] left the first factor
/// \param[in] right the second factor
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise product
ARROW_EXPORT
Result<Datum> Multiply(const Datum& left, const Datum& right,
                       ArithmeticOptions options = ArithmeticOptions(),
                       ExecContext* ctx = NULLPTR);

/// \brief Divide two values. Array values must be the same length. If either
/// argument is null the result will be null. For integer types, if there is
/// a zero divisor, an error will be raised.
///
/// \param[in] left the dividend
/// \param[in] right the divisor
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise quotient
ARROW_EXPORT
Result<Datum> Divide(const Datum& left, const Datum& right,
                     ArithmeticOptions options = ArithmeticOptions(),
                     ExecContext* ctx = NULLPTR);

/// \brief Negate values.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value negated
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise negation
ARROW_EXPORT
Result<Datum> Negate(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                     ExecContext* ctx = NULLPTR);

/// \brief Raise the values of base array to the power of the exponent array values.
/// Array values must be the same length. If either base or exponent is null the result
/// will be null.
///
/// \param[in] left the base
/// \param[in] right the exponent
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise base value raised to the power of exponent
ARROW_EXPORT
Result<Datum> Power(const Datum& left, const Datum& right,
                    ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Left shift the left array by the right array. Array values must be the
/// same length. If either operand is null, the result will be null.
///
/// \param[in] left the value to shift
/// \param[in] right the value to shift by
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise left value shifted left by the right value
ARROW_EXPORT
Result<Datum> ShiftLeft(const Datum& left, const Datum& right,
                        ArithmeticOptions options = ArithmeticOptions(),
                        ExecContext* ctx = NULLPTR);

/// \brief Right shift the left array by the right array. Array values must be the
/// same length. If either operand is null, the result will be null. Performs a
/// logical shift for unsigned values, and an arithmetic shift for signed values.
///
/// \param[in] left the value to shift
/// \param[in] right the value to shift by
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise left value shifted right by the right value
ARROW_EXPORT
Result<Datum> ShiftRight(const Datum& left, const Datum& right,
                         ArithmeticOptions options = ArithmeticOptions(),
                         ExecContext* ctx = NULLPTR);

/// \brief Compute the sine of the array values.
/// \param[in] arg The values to compute the sine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise sine of the values
ARROW_EXPORT
Result<Datum> Sin(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the cosine of the array values.
/// \param[in] arg The values to compute the cosine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise cosine of the values
ARROW_EXPORT
Result<Datum> Cos(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse sine (arcsine) of the array values.
/// \param[in] arg The values to compute the inverse sine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse sine of the values
ARROW_EXPORT
Result<Datum> Asin(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse cosine (arccosine) of the array values.
/// \param[in] arg The values to compute the inverse cosine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse cosine of the values
ARROW_EXPORT
Result<Datum> Acos(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Compute the tangent of the array values.
/// \param[in] arg The values to compute the tangent for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise tangent of the values
ARROW_EXPORT
Result<Datum> Tan(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse tangent (arctangent) of the array values.
/// \param[in] arg The values to compute the inverse tangent for.
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse tangent of the values
ARROW_EXPORT
Result<Datum> Atan(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse tangent (arctangent) of y/x, using the
/// argument signs to determine the correct quadrant.
/// \param[in] y The y-values to compute the inverse tangent for.
/// \param[in] x The x-values to compute the inverse tangent for.
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse tangent of the values
ARROW_EXPORT
Result<Datum> Atan2(const Datum& y, const Datum& x, ExecContext* ctx = NULLPTR);

/// \brief Get the natural log of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise natural log
ARROW_EXPORT
Result<Datum> Ln(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                 ExecContext* ctx = NULLPTR);

/// \brief Get the log base 10 of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise log base 10
ARROW_EXPORT
Result<Datum> Log10(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Get the log base 2 of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise log base 2
ARROW_EXPORT
Result<Datum> Log2(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Get the natural log of (1 + value).
///
/// If argument is null the result will be null.
/// This function may be more accurate than Log(1 + value) for values close to zero.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise natural log
ARROW_EXPORT
Result<Datum> Log1p(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Find the element-wise maximum of any number of arrays or scalars.
/// Array values must be the same length.
///
/// \param[in] args arrays or scalars to operate on.
/// \param[in] options options for handling nulls, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise maximum
ARROW_EXPORT
Result<Datum> MaxElementWise(
    const std::vector<Datum>& args,
    ElementWiseAggregateOptions options = ElementWiseAggregateOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Find the element-wise minimum of any number of arrays or scalars.
/// Array values must be the same length.
///
/// \param[in] args arrays or scalars to operate on.
/// \param[in] options options for handling nulls, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise minimum
ARROW_EXPORT
Result<Datum> MinElementWise(
    const std::vector<Datum>& args,
    ElementWiseAggregateOptions options = ElementWiseAggregateOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Compare a numeric array with a scalar.
///
/// \param[in] left datum to compare, must be an Array
/// \param[in] right datum to compare, must be a Scalar of the same type than
///            left Datum.
/// \param[in] options compare options
/// \param[in] ctx the function execution context, optional
/// \return resulting datum
///
/// Note on floating point arrays, this uses ieee-754 compare semantics.
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Compare(const Datum& left, const Datum& right, CompareOptions options,
                      ExecContext* ctx = NULLPTR);

/// \brief Invert the values of a boolean datum
/// \param[in] value datum to invert
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Invert(const Datum& value, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND of two boolean datums which always propagates nulls
/// (null and false is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> And(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND of two boolean datums with a Kleene truth table
/// (null and false is false).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneAnd(const Datum& left, const Datum& right,
                        ExecContext* ctx = NULLPTR);

/// \brief Element-wise OR of two boolean datums which always propagates nulls
/// (null and true is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Or(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise OR of two boolean datums with a Kleene truth table
/// (null or true is true).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneOr(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise XOR of two boolean datums
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Xor(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND NOT of two boolean datums which always propagates nulls
/// (null and not true is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> AndNot(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND NOT of two boolean datums with a Kleene truth table
/// (false and not null is false, null and not true is false).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneAndNot(const Datum& left, const Datum& right,
                           ExecContext* ctx = NULLPTR);

/// \brief IsIn returns true for each element of `values` that is contained in
/// `value_set`
///
/// Behaviour of nulls is governed by SetLookupOptions::skip_nulls.
///
/// \param[in] values array-like input to look up in value_set
/// \param[in] options SetLookupOptions
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsIn(const Datum& values, const SetLookupOptions& options,
                   ExecContext* ctx = NULLPTR);
ARROW_EXPORT
Result<Datum> IsIn(const Datum& values, const Datum& value_set,
                   ExecContext* ctx = NULLPTR);

/// \brief IndexIn examines each slot in the values against a value_set array.
/// If the value is not found in value_set, null will be output.
/// If found, the index of occurrence within value_set (ignoring duplicates)
/// will be output.
///
/// For example given values = [99, 42, 3, null] and
/// value_set = [3, 3, 99], the output will be = [1, null, 0, null]
///
/// Behaviour of nulls is governed by SetLookupOptions::skip_nulls.
///
/// \param[in] values array-like input
/// \param[in] options SetLookupOptions
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IndexIn(const Datum& values, const SetLookupOptions& options,
                      ExecContext* ctx = NULLPTR);
ARROW_EXPORT
Result<Datum> IndexIn(const Datum& values, const Datum& value_set,
                      ExecContext* ctx = NULLPTR);

/// \brief IsValid returns true for each element of `values` that is not null,
/// false otherwise
///
/// \param[in] values input to examine for validity
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsValid(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief IsNull returns true for each element of `values` that is null,
/// false otherwise
///
/// \param[in] values input to examine for nullity
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsNull(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief IsNan returns true for each element of `values` that is NaN,
/// false otherwise
///
/// \param[in] values input to look for NaN
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsNan(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief FillNull replaces each null element in `values`
/// with `fill_value`
///
/// \param[in] values input to examine for nullity
/// \param[in] fill_value scalar
/// \param[in] ctx the function execution context, optional
///
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> FillNull(const Datum& values, const Datum& fill_value,
                       ExecContext* ctx = NULLPTR);

/// \brief IfElse returns elements chosen from `left` or `right`
/// depending on `cond`. `null` values in `cond` will be promoted to the result
///
/// \param[in] cond `Boolean` condition Scalar/ Array
/// \param[in] left Scalar/ Array
/// \param[in] right Scalar/ Array
/// \param[in] ctx the function execution context, optional
///
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IfElse(const Datum& cond, const Datum& left, const Datum& right,
                     ExecContext* ctx = NULLPTR);

/// \brief CaseWhen behaves like a switch/case or if-else if-else statement: for
/// each row, select the first value for which the corresponding condition is
/// true, or (if given) select the 'else' value, else emit null. Note that a
/// null condition is the same as false.
///
/// \param[in] cond Conditions (Boolean)
/// \param[in] cases Values (any type), along with an optional 'else' value.
/// \param[in] ctx the function execution context, optional
///
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> CaseWhen(const Datum& cond, const std::vector<Datum>& cases,
                       ExecContext* ctx = NULLPTR);

/// \brief Year returns year for each element of `values`
///
/// \param[in] values input to extract year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Year(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Month returns month for each element of `values`.
/// Month is encoded as January=1, December=12
///
/// \param[in] values input to extract month from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Month(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Day returns day number for each element of `values`
///
/// \param[in] values input to extract day from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Day(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief DayOfWeek returns number of the day of the week value for each element of
/// `values`.
///
/// By default week starts on Monday denoted by 0 and ends on Sunday denoted
/// by 6. Start day of the week (Monday=1, Sunday=7) and numbering base (0 or 1) can be
/// set using DayOfWeekOptions
///
/// \param[in] values input to extract number of the day of the week from
/// \param[in] options for setting start of the week and day numbering
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DayOfWeek(const Datum& values,
                                     DayOfWeekOptions options = DayOfWeekOptions(),
                                     ExecContext* ctx = NULLPTR);

/// \brief DayOfYear returns number of day of the year for each element of `values`.
/// January 1st maps to day number 1, February 1st to 32, etc.
///
/// \param[in] values input to extract number of day of the year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DayOfYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief ISOYear returns ISO year number for each element of `values`.
/// First week of an ISO year has the majority (4 or more) of its days in January.
///
/// \param[in] values input to extract ISO year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> ISOYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief ISOWeek returns ISO week of year number for each element of `values`.
/// First ISO week has the majority (4 or more) of its days in January.
/// Week of the year starts with 1 and can run up to 53.
///
/// \param[in] values input to extract ISO week of year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> ISOWeek(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief ISOCalendar returns a (ISO year, ISO week, ISO day of week) struct for
/// each element of `values`.
/// ISO week starts on Monday denoted by 1 and ends on Sunday denoted by 7.
///
/// \param[in] values input to ISO calendar struct from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> ISOCalendar(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Quarter returns the quarter of year number for each element of `values`
/// First quarter maps to 1 and fourth quarter maps to 4.
///
/// \param[in] values input to extract quarter of year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Quarter(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Hour returns hour value for each element of `values`
///
/// \param[in] values input to extract hour from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Hour(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Minute returns minutes value for each element of `values`
///
/// \param[in] values input to extract minutes from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Minute(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Second returns seconds value for each element of `values`
///
/// \param[in] values input to extract seconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Second(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Millisecond returns number of milliseconds since the last full second
/// for each element of `values`
///
/// \param[in] values input to extract milliseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Millisecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Microsecond returns number of microseconds since the last full millisecond
/// for each element of `values`
///
/// \param[in] values input to extract microseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Microsecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Nanosecond returns number of nanoseconds since the last full millisecond
/// for each element of `values`
///
/// \param[in] values input to extract nanoseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Nanosecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Subsecond returns the fraction of second elapsed since last full second
/// as a float for each element of `values`
///
/// \param[in] values input to extract subsecond from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Subsecond(const Datum& values, ExecContext* ctx = NULLPTR);

}  // namespace compute
}  // namespace arrow
