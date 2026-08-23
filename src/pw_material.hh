#ifndef PW_MATERIAL_HH_
#define PW_MATERIAL_HH_

#include <algorithm>
#include <array>
#include <iterator>
#include <functional>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include "cpp23_support.hh"

namespace gmes
{
  bool openmp_enabled() noexcept;
  int openmp_max_threads() noexcept;
  std::size_t openmp_cell_threshold() noexcept;

  struct PwMaterialParam
  {
    virtual ~PwMaterialParam() = default;
  }; // struct PwMaterialParam

  template <typename T>
  struct ElectricParam: public PwMaterialParam
  {
    double eps_inf;
  }; // template ElectricParam

  template <typename T>
  struct MagneticParam: public PwMaterialParam
  {
    double mu_inf;
  }; // template MagneticParam

  typedef std::array<int, 3> Index3;
  using IdxCnt = std::vector<Index3>;

  struct UpdateOffsets
  {
    std::size_t target;
    std::size_t in1_first;
    std::size_t in1_second;
    std::size_t in2_first;
    std::size_t in2_second;
  };

  struct UpdateRun
  {
    UpdateOffsets offsets;
    std::array<std::size_t, 5> strides;
    std::size_t begin;
    std::size_t size;
  };

  inline Index3
  make_index(const int* const data, int size)
  {
    if (size != 3)
      throw std::invalid_argument("field indices must contain exactly three values");

    const std::span<const int, 3> values(data, 3);
    return {values[0], values[1], values[2]};
  }

  template <typename T>
  class PwMaterial
  {
  public:
    static_assert(FieldScalar<T>,
                  "PwMaterial requires a floating-point or complex field type");

    virtual
    ~PwMaterial() = default;

    virtual const std::string& name() const = 0;

    // TODO: just copy PwMaterialParam*.
    virtual PwMaterial<T>*
    attach(const int* const idx, int idx_size,
	   const PwMaterialParam* const parameter) = 0;

    void
    finalize(int component,
             int target_dim1, int target_dim2, int target_dim3,
             int in1_dim1, int in1_dim2, int in1_dim3,
             int in2_dim1, int in2_dim2, int in2_dim3)
    {
      if (component < 0 || component > 5)
	throw std::invalid_argument("field component must be in [0, 5]");

      const std::array<int, 10> signature = {
        component,
        target_dim1, target_dim2, target_dim3,
        in1_dim1, in1_dim2, in1_dim3,
        in2_dim1, in2_dim2, in2_dim3,
      };
      if (plan_signature == signature && planned_cells == idx_list.size())
	return;

      const std::array<std::array<int, 3>, 4> component_stencils[] = {
        {{{1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 0, 0}}},
        {{{0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 1, 0}}},
        {{{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 1}}},
        {{{0, 0, -1}, {0, -1, -1}, {0, -1, 0}, {0, -1, -1}}},
        {{{-1, 0, 0}, {-1, 0, -1}, {0, 0, -1}, {-1, 0, -1}}},
        {{{0, -1, 0}, {-1, -1, 0}, {-1, 0, 0}, {-1, -1, 0}}},
      };
      const std::array<std::array<int, 3>, 3> collapsed_axes = {{
        {{1, 0, 2}},
        {{2, 1, 0}},
        {{0, 2, 1}},
      }};

      const auto& stencil = component_stencils[component];
      const int family = component % 3;
      const auto& axes = collapsed_axes[family];
      const bool target_collapsed = signature[1 + axes[0]] == 1;
      const bool in1_collapsed = signature[4 + axes[1]] == 1;
      const bool in2_collapsed = signature[7 + axes[2]] == 1;

      std::vector<UpdateRun> new_runs;
      new_runs.reserve(
        (idx_list.size() + update_tile_size - 1) / update_tile_size);
      bool new_parallel_safe = true;
      bool targets_strictly_increasing = true;
      std::size_t previous_target = 0;
      for (std::size_t position = 0; position < idx_list.size(); ++position) {
        const auto& idx = idx_list[position];

        UpdateOffsets offsets{};
        offsets.target = field_offset(
          target_dim1, target_dim2, target_dim3, target_collapsed,
          idx[0], idx[1], idx[2]);
        if (position > 0 && offsets.target <= previous_target) {
          targets_strictly_increasing = false;
          if (offsets.target == previous_target)
            new_parallel_safe = false;
        }
        previous_target = offsets.target;
        if (uses_input_stencil()) {
          offsets.in1_first = offset_with_delta(
            idx, stencil[0], in1_dim1, in1_dim2, in1_dim3, in1_collapsed);
          offsets.in1_second = offset_with_delta(
            idx, stencil[1], in1_dim1, in1_dim2, in1_dim3, in1_collapsed);
          offsets.in2_first = offset_with_delta(
            idx, stencil[2], in2_dim1, in2_dim2, in2_dim3, in2_collapsed);
          offsets.in2_second = offset_with_delta(
            idx, stencil[3], in2_dim1, in2_dim2, in2_dim3, in2_collapsed);
        }
        if (new_runs.empty() ||
            new_runs.back().size >= update_tile_size ||
            !offsets_continue(new_runs.back(), offsets, uses_input_stencil()))
          new_runs.push_back({offsets, {}, position, 1});
        else
          ++new_runs.back().size;
      }

      if (!targets_strictly_increasing && new_parallel_safe) {
        std::vector<std::size_t> targets;
        targets.reserve(idx_list.size());
        for (const auto& idx : idx_list) {
          targets.push_back(field_offset(
            target_dim1, target_dim2, target_dim3, target_collapsed,
            idx[0], idx[1], idx[2]));
        }
        std::ranges::sort(targets);
        new_parallel_safe = std::adjacent_find(targets.begin(), targets.end()) ==
                            targets.end();
      }

      update_runs.swap(new_runs);
      planned_cells = idx_list.size();
      plan_signature = signature;
      parallel_safe_plan = new_parallel_safe;
    }

    bool
    is_finalized() const noexcept
    {
      return planned_cells == idx_list.size() &&
             plan_signature[0] != invalid_component;
    }

    std::size_t
    plan_size() const noexcept
    {
      return planned_cells;
    }

    std::size_t
    plan_run_count() const noexcept
    {
      return update_runs.size();
    }

    bool
    plan_is_parallel_safe() const noexcept
    {
      return parallel_safe_plan;
    }

    std::size_t
    plan_bytes() const noexcept
    {
      return update_runs.capacity() * sizeof(UpdateRun);
    }

    PwMaterial<T>*
    attach_many(const int* const indices, int index_rows, int index_cols,
		const std::vector<PwMaterialParam*>& parameters)
    {
      if (index_cols != 3)
	throw std::invalid_argument("bulk field indices must have shape (n, 3)");
      if (index_rows != static_cast<int>(parameters.size()))
	throw std::invalid_argument("bulk indices and parameters must have equal lengths");

      std::set<Index3> unique_indices(idx_list.begin(), idx_list.end());
      for (int i = 0; i < index_rows; ++i) {
	const Index3 index = make_index(indices + 3 * i, index_cols);
	if (std::ranges::any_of(index, [](int value) { return value < 0; }))
	  throw std::out_of_range("field indices must be non-negative");
	if (!unique_indices.insert(index).second)
	  throw std::invalid_argument("bulk field indices must not contain duplicates");
	validate_parameter(parameters[i]);
      }

      reserve(idx_list.size() + parameters.size());
      for (int i = 0; i < index_rows; ++i)
	attach(indices + 3 * i, index_cols, parameters[i]);

      return this;
    }

    void
    reserve(std::size_t capacity)
    {
      idx_list.reserve(capacity);
      reserve_parameters(capacity);
    }

    virtual void
    update_all(T* const inplace_field,
	       int inplace_dim1, int inplace_dim2, int inplace_dim3,
	       const T* const in_field1,
	       int in1_dim1, int in1_dim2, int in1_dim3,
	       const T* const in_field2,
	       int in2_dim1, int in2_dim2, int in2_dim3,
	       double d1, double d2, double dt, double n) = 0;

    IdxCnt::const_iterator
    find(const Index3& idx) const
    {
      auto it = std::find(idx_list.begin(), idx_list.end(), idx);
      return it;
    }

    virtual PwMaterial<T>*
    merge(const PwMaterial<T>* const pm) = 0;

    IdxCnt::size_type
    idx_size() const
    {
      return idx_list.size();
    }

    bool
    indices_in_bounds(int dim1, int dim2, int dim3) const
    {
      return std::ranges::all_of(idx_list, [=](const Index3& idx) {
        return idx[0] >= 0 && idx[0] < dim1 &&
               idx[1] >= 0 && idx[1] < dim2 &&
               idx[2] >= 0 && idx[2] < dim3;
      });
    }

  protected:
    static constexpr int invalid_component = -1;
    static constexpr std::size_t update_tile_size = 2048;

    virtual bool
    uses_input_stencil() const noexcept
    {
      return true;
    }

    void
    finalize_update_plan(int component,
                         int target_dim1, int target_dim2, int target_dim3,
                         int in1_dim1, int in1_dim2, int in1_dim3,
                         int in2_dim1, int in2_dim2, int in2_dim3)
    {
      finalize(component,
               target_dim1, target_dim2, target_dim3,
               in1_dim1, in1_dim2, in1_dim3,
               in2_dim1, in2_dim2, in2_dim3);
    }

    template <std::ranges::random_access_range Parameters, typename Function>
    void
    for_each_planned(Parameters& parameters, Function&& function)
    {
      if (std::ranges::size(parameters) != planned_cells)
	throw std::logic_error("material plan and parameters are out of sync");

      auto apply_run = [&](const UpdateRun& run) {
        UpdateOffsets offsets = run.offsets;
        for (std::size_t local = 0; local < run.size; ++local) {
          function(offsets, parameters[run.begin + local]);
          offsets.target += run.strides[0];
          offsets.in1_first += run.strides[1];
          offsets.in1_second += run.strides[2];
          offsets.in2_first += run.strides[3];
          offsets.in2_second += run.strides[4];
        }
      };

#if defined(_OPENMP)
      if (parallel_safe_plan && planned_cells >= openmp_threshold()) {
        std::exception_ptr error;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t position = 0;
             position < static_cast<std::ptrdiff_t>(update_runs.size());
             ++position) {
          try {
            apply_run(update_runs[position]);
          }
          catch (...) {
#pragma omp critical(gmes_for_each_planned_exception)
            {
              if (!error)
                error = std::current_exception();
            }
          }
        }
        if (error)
          std::rethrow_exception(error);
        return;
      }
#endif

      for (const auto& run : update_runs)
        apply_run(run);
    }

    virtual bool
    accepts_parameter(const PwMaterialParam* parameter) const noexcept = 0;

    virtual void
    reserve_parameters(std::size_t capacity) = 0;

    void
    validate_parameter(const PwMaterialParam* parameter) const
    {
      if (!accepts_parameter(parameter))
	throw std::invalid_argument("parameter type does not match pointwise material");
    }

    int
    position(const Index3& idx) const
    {
      auto it = find(idx);

      size_t pos = 0;
      if (it == idx_list.end())
	return pos - 1;
      else {
	pos = std::distance(idx_list.begin(), it);
	return pos;
      }
    }

    IdxCnt idx_list;

  private:
    static std::size_t
    offset_with_delta(const Index3& idx, const std::array<int, 3>& delta,
                      int dim1, int dim2, int dim3, bool collapsed)
    {
      return field_offset(dim1, dim2, dim3, collapsed,
                          idx[0] + delta[0],
                          idx[1] + delta[1],
                          idx[2] + delta[2]);
    }

    static bool
    offsets_continue(UpdateRun& run, const UpdateOffsets& offsets,
                     bool include_stencil) noexcept
    {
      const std::array<std::size_t, 5> starts = {
        run.offsets.target,
        run.offsets.in1_first,
        run.offsets.in1_second,
        run.offsets.in2_first,
        run.offsets.in2_second,
      };
      const std::array<std::size_t, 5> next_offsets = {
        offsets.target,
        offsets.in1_first,
        offsets.in1_second,
        offsets.in2_first,
        offsets.in2_second,
      };
      const std::size_t count = include_stencil ? starts.size() : 1;
      for (std::size_t field = 0; field < count; ++field) {
        if (run.size == 1) {
          if (next_offsets[field] < starts[field])
            return false;
          run.strides[field] = next_offsets[field] - starts[field];
        }
        else if (next_offsets[field] !=
                 starts[field] + run.strides[field] * run.size) {
          return false;
        }
      }
      return true;
    }

    std::vector<UpdateRun> update_runs;
    std::size_t planned_cells = 0;
    std::array<int, 10> plan_signature = {
      invalid_component, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    bool parallel_safe_plan = true;
  }; // template PwMaterial

  template <typename T>
  class MaterialElectric: public PwMaterial<T>
  {
  public:
    virtual double
    get_eps_inf(const int* const idx, int idx_size) const = 0;

    using PwMaterial<T>::find;

  protected:
    using PwMaterial<T>::position;
    using PwMaterial<T>::idx_list;
  }; // template MaterialElectric

  template <typename T>
  class MaterialMagnetic: public PwMaterial<T>
  {
  public:
    virtual double
    get_mu_inf(const int* const idx, int idx_size) const = 0;

    using PwMaterial<T>::find;

  protected:
    using PwMaterial<T>::position;
    using PwMaterial<T>::idx_list;
  }; // template MaterialMagnetic
} // namespace gmes

#endif // PW_MATERIAL_HH_
