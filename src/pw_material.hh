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
