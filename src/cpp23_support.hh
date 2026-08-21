#ifndef CPP23_SUPPORT_HH_
#define CPP23_SUPPORT_HH_

#include <complex>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <ranges>
#include <stdexcept>
#include <type_traits>

namespace gmes
{
  template <typename T>
  concept FieldScalar =
    std::floating_point<T> ||
    std::same_as<std::remove_cv_t<T>, std::complex<float>> ||
    std::same_as<std::remove_cv_t<T>, std::complex<double>>;

  template <FieldScalar T>
  inline T&
  field_at(T* const data,
           int x_size, int y_size, int z_size,
           bool collapsed,
           int i, int j, int k)
  {
    if (collapsed)
      return data[0];

    using Extents = std::dextents<std::size_t, 3>;
    std::mdspan<T, Extents> field(data, x_size, y_size, z_size);
    return field[i, j, k];
  }

  template <std::ranges::sized_range First, std::ranges::sized_range Second>
  inline auto
  zip_equal(First& first, Second& second)
  {
    if (std::ranges::size(first) != std::ranges::size(second))
      throw std::logic_error("material indices and parameters are out of sync");
    return std::views::zip(first, second);
  }
} // namespace gmes

#endif // CPP23_SUPPORT_HH_
