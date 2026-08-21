#ifndef CPP23_SUPPORT_HH_
#define CPP23_SUPPORT_HH_

#include <complex>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#if __has_include(<mdspan>)
#include <mdspan>
#define GMES_HAS_STD_MDSPAN 1
#else
#define GMES_HAS_STD_MDSPAN 0
#endif

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

#if GMES_HAS_STD_MDSPAN
    using Extents = std::dextents<std::size_t, 3>;
    std::mdspan<T, Extents> field(data, x_size, y_size, z_size);
    return field[i, j, k];
#else
    static_cast<void>(x_size);
    const auto offset =
      (static_cast<std::size_t>(i) * static_cast<std::size_t>(y_size) +
       static_cast<std::size_t>(j)) * static_cast<std::size_t>(z_size) +
      static_cast<std::size_t>(k);
    return data[offset];
#endif
  }

#if !defined(__cpp_lib_ranges_zip) || __cpp_lib_ranges_zip < 202110L
  template <std::ranges::range First, std::ranges::range Second>
  class EqualZipView
  {
    First* first;
    Second* second;

    class Iterator
    {
      std::ranges::iterator_t<First> first_iterator;
      std::ranges::iterator_t<Second> second_iterator;

    public:
      Iterator(std::ranges::iterator_t<First> first_arg,
               std::ranges::iterator_t<Second> second_arg):
        first_iterator(first_arg), second_iterator(second_arg)
      {}

      auto
      operator*() const
      {
        return std::tie(*first_iterator, *second_iterator);
      }

      Iterator&
      operator++()
      {
        ++first_iterator;
        ++second_iterator;
        return *this;
      }

      bool
      operator!=(const Iterator& other) const
      {
        return first_iterator != other.first_iterator;
      }
    };

  public:
    EqualZipView(First& first_arg, Second& second_arg):
      first(&first_arg), second(&second_arg)
    {}

    Iterator
    begin()
    {
      return Iterator(std::ranges::begin(*first), std::ranges::begin(*second));
    }

    Iterator
    end()
    {
      return Iterator(std::ranges::end(*first), std::ranges::end(*second));
    }
  };
#endif

  template <std::ranges::sized_range First, std::ranges::sized_range Second>
  inline auto
  zip_equal(First& first, Second& second)
  {
    if (std::ranges::size(first) != std::ranges::size(second))
      throw std::logic_error("material indices and parameters are out of sync");
#if defined(__cpp_lib_ranges_zip) && __cpp_lib_ranges_zip >= 202110L
    return std::views::zip(first, second);
#else
    return EqualZipView(first, second);
#endif
  }
} // namespace gmes

#undef GMES_HAS_STD_MDSPAN

#endif // CPP23_SUPPORT_HH_
