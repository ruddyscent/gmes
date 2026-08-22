#include "pw_material.hh"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gmes
{
  bool
  openmp_enabled() noexcept
  {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
  }

  int
  openmp_max_threads() noexcept
  {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
  }

  std::size_t
  openmp_cell_threshold() noexcept
  {
    return openmp_threshold();
  }
} // namespace gmes
