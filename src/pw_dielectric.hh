/* This implementation is based on the following article.
 *
 * K. S. Yee, "Numerical solution of initial boundary value
 * problems involving Maxwell¡¯s equations in isotropic media,"
 * IEEE Trans. Antennas Propag. 14, 302-307 (1966).
 */

#ifndef PW_DIELECTRIC_HH_
#define PW_DIELECTRIC_HH_

#include <utility>
#include "pw_material.hh"


namespace gmes
{
  template <typename T>
  struct DielectricElectricParam: ElectricParam<T>
  {
  }; // template DielectricElectricParam

  template <typename T>
  struct DielectricMagneticParam: MagneticParam<T>
  {
  }; // template DielectricMagneticParam

  template <typename T>
  class DielectricElectric: public MaterialElectric<T>
  {
  public:
    const std::string&
    name() const
    {
      return DielectricElectric<T>::tag;
    }

    double
    get_eps_inf(const int* const idx, int idx_size) const
    {
      const Index3 index = make_index(idx, idx_size);
      const int i = position(index);
      if (i < 0)
	return 0;
      else
	return param_list[i].eps_inf;
    }

    PwMaterial<T>*
    attach(const int* const idx, int idx_size,
	   const PwMaterialParam* const pm_param_ptr)
    {
      const Index3 index = make_index(idx, idx_size);

      this->validate_parameter(pm_param_ptr);
      const auto& dielectric_param = *static_cast<const DielectricElectricParam<T>*>(pm_param_ptr);

      idx_list.push_back(index);
      param_list.push_back(dielectric_param);

      return this;
    }

    PwMaterial<T>*
    merge(const PwMaterial<T>* const pm_ptr)
    {
      auto dielectric_ptr
	= static_cast<const DielectricElectric<T>*>(pm_ptr);
      std::copy(dielectric_ptr->idx_list.begin(),
		dielectric_ptr->idx_list.end(),
		std::back_inserter(idx_list));
      std::copy(dielectric_ptr->param_list.begin(),
		dielectric_ptr->param_list.end(),
		std::back_inserter(param_list));
      return this;
    }

    std::size_t
    oracle_parameter_bytes() const noexcept override
    {
      return param_list.capacity() * sizeof(DielectricElectricParam<T>);
    }

  protected:
    bool
    accepts_parameter(const PwMaterialParam* parameter) const noexcept override
    {
      return dynamic_cast<const DielectricElectricParam<T>*>(parameter) != nullptr;
    }

    void
    reserve_parameters(std::size_t capacity) override
    {
      param_list.reserve(capacity);
    }

    using MaterialElectric<T>::position;
    using MaterialElectric<T>::idx_list;
    std::vector<DielectricElectricParam<T> > param_list;

  private:
    static const std::string tag; // "DielectricElectric"
  }; // template DielectricElectric

  template <typename T>
  const std::string DielectricElectric<T>::tag = "DielectricElectric";

  template <typename T>
  class DielectricEx: public DielectricElectric<T>
  {
  public:
    void
    update_all(T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	       const T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	       const T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	       double dy, double dz, double dt, double n)
    {
      this->finalize_update_plan(0, ex_x_size, ex_y_size, ex_z_size, hz_x_size, hz_y_size, hz_z_size, hy_x_size, hy_y_size, hy_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
    	update(ex, ex_x_size, ex_y_size, ex_z_size,
	       hz, hz_x_size, hz_y_size, hz_z_size,
	       hy, hy_x_size, hy_y_size, hy_z_size,
	       dy, dz, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	   const T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	   const T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	   double dy, double dz, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricElectricParam<T>& dielectric_param) const
    {
      const double eps_inf = dielectric_param.eps_inf;

      ex[offsets.target] += dt / eps_inf * ((hz[offsets.in1_first] - hz[offsets.in1_second]) / dy -
				   (hy[offsets.in2_first] - hy[offsets.in2_second]) / dz);
    }

  protected:
    using DielectricElectric<T>::idx_list;
    using DielectricElectric<T>::param_list;
  }; // template DielectricEx

  template <typename T>
  class DielectricEy: public DielectricElectric<T>
  {
  public:
    void
    update_all(T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	       const T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	       const T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	       double dz, double dx, double dt, double n)
    {
      this->finalize_update_plan(1, ey_x_size, ey_y_size, ey_z_size, hx_x_size, hx_y_size, hx_z_size, hz_x_size, hz_y_size, hz_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
      	update(ey, ey_x_size, ey_y_size, ey_z_size,
	       hx, hx_x_size, hx_y_size, hx_z_size,
	       hz, hz_x_size, hz_y_size, hz_z_size,
	       dz, dx, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	   const T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	   const T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	   double dz, double dx, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricElectricParam<T>& dielectric_param) const
    {
      const double eps_inf = dielectric_param.eps_inf;

      ey[offsets.target] += dt / eps_inf * ((hx[offsets.in1_first] - hx[offsets.in1_second]) / dz -
				   (hz[offsets.in2_first] - hz[offsets.in2_second]) / dx);
    }

  protected:
    using DielectricElectric<T>::idx_list;
    using DielectricElectric<T>::param_list;
  }; // template DielectricEy

  template <typename T>
  class DielectricEz: public DielectricElectric<T>
  {
  public:
    void
    update_all(T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	       const T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	       const T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	       double dx, double dy, double dt, double n)
    {
      this->finalize_update_plan(2, ez_x_size, ez_y_size, ez_z_size, hy_x_size, hy_y_size, hy_z_size, hx_x_size, hx_y_size, hx_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
	update(ez, ez_x_size, ez_y_size, ez_z_size,
	       hy, hy_x_size, hy_y_size, hy_z_size,
	       hx, hx_x_size, hx_y_size, hx_z_size,
	       dx, dy, dt, n,
               idx, param);
      });
    }

  private:
    void
    update(T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	   const T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	   const T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	   double dx, double dy, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricElectricParam<T>& dielectric_param) const
    {
      const double eps_inf = dielectric_param.eps_inf;

      ez[offsets.target] += dt / eps_inf * ((hy[offsets.in1_first] - hy[offsets.in1_second]) / dx -
                                   (hx[offsets.in2_first] - hx[offsets.in2_second]) / dy);
    }

  protected:
    using DielectricElectric<T>::idx_list;
    using DielectricElectric<T>::param_list;
  }; // template DielectricEz

  template <typename T>
  class DielectricMagnetic: public MaterialMagnetic<T>
  {
  public:
    const std::string&
    name() const
    {
      return DielectricMagnetic<T>::tag;
    }

    double
    get_mu_inf(const int* const idx, int idx_size) const
    {
      const Index3 index = make_index(idx, idx_size);
      const int i = position(index);
      if (i < 0)
	return 0;
      else
	return param_list[i].mu_inf;
    }

    PwMaterial<T>*
    attach(const int* const idx, int idx_size,
	   const PwMaterialParam* const pm_param_ptr)
    {
      const Index3 index = make_index(idx, idx_size);

      this->validate_parameter(pm_param_ptr);
      const auto& magnetic_param = *static_cast<const MagneticParam<T>*>(pm_param_ptr);
      DielectricMagneticParam<T> dielectric_param;
      dielectric_param.mu_inf = magnetic_param.mu_inf;

      idx_list.push_back(index);
      param_list.push_back(dielectric_param);

      return this;
    }

    PwMaterial<T>*
    merge(const PwMaterial<T>* const pm_ptr)
    {
      auto dielectric_ptr = static_cast<const DielectricMagnetic<T>*>(pm_ptr);
      std::copy(dielectric_ptr->idx_list.begin(), dielectric_ptr->idx_list.end(), std::back_inserter(idx_list));
      std::copy(dielectric_ptr->param_list.begin(), dielectric_ptr->param_list.end(), std::back_inserter(param_list));
      return this;
    }

    std::size_t
    oracle_parameter_bytes() const noexcept override
    {
      return param_list.capacity() * sizeof(DielectricMagneticParam<T>);
    }

  protected:
    bool
    accepts_parameter(const PwMaterialParam* parameter) const noexcept override
    {
      return dynamic_cast<const MagneticParam<T>*>(parameter) != nullptr;
    }

    void
    reserve_parameters(std::size_t capacity) override
    {
      param_list.reserve(capacity);
    }

    using MaterialMagnetic<T>::position;
    using MaterialMagnetic<T>::idx_list;
    std::vector<DielectricMagneticParam<T> > param_list;

  private:
    static const std::string tag; // "DielectricMagnetic"
  }; // template DielectriMagnetic

  template <typename T>
  const std::string DielectricMagnetic<T>::tag = "DielectricMagnetic";

  template <typename T>
  class DielectricHx: public DielectricMagnetic<T>
  {
  public:
    void
    update_all(T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	       const T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	       const T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	       double dy, double dz, double dt, double n)
    {
      this->finalize_update_plan(3, hx_x_size, hx_y_size, hx_z_size, ez_x_size, ez_y_size, ez_z_size, ey_x_size, ey_y_size, ey_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
      	update(hx, hx_x_size, hx_y_size, hx_z_size,
	       ez, ez_x_size, ez_y_size, ez_z_size,
	       ey, ey_x_size, ey_y_size, ey_z_size,
	       dy, dz, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	   const T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	   const T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	   double dy, double dz, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricMagneticParam<T>& dielectric_param) const
    {
      const double mu_inf = dielectric_param.mu_inf;

      hx[offsets.target] += dt / mu_inf * ((ey[offsets.in2_first] - ey[offsets.in2_second]) / dz -
                                  (ez[offsets.in1_first] - ez[offsets.in1_second]) / dy);
    }

  protected:
    using DielectricMagnetic<T>::idx_list;
    using DielectricMagnetic<T>::param_list;
  }; // template DielectricHx

  template <typename T>
  class DielectricHy: public DielectricMagnetic<T>
  {
  public:
    void
    update_all(T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	       const T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	       const T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	       double dz, double dx, double dt, double n)
    {
      this->finalize_update_plan(4, hy_x_size, hy_y_size, hy_z_size, ex_x_size, ex_y_size, ex_z_size, ez_x_size, ez_y_size, ez_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
      	update(hy, hy_x_size, hy_y_size, hy_z_size,
	       ex, ex_x_size, ex_y_size, ex_z_size,
	       ez, ez_x_size, ez_y_size, ez_z_size,
	       dz, dx, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	   const T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	   const T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	   double dz, double dx, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricMagneticParam<T>& dielectric_param) const
    {
      const double mu_inf = dielectric_param.mu_inf;

      hy[offsets.target] += dt / mu_inf * ((ez[offsets.in2_first] - ez[offsets.in2_second]) / dx -
                                  (ex[offsets.in1_first] - ex[offsets.in1_second]) / dz);
    }

  protected:
    using DielectricMagnetic<T>::idx_list;
    using DielectricMagnetic<T>::param_list;
  }; // template DielectricHy

  template <typename T>
  class DielectricHz: public DielectricMagnetic<T>
  {
  public:
    void
    update_all(T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	       const T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	       const T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	       double dx, double dy, double dt, double n)
    {
      this->finalize_update_plan(5, hz_x_size, hz_y_size, hz_z_size, ey_x_size, ey_y_size, ey_z_size, ex_x_size, ex_y_size, ex_z_size);
      this->for_each_planned(param_list, [&](const auto& idx, auto& param) {
    	update(hz, hz_x_size, hz_y_size, hz_z_size,
	       ey, ey_x_size, ey_y_size, ey_z_size,
	       ex, ex_x_size, ex_y_size, ex_z_size,
	       dx, dy, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const hz, int hz_x_size, int hz_y_size, int hz_z_size,
	   const T* const ey, int ey_x_size, int ey_y_size, int ey_z_size,
	   const T* const ex, int ex_x_size, int ex_y_size, int ex_z_size,
	   double dx, double dy, double dt, double n,
	   const UpdateOffsets& offsets,
	   const DielectricMagneticParam<T>& dielectric_param) const
    {
      const double mu_inf = dielectric_param.mu_inf;

      hz[offsets.target] += dt / mu_inf * ((ex[offsets.in2_first] - ex[offsets.in2_second]) / dy -
				  (ey[offsets.in1_first] - ey[offsets.in1_second]) / dx);
    }

  protected:
    using DielectricMagnetic<T>::idx_list;
    using DielectricMagnetic<T>::param_list;
  }; // template DielectricHz
} // namespace gmes

#undef ex
#undef ey
#undef ez
#undef hx
#undef hy
#undef hz

#endif // PW_DIELECTRIC_HH_
