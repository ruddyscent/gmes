/* This implementation is based on the following article.
 *
 * M. Okoniewski and E. Okoniewska, "Drude dispersion in ADE FDTD
 * revisited," Electron. Lett., 42, 503-504, (2006).
 */

#ifndef PW_DRUDE_HH_
#define PW_DRUDE_HH_

#include <array>
#include <vector>
#include "pw_dielectric.hh"


namespace gmes
{
  template <typename T>
  struct DrudeElectricParam: public ElectricParam<T>
  {
    std::vector<std::array<double, 3> > a;
    std::array<double, 3> c;
    std::vector<T> q_now, q_new;
  }; // template DrudeElectricParam

  template <typename T>
  struct DrudeMagneticParam: public MagneticParam<T>
  {
  }; // template DrudeMagneticParam

  template <typename T>
  class DrudeElectric: public MaterialElectric<T>
  {
  public:
    const std::string&
    name() const
    {
      return DrudeElectric<T>::tag;
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
      const auto& drude_param = *static_cast<const DrudeElectricParam<T>*>(pm_param_ptr);

      idx_list.push_back(index);
      param_list.push_back(drude_param);

      return this;
    };

    PwMaterial<T>*
    merge(const PwMaterial<T>* const pm_ptr)
    {
      auto drude_ptr = static_cast<const DrudeElectric<T>*>(pm_ptr);
      std::copy(drude_ptr->idx_list.begin(), drude_ptr->idx_list.end(), std::back_inserter(idx_list));
      std::copy(drude_ptr->param_list.begin(), drude_ptr->param_list.end(), std::back_inserter(param_list));
      return this;
    }

    std::vector<std::complex<double>>
    oracle_state() const override
    {
      std::vector<std::complex<double>> values;
      for (const auto& parameter : param_list) {
        values.insert(values.end(), parameter.q_now.begin(), parameter.q_now.end());
        values.insert(values.end(), parameter.q_new.begin(), parameter.q_new.end());
      }
      return values;
    }

    std::size_t
    oracle_state_bytes() const noexcept override
    {
      std::size_t bytes = 0;
      for (const auto& parameter : param_list)
        bytes += (parameter.q_now.capacity() + parameter.q_new.capacity()) * sizeof(T);
      return bytes;
    }

    std::size_t
    oracle_parameter_bytes() const noexcept override
    {
      std::size_t bytes =
        param_list.capacity() * sizeof(DrudeElectricParam<T>);
      for (const auto& parameter : param_list)
        bytes += parameter.a.capacity() * sizeof(std::array<double, 3>) +
                 (parameter.q_now.capacity() + parameter.q_new.capacity()) *
                   sizeof(T);
      return bytes;
    }

    T
    dps_sum(const T& init, const DrudeElectricParam<T>& drude_param) const
    {
      const auto& a = drude_param.a;
      const auto& q_now = drude_param.q_now;
      const auto& q_new = drude_param.q_new;

      T sum(init);
      for (typename std::vector<T>::size_type i = 0; i < a.size(); ++i)	{
	sum += q_new[i] - q_now[i];
      }

      return sum;
    }

    void
    update_q(const T& e_now, DrudeElectricParam<T>& drude_param)
    {
      const std::vector<std::array<double, 3> >& a = drude_param.a;
      std::vector<T>& q_now = drude_param.q_now;
      std::vector<T>& q_new = drude_param.q_new;

      for (typename std::vector<T>::size_type i = 0; i < a.size(); ++i)	{
	const T q_old = q_now[i];
	q_now[i] = q_new[i];
	q_new[i] = a[i][0] * q_old + a[i][1] * q_now[i] + a[i][2] * e_now;
      }
    }

  protected:
    bool
    accepts_parameter(const PwMaterialParam* parameter) const noexcept override
    {
      return dynamic_cast<const DrudeElectricParam<T>*>(parameter) != nullptr;
    }

    void
    reserve_parameters(std::size_t capacity) override
    {
      param_list.reserve(capacity);
    }

    using MaterialElectric<T>::position;
    using MaterialElectric<T>::idx_list;
    std::vector<DrudeElectricParam<T> > param_list;

  private:
    static const std::string tag; // "DrudeElectric"
  }; // template DrudeElectric

  template <typename T>
  const std::string DrudeElectric<T>::tag = "DrudeElectric";

  template <typename T>
  class DrudeEx: public DrudeElectric<T>
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
	   DrudeElectricParam<T>& drude_param)
    {

      const auto& c = drude_param.c;

      const T& e_now = ex[offsets.target];
      update_q(e_now, drude_param);
      ex[offsets.target] = c[0] * ((hz[offsets.in1_first] - hz[offsets.in1_second]) / dy -
			  (hy[offsets.in2_first] - hy[offsets.in2_second]) / dz)
	+ c[1] * dps_sum(static_cast<T>(0), drude_param) + c[2] * e_now;
    }

  protected:
    using DrudeElectric<T>::idx_list;
    using DrudeElectric<T>::param_list;
    using DrudeElectric<T>::update_q;
    using DrudeElectric<T>::dps_sum;
  }; // template DrudeEx

  template <typename T>
  class DrudeEy: public DrudeElectric<T>
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
	   DrudeElectricParam<T>& drude_param)
    {

      const auto& c = drude_param.c;

      const T& e_now = ey[offsets.target];
      update_q(e_now, drude_param);
      ey[offsets.target] = c[0] * ((hx[offsets.in1_first] - hx[offsets.in1_second]) / dz -
			  (hz[offsets.in2_first] - hz[offsets.in2_second]) / dx)
	+ c[1] * dps_sum(static_cast<T>(0), drude_param) + c[2] * e_now;
    }

  protected:
    using DrudeElectric<T>::idx_list;
    using DrudeElectric<T>::param_list;
    using DrudeElectric<T>::update_q;
    using DrudeElectric<T>::dps_sum;
  }; // template DrudeEy

  template <typename T> class DrudeEz: public DrudeElectric<T>
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
	       dx, dy, dt, n, idx, param);
      });
    }

  private:
    void
    update(T* const ez, int ez_x_size, int ez_y_size, int ez_z_size,
	   const T* const hy, int hy_x_size, int hy_y_size, int hy_z_size,
	   const T* const hx, int hx_x_size, int hx_y_size, int hx_z_size,
	   double dx, double dy, double dt, double n,
	   const UpdateOffsets& offsets,
	   DrudeElectricParam<T>& drude_param)
    {

      const auto& c = drude_param.c;

      const T& e_now = ez[offsets.target];
      update_q(e_now, drude_param);
      ez[offsets.target] = c[0] * ((hy[offsets.in1_first] - hy[offsets.in1_second]) / dx -
			  (hx[offsets.in2_first] - hx[offsets.in2_second]) / dy)
	+ c[1] * dps_sum(static_cast<T>(0), drude_param) + c[2] * e_now;
    }

  protected:
    using DrudeElectric<T>::idx_list;
    using DrudeElectric<T>::param_list;
    using DrudeElectric<T>::update_q;
    using DrudeElectric<T>::dps_sum;
  }; // template DrudeEz

  template <typename T>
  class DrudeHx: public DielectricHx<T>
  {
  public:
    const std::string&
    name() const
    {
      return DrudeHx<T>::tag;
    }

  private:
    static const std::string tag; // "DrudeMagnetic"
  }; // template DrudeHx

  template <typename T>
  const std::string DrudeHx<T>::tag = "DrudeMagnetic";

  template <typename T>
  class DrudeHy: public DielectricHy<T>
  {
  public:
    const std::string&
    name() const
    {
      return DrudeHy<T>::tag;
    }

  private:
    static const std::string tag; // "DrudeMagnetic"
  }; // template DrudeHy

  template <typename T>
  const std::string DrudeHy<T>::tag = "DrudeMagnetic";

  template <typename T>
  class DrudeHz: public DielectricHz<T>
  {
  public:
    const std::string&
    name() const
    {
      return DrudeHz<T>::tag;
    }

  private:
    static const std::string tag; // "DrudeMagnetic"
  }; // template DrudeHz

  template <typename T>
  const std::string DrudeHz<T>::tag = "DrudeMagnetic";

} // namespace gmes

#undef ex
#undef ey
#undef ez
#undef hx
#undef hy
#undef hz

#endif // PW_DRUDE_HH_
