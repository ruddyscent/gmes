/* This implementation is based on the following article:
 *
 * M. Okoniewski, M. Mrozowski, and M. A. Stuchly, "Simple
 * treatment of multi-term dispersion in FDTD," IEEE Microw.
 * Guided Wave Lett. 7, 121-123 (1997).
 */

#ifndef PW_LORENTZ_HH_
#define PW_LORENTZ_HH_

#include <array>
#include <vector>
#include "pw_dielectric.hh"


namespace gmes
{
  template <typename T>
  struct LorentzElectricParam: public ElectricParam<T>
  {
    std::vector<std::array<double, 3> > a;
    std::array<double, 3> c;
    std::vector<T> l_now, l_new;
  }; // template LorentzElectricParam

  template <typename T>
  struct LorentzMagneticParam: public MagneticParam<T>
  {
  }; // template LorentzMagneticParam

  template <typename T>
  class LorentzElectric: public MaterialElectric<T>
  {
  public:
    const std::string&
    name() const
    {
      return LorentzElectric<T>::tag;
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
      const auto& lorentz_param = *static_cast<const LorentzElectricParam<T> * const>(pm_param_ptr);

      idx_list.push_back(index);
      param_list.push_back(lorentz_param);

      return this;
    };

    PwMaterial<T>*
    merge(const PwMaterial<T>* const pm_ptr)
    {
      auto lorentz_ptr
	= static_cast<const LorentzElectric<T>*>(pm_ptr);
      std::copy(lorentz_ptr->idx_list.begin(),
		lorentz_ptr->idx_list.end(),
		std::back_inserter(idx_list));
      std::copy(lorentz_ptr->param_list.begin(),
		lorentz_ptr->param_list.end(),
		std::back_inserter(param_list));
      return this;
    }

    T
    lps_sum(const T& init, const LorentzElectricParam<T>& lorentz_param) const
    {
      const auto& a = lorentz_param.a;
      const auto& l_now = lorentz_param.l_now;
      const auto& l_new = lorentz_param.l_new;

      T sum(init);
      for (typename std::vector<T>::size_type i = 0; i < a.size(); ++i)	{
	sum += l_new[i] - l_now[i];
      }

      return sum;
    }

    void
    update_l(const T& e_now, LorentzElectricParam<T>& lorentz_param)
    {
      const auto& a = lorentz_param.a;
      auto& l_now = lorentz_param.l_now;
      auto& l_new = lorentz_param.l_new;

      for (typename std::vector<T>::size_type i = 0; i < a.size(); ++i)	{
	const T l_old = l_now[i];
	l_now[i] = l_new[i];
	l_new[i] = a[i][0] * l_old + a[i][1] * l_now[i] + a[i][2] * e_now;
      }
    }

  protected:
    bool
    accepts_parameter(const PwMaterialParam* parameter) const noexcept override
    {
      return dynamic_cast<const LorentzElectricParam<T>*>(parameter) != nullptr;
    }

    void
    reserve_parameters(std::size_t capacity) override
    {
      param_list.reserve(capacity);
    }

    using MaterialElectric<T>::position;
    using MaterialElectric<T>::idx_list;
    std::vector<LorentzElectricParam<T> > param_list;

  private:
    static const std::string tag; // "LorentzElectric"
  }; // template LorentzElectric

  template <typename T>
  const std::string LorentzElectric<T>::tag = "LorentzElectric";

  template <typename T>
  class LorentzEx: public LorentzElectric<T>
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
	   LorentzElectricParam<T>& lorentz_param)
    {

      const auto& c = lorentz_param.c;

      const T& e_now = ex[offsets.target];
      update_l(e_now, lorentz_param);
      ex[offsets.target] = c[0] * ((hz[offsets.in1_first] - hz[offsets.in1_second]) / dy -
			  (hy[offsets.in2_first] - hy[offsets.in2_second]) / dz)
	+ c[1] * lps_sum(static_cast<T>(0), lorentz_param) + c[2] * e_now;
    }

  protected:
    using LorentzElectric<T>::idx_list;
    using LorentzElectric<T>::param_list;
    using LorentzElectric<T>::update_l;
    using LorentzElectric<T>::lps_sum;
  }; // template LorentzEx

  template <typename T>
  class LorentzEy: public LorentzElectric<T>
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
	   LorentzElectricParam<T>& lorentz_param)
    {

      const auto& c = lorentz_param.c;

      const T& e_now = ey[offsets.target];
      update_l(e_now, lorentz_param);
      ey[offsets.target] = c[0] * ((hx[offsets.in1_first] - hx[offsets.in1_second]) / dz -
			  (hz[offsets.in2_first] - hz[offsets.in2_second]) / dx)
	+ c[1] * lps_sum(static_cast<T>(0), lorentz_param) + c[2] * e_now;
    }

  protected:
    using LorentzElectric<T>::idx_list;
    using LorentzElectric<T>::param_list;
    using LorentzElectric<T>::update_l;
    using LorentzElectric<T>::lps_sum;
  }; // template LorentzEy

  template <typename T>
  class LorentzEz: public LorentzElectric<T>
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
	   LorentzElectricParam<T>& lorentz_param)
    {

      const auto& c = lorentz_param.c;

      const T& e_now = ez[offsets.target];
      update_l(e_now, lorentz_param);
      ez[offsets.target] = c[0] * ((hy[offsets.in1_first] - hy[offsets.in1_second]) / dx -
			  (hx[offsets.in2_first] - hx[offsets.in2_second]) / dy)
	+ c[1] * lps_sum(static_cast<T>(0), lorentz_param) + c[2] * e_now;
    }

  protected:
    using LorentzElectric<T>::idx_list;
    using LorentzElectric<T>::param_list;
    using LorentzElectric<T>::update_l;
    using LorentzElectric<T>::lps_sum;
  }; // template LorentzEz

  template <typename T>
  class LorentzHx: public DielectricHx<T>
  {
  public:
    const std::string&
    name() const
    {
      return LorentzHx<T>::tag;
    }

  private:
    static const std::string tag; // "LorentzMagnetic"
  }; // template LorentzHx

  template <typename T>
  const std::string LorentzHx<T>::tag = "LorentzMagnetic";

  template <typename T>
  class LorentzHy: public DielectricHy<T>
  {
  public:
    const std::string&
    name() const
    {
      return LorentzHy<T>::tag;
    }

  private:
    static const std::string tag; // "LorentzMagnetic"
  }; // template LorentzHy

  template <typename T>
  const std::string LorentzHy<T>::tag = "LorentzMagnetic";

  template <typename T>
  class LorentzHz: public DielectricHz<T>
  {
  public:
    const std::string&
    name() const
    {
      return LorentzHz<T>::tag;
    }

  private:
    static const std::string tag; // "LorentzMagnetic"
  }; // template LorentzHz

  template <typename T>
  const std::string LorentzHz<T>::tag = "LorentzMagnetic";
} // namespace gmes

#undef ex
#undef ey
#undef ez
#undef hx
#undef hy
#undef hz

#endif // PW_LORENTZ_HH_
