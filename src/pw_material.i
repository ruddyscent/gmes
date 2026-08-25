%module pw_material

%{
#define SWIG_FILE_WITH_INIT
#include "pw_material.hh"
#include "pw_dummy.hh"
#include "pw_const.hh"
#include "pw_dielectric.hh"
#include "pw_upml.hh"
#include "pw_cpml.hh"
#include "pw_drude.hh"
#include "pw_lorentz.hh"
#include "pw_dcp.hh"
#include "pw_dm2.hh"
%}

%include <std_string.i>
%include <std_complex.i>
%include <std_except.i>
%include <std_vector.i>
%include <exception.i>
%include "numpy.i"

%exception {
  try {
    $action
  } catch (const std::invalid_argument& error) {
    SWIG_exception(SWIG_ValueError, error.what());
  } catch (const std::out_of_range& error) {
    SWIG_exception(SWIG_IndexError, error.what());
  } catch (const std::exception& error) {
    SWIG_exception(SWIG_RuntimeError, error.what());
  }
}

%numpy_typemaps(std::complex<double>, NPY_CDOUBLE, int)
%apply size_t { gmes::IdxCnt::size_type };

%init %{
import_array();
%}

%ignore gmes::PwMaterial::indices_in_bounds;
%rename(_attach_many_native) gmes::PwMaterial::attach_many;

// Declare numpy typemaps.
%define %apply_numpy_typemaps(TYPE)
%typemap(check)
      (TYPE* const inplace_field, int inplace_dim1, int inplace_dim2, int inplace_dim3)
{
  if (!arg1->indices_in_bounds($2, $3, $4))
    SWIG_exception_fail(SWIG_IndexError, "field index is out of bounds");
}

%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const in_field1, int in1_dim1, int in1_dim2, int in1_dim3)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const in_field2, int in2_dim1, int in2_dim2, int in2_dim3)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const inplace_field, int inplace_dim1, int inplace_dim2, int inplace_dim3)};

%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const ex, int ex_x_size, int ex_y_size, int ex_z_size)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const ey, int ey_x_size, int ey_y_size, int ey_z_size)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const ez, int ez_x_size, int ez_y_size, int ez_z_size)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const hx, int hx_x_size, int hx_y_size, int hx_z_size)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const hy, int hy_x_size, int hy_y_size, int hy_z_size)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(const TYPE* const hz, int hz_x_size, int hz_y_size, int hz_z_size)};

%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const ex, int ex_x_size, int ex_y_size, int ex_z_size)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const ey, int ey_x_size, int ey_y_size, int ey_z_size)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const ez, int ez_x_size, int ez_y_size, int ez_z_size)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const hx, int hx_x_size, int hx_y_size, int hx_z_size)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const hy, int hy_x_size, int hy_y_size, int hy_z_size)};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* const hz, int hz_x_size, int hz_y_size, int hz_z_size)};
%enddef    /* apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)
%apply_numpy_typemaps(std::complex<double>)

%apply (int* IN_ARRAY1, int DIM1) {(const int* const idx, int idx_size)};
%apply (int* IN_ARRAY2, int DIM1, int DIM2)
      {(const int* const indices, int index_rows, int index_cols)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* const a, int a_size1, int a_size2)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* const b, int b_size1, int b_size2)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* const u_new_values, int u_new_rows, int u_new_cols)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* const u_ref_values, int u_ref_rows, int u_ref_cols)};
%apply (std::complex<double>* IN_ARRAY2, int DIM1, int DIM2) {(const std::complex<double>* const b, int b_size1, int b_size2)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* const c, int c_size)};

%apply (double* IN_ARRAY1, int DIM1) {(const double* const omega, int omega_size)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* const n, int n_size)};

%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* const u, int u_size)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* const v, int v_size)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* const w, int w_size)};

// Include the header file to be wrapped
%template(OracleIndexVector) std::vector<int>;
%template(OracleStateVector) std::vector<std::complex<double>>;
%include "pw_material.hh"
%template(PwMaterialParamVector) std::vector<gmes::PwMaterialParam *>;
%include "pw_dummy.hh"
%include "pw_const.hh"
%include "pw_dielectric.hh"
%include "pw_upml.hh"
%include "pw_cpml.hh"
%include "pw_drude.hh"
%include "pw_lorentz.hh"
%include "pw_dcp.hh"
%include "pw_dm2.hh"

%inline %{
double _dm2_relative_error(
  double e_new,
  const double* const u_new_values, int u_new_rows, int u_new_cols,
  double e_ref,
  const double* const u_ref_values, int u_ref_rows, int u_ref_cols)
{
  if (u_new_cols != 3 || u_ref_cols != 3 || u_new_rows != u_ref_rows)
    throw std::invalid_argument("atomic states must have matching (n, 3) shapes");

  std::vector<std::array<double, 3> > u_new(u_new_rows), u_ref(u_ref_rows);
  for (int i = 0; i < u_new_rows; ++i) {
    std::copy_n(u_new_values + 3 * i, 3, u_new[i].begin());
    std::copy_n(u_ref_values + 3 * i, 3, u_ref[i].begin());
  }

  return gmes::rel_error(e_new, u_new, e_ref, u_ref);
}
%}

// Instantiate template classes
%define %linear_wrap(T, postfix)

%template(ElectricParam ## postfix) gmes::ElectricParam<T >;
%template(MagneticParam ## postfix) gmes::MagneticParam<T >;
%template(PwMaterial ## postfix) gmes::PwMaterial<T >;
%template(MaterialElectric ## postfix) gmes::MaterialElectric<T >;
%template(MaterialMagnetic ## postfix) gmes::MaterialMagnetic<T >;

%template(DummyElectricParam ## postfix) gmes::DummyElectricParam<T >;
%template(DummyMagneticParam ## postfix) gmes::DummyMagneticParam<T >;
%template(DummyElectric ## postfix) gmes::DummyElectric<T >;
%template(DummyMagnetic ## postfix) gmes::DummyMagnetic<T >;
%template(DummyEx ## postfix) gmes::DummyEx<T >;
%template(DummyEy ## postfix) gmes::DummyEy<T >;
%template(DummyEz ## postfix) gmes::DummyEz<T >;
%template(DummyHx ## postfix) gmes::DummyHx<T >;
%template(DummyHy ## postfix) gmes::DummyHy<T >;
%template(DummyHz ## postfix) gmes::DummyHz<T >;

%template(ConstElectricParam ## postfix) gmes::ConstElectricParam<T >;
%template(ConstMagneticParam ## postfix) gmes::ConstMagneticParam<T >;
%template(ConstElectric ## postfix) gmes::ConstElectric<T >;
%template(ConstMagnetic ## postfix) gmes::ConstMagnetic<T >;
%template(ConstEx ## postfix) gmes::ConstEx<T >;
%template(ConstEy ## postfix) gmes::ConstEy<T >;
%template(ConstEz ## postfix) gmes::ConstEz<T >;
%template(ConstHx ## postfix) gmes::ConstHx<T >;
%template(ConstHy ## postfix) gmes::ConstHy<T >;
%template(ConstHz ## postfix) gmes::ConstHz<T >;

// Non-dispersive linear isotropic dielectrics
%template(DielectricElectricParam ## postfix) gmes::DielectricElectricParam<T >;
%template(DielectricMagneticParam ## postfix) gmes::DielectricMagneticParam<T >;
%template(DielectricElectric ## postfix) gmes::DielectricElectric<T >;
%template(DielectricMagnetic ## postfix) gmes::DielectricMagnetic<T >;
%template(DielectricEx ## postfix) gmes::DielectricEx<T >;
%template(DielectricEy ## postfix) gmes::DielectricEy<T >;
%template(DielectricEz ## postfix) gmes::DielectricEz<T >;
%template(DielectricHx ## postfix) gmes::DielectricHx<T >;
%template(DielectricHy ## postfix) gmes::DielectricHy<T >;
%template(DielectricHz ## postfix) gmes::DielectricHz<T >;

// UPML
%template(UpmlElectricParam ## postfix) gmes::UpmlElectricParam<T >;
%template(UpmlMagneticParam ## postfix) gmes::UpmlMagneticParam<T >;
%template(UpmlElectric ## postfix) gmes::UpmlElectric<T >;
%template(UpmlMagnetic ## postfix) gmes::UpmlMagnetic<T >;
%template(UpmlEx ## postfix) gmes::UpmlEx<T >;
%template(UpmlEy ## postfix) gmes::UpmlEy<T >;
%template(UpmlEz ## postfix) gmes::UpmlEz<T >;
%template(UpmlHx ## postfix) gmes::UpmlHx<T >;
%template(UpmlHy ## postfix) gmes::UpmlHy<T >;
%template(UpmlHz ## postfix) gmes::UpmlHz<T >;

// CPML
%template(CpmlElectricParam ## postfix) gmes::CpmlElectricParam<T >;
%template(CpmlMagneticParam ## postfix) gmes::CpmlMagneticParam<T >;
%template(CpmlElectric ## postfix) gmes::CpmlElectric<T >;
%template(CpmlMagnetic ## postfix) gmes::CpmlMagnetic<T >;
%template(CpmlEx ## postfix) gmes::CpmlEx<T >;
%template(CpmlEy ## postfix) gmes::CpmlEy<T >;
%template(CpmlEz ## postfix) gmes::CpmlEz<T >;
%template(CpmlHx ## postfix) gmes::CpmlHx<T >;
%template(CpmlHy ## postfix) gmes::CpmlHy<T >;
%template(CpmlHz ## postfix) gmes::CpmlHz<T >;

// Drude model
%template(DrudeElectricParam ## postfix) gmes::DrudeElectricParam<T >;
%template(DrudeMagneticParam ## postfix) gmes::DrudeMagneticParam<T >;
%template(DrudeElectric ## postfix) gmes::DrudeElectric<T >;
%template(DrudeEx ## postfix) gmes::DrudeEx<T >;
%template(DrudeEy ## postfix) gmes::DrudeEy<T >;
%template(DrudeEz ## postfix) gmes::DrudeEz<T >;
%template(DrudeHx ## postfix) gmes::DrudeHx<T >;
%template(DrudeHy ## postfix) gmes::DrudeHy<T >;
%template(DrudeHz ## postfix) gmes::DrudeHz<T >;

%extend gmes::DrudeElectricParam<T >
{
  void set(const double* const a, int a_size1, int a_size2,
	   const double* const c, int c_size)
  {
    if (a_size2 != 3 || c_size != 3)
      throw std::invalid_argument("Drude coefficients require shapes (n, 3) and (3,)");

    for (int i = 0; i < a_size1; i++) {
      std::array<double, 3> tmp;
      std::copy(a + i * a_size2, a + i * a_size2 + 3, tmp.begin());
      $self->a.push_back(tmp);
    }

    std::copy(c, c + c_size, $self->c.begin());

    $self->q_now.resize(a_size1, T(0));
    $self->q_new.resize(a_size1, T(0));
  }
};

// Lerentz model
%template(LorentzElectricParam ## postfix) gmes::LorentzElectricParam<T >;
%template(LorentzMagneticParam ## postfix) gmes::LorentzMagneticParam<T >;
%template(LorentzElectric ## postfix) gmes::LorentzElectric<T >;
%template(LorentzEx ## postfix) gmes::LorentzEx<T >;
%template(LorentzEy ## postfix) gmes::LorentzEy<T >;
%template(LorentzEz ## postfix) gmes::LorentzEz<T >;
%template(LorentzHx ## postfix) gmes::LorentzHx<T >;
%template(LorentzHy ## postfix) gmes::LorentzHy<T >;
%template(LorentzHz ## postfix) gmes::LorentzHz<T >;

%extend gmes::LorentzElectricParam<T >
{
  void set(const double* const a, int a_size1, int a_size2,
	   const double* const c, int c_size)
  {
    if (a_size2 != 3 || c_size != 3)
      throw std::invalid_argument("Lorentz coefficients require shapes (n, 3) and (3,)");

    for (int i = 0; i < a_size1; i++) {
      std::array<double, 3> tmp;
      std::copy(a + i * a_size2, a + i * a_size2 + 3, tmp.begin());
      $self->a.push_back(tmp);
    }

    std::copy(c, c + c_size, $self->c.begin());

    $self->l_now.resize(a_size1);
    $self->l_new.resize(a_size1);
  }
};

// ADE implementation of the Drude-critical points model
%template(DcpAdeElectricParam ## postfix) gmes::DcpAdeElectricParam<T >;
%template(DcpAdeMagneticParam ## postfix) gmes::DcpAdeMagneticParam<T >;
%template(DcpAdeElectric ## postfix) gmes::DcpAdeElectric<T >;
%template(DcpAdeEx ## postfix) gmes::DcpAdeEx<T >;
%template(DcpAdeEy ## postfix) gmes::DcpAdeEy<T >;
%template(DcpAdeEz ## postfix) gmes::DcpAdeEz<T >;
%template(DcpAdeHx ## postfix) gmes::DcpAdeHx<T >;
%template(DcpAdeHy ## postfix) gmes::DcpAdeHy<T >;
%template(DcpAdeHz ## postfix) gmes::DcpAdeHz<T >;

%extend gmes::DcpAdeElectricParam<T >
{
  void set(const double* const a, int a_size1, int a_size2,
	   const double* const b, int b_size1, int b_size2,
	   const double* const c, int c_size)
  {
    if (a_size2 != 3 || b_size2 != 5 || c_size != 4)
      throw std::invalid_argument("DcpAde coefficients require shapes (n, 3), (n, 5), and (4,)");

    for (int i = 0; i < a_size1; i++) {
      std::array<double, 3> tmp;
      std::copy(a + i * a_size2, a + i * a_size2 + 3, tmp.begin());
      $self->a.push_back(tmp);
    }

    for (int i = 0; i < b_size1; i++) {
      std::array<double, 5> tmp;
      std::copy(b + i * b_size2, b + i * b_size2 + 5, tmp.begin());
      $self->b.push_back(tmp);
    }

    std::copy(c, c + c_size, $self->c.begin());

    $self->q_old.resize(a_size1);
    $self->q_now.resize(a_size1);
    $self->p_old.resize(b_size1);
    $self->p_now.resize(b_size1);
  }
};

// PLRC implementation of the Drude-critical points model
%template(DcpPlrcElectricParam ## postfix) gmes::DcpPlrcElectricParam<T >;
%template(DcpPlrcMagneticParam ## postfix) gmes::DcpPlrcMagneticParam<T >;
%template(DcpPlrcElectric ## postfix) gmes::DcpPlrcElectric<T >;
%template(DcpPlrcEx ## postfix) gmes::DcpPlrcEx<T >;
%template(DcpPlrcEy ## postfix) gmes::DcpPlrcEy<T >;
%template(DcpPlrcEz ## postfix) gmes::DcpPlrcEz<T >;
%template(DcpPlrcHx ## postfix) gmes::DcpPlrcHx<T >;
%template(DcpPlrcHy ## postfix) gmes::DcpPlrcHy<T >;
%template(DcpPlrcHz ## postfix) gmes::DcpPlrcHz<T >;

%extend gmes::DcpPlrcElectricParam<T >
{
  void set(const double* const a, int a_size1, int a_size2,
	   const std::complex<double>* const b, int b_size1, int b_size2,
	   const double* const c, int c_size)
  {
    if (a_size2 != 3 || b_size2 != 3 || c_size != 3)
      throw std::invalid_argument("DcpPlrc coefficients require shapes (n, 3), (n, 3), and (3,)");

    for (int i = 0; i < a_size1; i++) {
      std::array<double, 3> tmp;
      std::copy(a + i * a_size2, a + i * a_size2 + 3, tmp.begin());
      $self->a.push_back(tmp);
    }

    for (int i = 0; i < b_size1; i++) {
      std::array<std::complex<double>, 3> tmp;
      std::copy(b + i * b_size2, b + i * b_size2 + 3, tmp.begin());
      $self->b.push_back(tmp);
    }

    std::copy(c, c + c_size, $self->c.begin());

    $self->psi_dp_re.resize(a_size1);
    $self->psi_dp_im.resize(a_size1);
    $self->psi_cp_re.resize(b_size1);
    $self->psi_cp_im.resize(b_size1);
  }
};

%enddef    /* linear_wrap() macro */

%define %nonlinear_wrap(T, postfix)

// density matrix implementation
%template(Dm2ElectricParam ## postfix) gmes::Dm2ElectricParam<T >;
%template(Dm2MagneticParam ## postfix) gmes::Dm2MagneticParam<T >;
%template(Dm2Electric ## postfix) gmes::Dm2Electric<T >;
%template(Dm2Ex ## postfix) gmes::Dm2Ex<T >;
%template(Dm2Ey ## postfix) gmes::Dm2Ey<T >;
%template(Dm2Ez ## postfix) gmes::Dm2Ez<T >;
%template(Dm2Hx ## postfix) gmes::Dm2Hx<T >;
%template(Dm2Hy ## postfix) gmes::Dm2Hy<T >;
%template(Dm2Hz ## postfix) gmes::Dm2Hz<T >;

%extend gmes::Dm2ElectricParam<T >
{
  void set(const double* const omega, int omega_size,
           const double* const n, int n_size)
  {
    if (omega_size != n_size)
      throw std::invalid_argument("omega and n_atom must have equal lengths");

    for (int i = 0; i < omega_size; i++) {
      $self->omega.push_back(*(omega + i));
      $self->n_atom.push_back(*(n + i));

      std::array<double, 3> u_tmp;
      u_tmp.fill(static_cast<T>(0));
      $self->u.push_back(u_tmp);
      $self->u_new_scratch.push_back(u_tmp);
      $self->u_previous_scratch.push_back(u_tmp);
    }
    $self->a_scratch.resize(omega_size);
    $self->b_scratch.resize(omega_size);
  }
};

%enddef    /* dm2_wrap() macro */

%linear_wrap(double, Real)
%linear_wrap(std::complex<double>, Cmplx)

%nonlinear_wrap(double, Real)

%pythoncode %{
import numpy as _np


def _pointwise_parameter_type(pointwise):
    name = type(pointwise).__name__
    if name.endswith("Real"):
        scalar = "Real"
    elif name.endswith("Cmplx"):
        scalar = "Cmplx"
    else:
        raise TypeError("unsupported pointwise material type")

    stem = name[: -len(scalar)]
    component = stem[-2:]
    if component not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        raise TypeError("unsupported pointwise material component")

    field = "Electric" if component.startswith("E") else "Magnetic"
    parameter_name = f"{stem[:-2]}{field}Param{scalar}"
    try:
        return globals()[parameter_name]
    except KeyError as error:
        raise TypeError("unsupported pointwise material parameter type") from error


def _attach_many(self, indices, parameters):
    """Attach an ordered batch of indices and matching parameter objects.

    Indices must be a C-contiguous ``numpy.intc`` array with shape ``(n, 3)``.
    Negative or duplicate indices are rejected before the updater is changed;
    upper bounds are checked later against the actual field passed to
    ``update_all``. Parameter objects must match the concrete updater type.
    """
    if not isinstance(indices, _np.ndarray):
        raise TypeError("bulk indices must be a NumPy array")
    if indices.dtype != _np.dtype(_np.intc):
        raise TypeError("bulk indices must use the platform C int dtype")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("bulk field indices must have shape (n, 3)")
    if not indices.flags.c_contiguous:
        raise ValueError("bulk indices must be C-contiguous")

    try:
        parameters = list(parameters)
    except TypeError as error:
        raise TypeError("bulk parameters must be iterable") from error

    if len(parameters) != indices.shape[0]:
        raise ValueError("bulk indices and parameters must have equal lengths")

    parameter_type = _pointwise_parameter_type(self)
    if any(not isinstance(parameter, parameter_type) for parameter in parameters):
        raise TypeError(
            f"bulk parameters for {type(self).__name__} must be "
            f"{parameter_type.__name__} instances"
        )

    self._attach_many_native(indices, parameters)
    return self


PwMaterialReal.attach_many = _attach_many
PwMaterialCmplx.attach_many = _attach_many
%}
