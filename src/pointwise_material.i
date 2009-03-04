%module pointwise_material

%{
#define SWIG_FILE_WITH_INIT
#include "pointwise_material.hh"
#include "pointwise_dummy.hh"
#include "pointwise_zero.hh"
#include "pointwise_one.hh"
#include "pointwise_dielectric.hh"
#include "pointwise_upml.hh"
#include "pointwise_cpml.hh"
#include "pointwise_drude.hh"
%}

// Get the NumPy typemaps
%include "numpy.i"

%init %{
import_array();
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE* IN_ARRAY1, int DIM1)
      {(const double * const omega_p, int omega_p_size)};
%apply (TYPE* IN_ARRAY1, int DIM1)
      {(const double * const gamma_p, int gamma_p_size)};
			
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
      
%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

%apply (int* IN_ARRAY1, int DIM1) {(const int* const idx, int size)};
      
%define Property(py, cpp, prop, get, set)
%feature("shadow") cpp::set %{ %}
%feature("shadow") cpp::get %{
__swig_setmethods__["prop"] = eval("_"+__name__.split('.')[-1]).##py##_##set
__swig_getmethods__["prop"] = eval("_"+__name__.split('.')[-1]).##py##_##get
if _newclass:prop = property(eval("_"+__name__.split('.')[-1]).##py##_##get, eval("_"+__name__.split('.')[-1]).##py##_##set)
%}
%enddef

Property(PointwiseMaterial, gmes::PointwiseMaterial, i, get_i, set_i)
Property(PointwiseMaterial, gmes::PointwiseMaterial, j, get_j, set_j)
Property(PointwiseMaterial, gmes::PointwiseMaterial, k, get_k, set_k)
Property(DummyElectric, gmes::MaterialElectric, epsilon, get_epsilon, set_epsilon)
Property(DummyMagnetic, gmes::MaterialMagnetic, mu, get_mu, set_mu)

// Include the header file to be wrapped
%include "pointwise_material.hh"
%include "pointwise_dummy.hh"
%include "pointwise_zero.hh"
%include "pointwise_one.hh"
%include "pointwise_dielectric.hh"
%include "pointwise_upml.hh"
%include "pointwise_cpml.hh"
%include "pointwise_drude.hh"

