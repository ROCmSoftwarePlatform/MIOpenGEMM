/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

template <>
char get_floattype_char<float>()
{
  return 'f';
}

template <>
char get_floattype_char<double>()
{
  return 'd';
}

Geometry::Geometry(
  size_t m_, size_t n_, size_t k_, bool tA_, bool tB_, std::vector<size_t> wSS, char ftype)
  : Geometry(true, tA_, tB_, false, tA_ ? k_ : m_, tB_ ? n_ : k_, m_, m_, n_, k_, wSS, ftype)
{
}

size_t get_mat_size(const Geometry& gg, const Offsets& toff, Mat::E emat)
{
  return (gg.get_padded_area(emat) + toff.offsets[emat] + toff.tails[emat]);
}

size_t get_mat_memsize(const Geometry& gg, const Offsets& toff, Mat::E emat)
{
  return gg.derived.float_size_bytes * get_mat_size(gg, toff, emat);
}

Offsets::Offsets(size_t              oa_,
                 size_t              ob_,
                 size_t              oc_,
                 std::vector<size_t> ow_,
                 size_t              ta_,
                 size_t              tb_,
                 size_t              tc_,
                 std::vector<size_t> tw_)
{

  if (ow_.size() != tw_.size())
  {
    throw miog_error(
      "tail offset and leading offset vectors are not of the same size if Offsets constructor");
  }

  offsets[Mat::E::A] = oa_;
  offsets[Mat::E::B] = ob_;
  offsets[Mat::E::C] = oc_;
  offsets_vws        = ow_;

  tails[Mat::E::A] = ta_;
  tails[Mat::E::B] = tb_;
  tails[Mat::E::C] = tc_;
  tails_vws        = tw_;
}

Offsets get_padding_offsets(size_t n_workspaces)
{

  std::vector<size_t> pre_w(n_workspaces, 101);
  std::vector<size_t> post_w(n_workspaces, 103);

  return Offsets(11, 17, 13, pre_w, 67, 15, 29, post_w);
}

Offsets get_zero_offsets(size_t n_workspaces)
{
  std::vector<size_t> wsp(n_workspaces, 1);
  return Offsets(0, 0, 0, wsp, 0, 0, 0, wsp);
}

Geometry get_tight_geometry(size_t n_workspaces)
{
  std::vector<size_t> wsp(n_workspaces, 1);
  return {false, false, false, false, 1, 1, 1, 1, 1, 1, wsp, 'f'};
}

char get_floattype(size_t nbits)
{

  char ft = 'x';
  if (nbits == 8 * sizeof(float))
  {
    ft = 'f';
  }
  else if (nbits == 8 * sizeof(double))
  {
    ft = 'd';
  }
  else
  {
    std::stringstream ss;
    ss << "floattype with number of bits : " << nbits << " ? (in get_floattype of geometry)";
    throw miog_error(ss.str());
  }
  return ft;
}

void GeometryDerived::reset(char floattype)
{
  if (floattype == 'f')
  {
    float_size_bytes = sizeof(float);
  }
  else if (floattype == 'd')
  {
    float_size_bytes = sizeof(double);
  }
  else
  {
    throw miog_error("what is this floattype : " + std::to_string(floattype) +
                     std::string(" ? in reset of geometry"));
  }
  float_size_bits = 8 * float_size_bytes;
}

// return one of the dimensions of matrix a,b,c.
// isCoal : the coalesced dimesion?
// For example, for A which is m x k,
// if tA = false, isColMajor = false,
// isCoal = true, then k is returned as k is
// the coalesced dim.
// (false == false) == true  evaluates to true,
// so gate is true, so m is returned
size_t Geometry::get_padless_dim(Mat::E M, bool isCoal) const
{

  bool gate = (tX.at(M) == isColMajor) == isCoal;

  if (M == Mat::E::A)
  {
    return gate ? k : m;
  }

  else if (M == Mat::E::B)
  {
    return gate ? n : k;
  }

  else if (M == Mat::E::C)
  {
    return gate ? n : m;
  }

  else
  {
    throw miog_error("unrecognised M passed to get_coal in get_padless_dim of geometry");
  }
}

size_t Geometry::get_non_k_dim(Mat::E M) const
{

  if (M == Mat::E::A)
  {
    return m;
  }

  else if (M == Mat::E::B)
  {
    return n;
  }

  else
  {
    throw miog_error("invalid char passed to get_non_k_dim in get_non_k_dim of "
                     "geometry, it should be either a or b");
  }
}

void Geometry::check_ldx_consistent() const
{

  bool error = false;
  for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (ldX[x] < get_coal(x))
    {
      error = true;
    }
  }

  if (error == true)
  {

    std::stringstream errm_ss;

    errm_ss << "Checking that lda, ldb, ldc are consistent with m,n,k. "
            << "In particulary, checking that ldx is at least as large as "
            << "coalesced dimension, "
            << "coal_x (for x in {a,b,c}) given by:  "
            << "coal_a = (tA == isColMajor ? k : m),  "
            << "coal_b = (tB == isColMajor ? n : k),  "
            << "coal_c = (tC == isColMajor ? n : m).  "
            << "\n\n"
            << "ldx = coal_x + pad_x, and so for consisteny it must be true "
            << "that ldx >= coal_x (can't have negative pad_x).  "
            << "As an example, if tA = false and isColMajor = false, then "
            << "coal_a = k.  "
            << "A full table of the lower bounds of ldx for x in {a,b,c} can "
            << "be found at, "
            << "https://software.intel.com/en-us/"
            << "mkl-developer-reference-c-cblas-gemm.  "
            << "\n\n"
            << "The particular geometry received by in geometry "
            << "check_ldx_consistent is  " << get_string() << ", and the problems detected are:  ";

    for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
    {
      if (ldX[x] < get_coal(x))
      {
        errm_ss << "ld" << Mat::M().name[x] << " (" << ldX[x] << ") <  coal_" << Mat::M().name[x]
                << " (" << get_coal(x) << ").  ";
      }
    }

    throw miog_error(errm_ss.str());
  }
}

size_t Geometry::get_uncoal(Mat::E M) const { return get_padless_dim(M, false); }

// this is lda, ldb, ldc if they are minimal.
size_t Geometry::get_coal(Mat::E M) const { return get_padless_dim(M, true); }

bool Geometry::coal_is_pll_k(Mat::E M) const
{
  // proof : false, false, true should give 1
  return (static_cast<size_t>(isColMajor) + static_cast<size_t>(tX.at(M)) +
          static_cast<size_t>(M == Mat::E::A)) %
         2;
}

void Geometry::initialise(bool                isColMajor_,
                          bool                tA_,
                          bool                tB_,
                          bool                tC_,
                          size_t              lda_,
                          size_t              ldb_,
                          size_t              ldc_,
                          size_t              m_,
                          size_t              n_,
                          size_t              k_,
                          std::vector<size_t> wSpaceSize_,
                          char                floattype_)
{

  isColMajor = isColMajor_;
  m          = m_;
  n          = n_;
  k          = k_;
  wSpaceSize = wSpaceSize_;
  std::sort(wSpaceSize.begin(), wSpaceSize.end(), std::greater<size_t>());

  floattype = floattype_;

  tX.resize(Mat::E::N);
  tX[Mat::E::A] = tA_;
  tX[Mat::E::B] = tB_;
  tX[Mat::E::C] = tC_;

  ldX.resize(Mat::E::N);
  ldX[Mat::E::A] = lda_;
  ldX[Mat::E::B] = ldb_;
  ldX[Mat::E::C] = ldc_;

  if (floattype != 'd' and floattype != 'f')
  {
    throw miog_error("floattype should be one of 'f' and 'd' (in Geometry constructor)");
  }

  check_ldx_consistent();

  derived.reset(floattype);

  metric_co[0] = std::log2(static_cast<double>(k));
  metric_co[1] = std::log2(static_cast<double>(m)) - std::log2(static_cast<double>(n));
  metric_co[2] = std::log2(static_cast<double>(m)) + std::log2(static_cast<double>(n));

  metric_co[3] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::A]));
  metric_co[4] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::B]));
  metric_co[5] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::C]));

  // memory required for copying (an estimate)
  std::array<size_t, Mat::E::N> forPadCopy;
  for (auto emat : {Mat::E::A, Mat::E::B})
  {
    forPadCopy[emat] = get_uncoal(emat) * (get_coal(emat) + 16);
  }

  // This is not correct, now should workspace sizes be compared in the metric?
  size_t wsp0 = 0;
  for (auto& x : wSpaceSize)
  {
    wsp0 += x;
  }
  wSpaceSufficient[0] = forPadCopy[Mat::E::A] < wsp0;
  wSpaceSufficient[1] = forPadCopy[Mat::E::B] < wsp0;
  wSpaceSufficient[2] = 1 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wsp0;
  wSpaceSufficient[3] = 2 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wsp0;
  wSpaceSufficient[4] = 4 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wsp0;
}

std::map<std::string, std::vector<size_t>> get_key_val_map(std::string geometry_string)
{
  auto frags = stringutil::split(geometry_string, "_");
  std::map<std::string, std::vector<size_t>> key_val_map;
  for (auto& frag : frags)
  {
    auto key_val = stringutil::splitnumeric(frag);
    auto key     = std::get<0>(key_val);
    auto val     = std::get<1>(key_val);
    if (key_val_map.count(key) == 0)
    {
      key_val_map[key] = {val};
    }
    else
    {
      key_val_map[key].push_back(val);
    }
  }
  return key_val_map;
}

std::vector<size_t> safeat(std::map<std::string, std::vector<size_t>>& map, std::string key)
{
  if (map.count(key) == 0)
  {
    std::stringstream errm;
    errm << "Unrecognised key `";
    errm << key << "' in safeat of geometry";
    throw miog_error(errm.str());
  }
  return map.at(key);
}

Geometry::Geometry(std::string geometry_string)
{

  auto key_val_map = get_key_val_map(geometry_string);

  Geometry goldstandard_geometry(
    false, false, false, false, 100, 100, 100, 100, 100, 100, {100, 200}, 'f');
  std::string goldstandard_geometry_string = goldstandard_geometry.get_string();
  auto        goldstandard_map             = get_key_val_map(goldstandard_geometry_string);

  std::stringstream errm_ss;
  bool              good_string{true};
  for (auto& x : key_val_map)
  {
    if (x.first != "ws" && x.second.size() != 1)
    {
      errm_ss << "The key in the geometry string `" << x.first << "' is appears " << x.second.size()
              << " times.";
      good_string = false;
    }

    if (goldstandard_map.count(x.first) == 0)
    {
      errm_ss << "The key in the geometry string `" << x.first << "' is not valid.  ";
      good_string = false;
    }
  }

  for (auto& x : goldstandard_map)
  {
    if (key_val_map.count(x.first) == 0 && x.first != "ws")
    {
      errm_ss << "The geometry string should contain key `" << x.first
              << "', but does not. The only optional key is ws. ";
      good_string = false;
    }
  }

  if (good_string == false)
  {
    throw miog_error(errm_ss.str());
  }

  if (key_val_map.count("ws") == 0)
  {
    key_val_map["ws"] = {};
  }

  initialise(safeat(key_val_map, "colMaj")[0],
             safeat(key_val_map, "tA")[0],
             safeat(key_val_map, "tB")[0],
             safeat(key_val_map, "tC")[0],
             safeat(key_val_map, "lda")[0],
             safeat(key_val_map, "ldb")[0],
             safeat(key_val_map, "ldc")[0],
             safeat(key_val_map, "m")[0],
             safeat(key_val_map, "n")[0],
             safeat(key_val_map, "k")[0],
             safeat(key_val_map, "ws"),
             get_floattype(safeat(key_val_map, "f")[0]));
}

std::string Geometry::get_string() const { return get_networkconfig_string(); }

std::string Geometry::get_networkconfig_string() const
{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tX[Mat::E::C] << "_tA" << tX[Mat::E::A] << "_tB" << tX[Mat::E::B]
                        << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda"
                        << ldX[Mat::E::A] << "_ldb" << ldX[Mat::E::B] << "_ldc" << ldX[Mat::E::C];
  // append_ws_frag(geometry_stringstream);
  for (auto& x : wSpaceSize)
  {
    geometry_stringstream << "_ws" << x;
  }

  geometry_stringstream << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}

// void Geometry::append_ws_frag(std::stringstream & ss) const
//{
// if (wSpaceSize.size() == 0){
// ss << 0;
//}
// else {
// ss << wSpaceSize[0];
// for (auto i = 1; i < wSpaceSize.size(); ++i){
// ss << '.' << wSpaceSize[i];
//}
//}
//}

std::string Geometry::get_tabbed_string() const
{

  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC=" << tX[Mat::E::C] << " tA=" << tX[Mat::E::A]
                        << " tB=" << tX[Mat::E::B] << " colMaj=" << isColMajor
                        << " m=" << stringutil::get_char_padded(m, 6)
                        << " n=" << stringutil::get_char_padded(n, 6)
                        << " k=" << stringutil::get_char_padded(k, 6)
                        << " lda=" << stringutil::get_char_padded(ldX[Mat::E::A], 6)
                        << " ldb=" << stringutil::get_char_padded(ldX[Mat::E::B], 6)
                        << " ldc=" << stringutil::get_char_padded(ldX[Mat::E::C], 6) << " ws={ ";
  for (auto& x : wSpaceSize)
  {
    geometry_stringstream << x << ' ';
  }
  geometry_stringstream << '}';
  geometry_stringstream << " f=" << derived.float_size_bits;

  return geometry_stringstream.str();
}

size_t Geometry::get_padded_area(Mat::E M) const { return get_uncoal(M) * ldX[M]; }

// Safer would be compare via get_string(), assuming get_string() is comprehensive.
bool Geometry::operator==(const Geometry& rhs) const
{
  return (isColMajor == rhs.isColMajor && tX == rhs.tX && ldX == rhs.ldX && m == rhs.m &&
          n == rhs.n && k == rhs.k && wSpaceSize == rhs.wSpaceSize && floattype == rhs.floattype);
}

double Geometry::get_gflops(double extime) const { return (2. * m * n * k) / (1e9 * extime); }

bool Geometry::same_transposes(const Geometry& g2) const
{
  return ((tX == g2.tX) && (isColMajor == g2.isColMajor));
}

double Geometry::get_distance(const Geometry& g2) const
{

  double distance = 0;
  if (same_transposes(g2) == false)
  {
    distance = std::numeric_limits<double>::max();
  }

  else
  {

    for (unsigned i = 0; i < 6; ++i)
    {
      distance += std::abs(metric_co[i] - g2.metric_co[i]);
    }
    for (size_t x : {2, 4, 8})
    {
      for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
      {
        distance += 0.2 * ((ldX[emat] % x == 0) != (g2.ldX[emat] % x == 0));
      }
    }

    for (size_t x : {256, 512, 1024})
    {
      for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
      {
        distance += 0.2 * (std::min<size_t>(ldX[emat] % x, x - ldX[emat] % x) % 4 ==
                           std::min<size_t>(g2.ldX[emat] % x, x - g2.ldX[emat] % x) % 4);
      }
    }

    for (size_t i = 0; i < wSpaceSufficient.size(); ++i)
    {
      distance += 0.2 * (wSpaceSufficient[i] == g2.wSpaceSufficient[i]);
    }
  }

  // this is not correct : distance between workspace, unclear how to define
  distance += 1e-5 * (wSpaceSize != g2.wSpaceSize);

  return distance;
}

size_t get_total_workspace(const Geometry& gg, const Offsets& toff, size_t workspace_index)
{

  if (workspace_index >= gg.wSpaceSize.size())
  {
    std::stringstream ss;
    ss << "workspace_index (" << workspace_index << ") exceeds number of workspaces ("
       << gg.wSpaceSize.size() << '.';
    throw miog_error(ss.str());
  }

  if (gg.wSpaceSize.size() != toff.tails_vws.size())
  {
    std::stringstream ss;
    ss << "number of workspaces (" << gg.wSpaceSize.size()
       << ") does not equal number of tails in offset (" << toff.tails_vws.size() << ").";
    throw miog_error(ss.str());
  }

  auto sum_wspace = 0;
  // for (size_t i = 0; i  < gg.wSpaceSize.size()){
  sum_wspace += gg.wSpaceSize[workspace_index];
  sum_wspace += toff.offsets_vws[workspace_index];
  sum_wspace += toff.tails_vws[workspace_index];
  //}
  return sum_wspace;
}
}
