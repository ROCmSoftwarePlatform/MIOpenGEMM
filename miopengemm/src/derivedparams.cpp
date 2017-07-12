/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/tiling.hpp>
#include <miopengemm/macgrid.hpp>

namespace MIOpenGEMM
{
//namespace derivedparams
//{

size_t DerivedParams::get_target_ld(Mat::E emat_x) const { return at(emat_x).cw1_target_ldx; }

size_t get_target(size_t grid_size, size_t above_distance, size_t x)
{
  size_t to_grid_line =
    (x - above_distance) / grid_size + ((x - above_distance) % grid_size != 0);

  return grid_size * to_grid_line + above_distance;
}

size_t get_copy_pad(Mat::E emat_x)
{
  if (emat_x == Mat::E::A)
  {
    return 3;
  }
  else
  {
    return 6;
  }
}

void DerivedParams::reset_cw_params(Mat::E emat_x)
{
  if (emat_x == Mat::E::B && ptr_hp->at(Mat::E::A).vs[Chi::E::WOS] != Scratch::E::UNUSED &&
      adps.cw_n_elements == uninitialised_size_t)
  {
    throw miog_error("make sure reset acw1 params is called before reset_bcw1_params, we "
                     "need that "
                     "adps.cw1_target_ldx be set here in derivedparams reset of bcw1");
  }

  // simple copy with padding
  if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::COPY)
  {
    at(emat_x).cw1_smallest_possible_ldx =
      ptr_gg->coal_is_pll_k(emat_x) ? ptr_gg->k : ptr_gg->get_non_k_dim(emat_x);
    at(emat_x).cw1_target_ldx =
      get_target(16, get_copy_pad(emat_x), at(emat_x).cw1_smallest_possible_ldx);
    at(emat_x).cw_n_elements = at(emat_x).cw1_target_ldx * ptr_gg->get_uncoal(emat_x);
  }

  else if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
  {

    at(emat_x).cw2_n_elements_perp_unroll = at(emat_x).n_groups * at(emat_x).macro_tile_length;
    at(emat_x).cw_n_elements              = at(emat_x).cw2_n_elements_perp_unroll * ptr_gg->k;

    cw2_n_macro_tiles_pll_unroll = ptr_gg->k / ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR] +
                                   ((ptr_gg->k % ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR]) != 0);
  }

  else
  {
    throw miog_error("copy type is neither 1 nor 2, so can't be correct that "
                     "there's a call to reset_cw_params");
  }

  at(emat_x).cw_global_offset = (emat_x == Mat::E::B && ptr_hp->at(Mat::E::A).vs[Chi::E::WOS] != Scratch::E::UNUSED)
                                  ? at(Mat::E::A).cw_n_elements
                                  : 0;
}

void DerivedParams::reset_ga3_params()
{
  if (main_split_on_k == 1)
  {
    ga3_super_column_width = static_cast<size_t>(
      std::floor(std::sqrt(static_cast<double>(ptr_hp->at(Mat::E::C).vs[NonChi::E::NAW]) /
                           static_cast<double>(ptr_hp->at(Mat::E::C).vs[NonChi::E::ICE]))));
  }
  else if (main_split_on_k == 0)
  {
    ga3_super_column_width = static_cast<size_t>(
      std::floor(std::sqrt(static_cast<double>(ptr_hp->at(Mat::E::C).vs[NonChi::E::NAW]))));
  }
  else
  {
    throw miog_error("main_split_on_k is neither 0 nor 1, how can this be? "
                     "Logic error in reset_ga3_params");
  }
  ga3_last_super_column_width = bdps.n_groups % ga3_super_column_width;
}

std::tuple<bool, std::string> get_deriveability(const HyperParams& hp,
                                                const Geometry& gg)
{
  DerivedParams dp(hp, gg, "uninitialised");
  return dp.set_fragile();
}

DerivedParams::DerivedParams(const HyperParams& hp_,
                             const Geometry&                 gg_,
                             std::string                     s)
  : ptr_hp(&hp_), ptr_gg(&gg_)
{

  initialise_chis();

  if (s.compare("uninitialised") != 0)
  {
    throw miog_error("the only string with which a DerivedParams object can be "
                     "initialised is `uninitialised'");
  }
}

std::tuple<bool, std::string> DerivedParams::set_fragile()
{

  set_should_be_hyperparams();

  auto grid_size_tuple =
    macgrid::get_grid(ptr_hp->at(Mat::E::C).vs[NonChi::E::MAC], ptr_hp->at(Mat::E::C).vs[NonChi::E::SKW]);
  if (std::get<0>(grid_size_tuple) == false)
  {
    return std::make_tuple(false, std::get<1>(grid_size_tuple));
  }

  auto grid_size = std::get<2>(grid_size_tuple);

  at(Mat::E::A).macro_tile_length = grid_size[Mat::E::A] * ptr_hp->at(Mat::E::A).vs[Chi::E::MIC];
  at(Mat::E::B).macro_tile_length = grid_size[Mat::E::B] * ptr_hp->at(Mat::E::B).vs[Chi::E::MIC];

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {
    at(emat_x).preshift_final_tile =
      1 + (ptr_gg->get_non_k_dim(emat_x) - 1) % at(emat_x).macro_tile_length;
    at(emat_x).n_groups = ptr_gg->get_non_k_dim(emat_x) / at(emat_x).macro_tile_length +
                          (at(emat_x).preshift_final_tile != at(emat_x).macro_tile_length);
    at(emat_x).main_macro_tile_length_and_pad =
      at(emat_x).macro_tile_length + ptr_hp->at(emat_x).vs[Chi::E::PAD];

    at(emat_x).main_n_elements_in_padded_unroll =
      at(emat_x).main_macro_tile_length_and_pad * ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR];
  }

  main_macro_tile_area = at(Mat::E::B).macro_tile_length * at(Mat::E::A).macro_tile_length;
  main_micro_tile_area =
    ptr_hp->at(Mat::E::B).vs[Chi::E::MIC] * ptr_hp->at(Mat::E::A).vs[Chi::E::MIC];
  main_n_work_items_per_workgroup = main_macro_tile_area / main_micro_tile_area;

  size_t required_workspace = 0;

  std::stringstream set_status_ss;

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {
    // check - 3 : the macro tile is too tall
    if (ptr_gg->m < at(Mat::E::A).macro_tile_length)
    {
      set_status_ss << "m < aps.macro_tile_length, not considering this kernel. ";
    }

    // check - 4 : the macro tile is too wide
    else if (ptr_gg->n < at(Mat::E::B).macro_tile_length)
    {
      set_status_ss << "m < bps.macro_tile_length, not considering this kernel. ";
    }

    at(emat_x).n_elements_in_unroll =
      at(emat_x).macro_tile_length * ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR];
    at(emat_x).main_n_elements_to_load_per_workitem =
      at(emat_x).n_elements_in_unroll / main_n_work_items_per_workgroup;

    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      at(emat_x).cw2_n_elements_to_load_per_workitem =
        at(emat_x).n_elements_in_unroll / at(emat_x).cw2_local_work_size;
    }


    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] != Scratch::E::UNUSED)
    {
      reset_cw_params(emat_x);
      required_workspace += at(emat_x).cw_n_elements;
    }

    // check 0 : macro tile not too large
    if (ptr_gg->get_non_k_dim(emat_x) < at(emat_x).macro_tile_length)
    {
      set_status_ss << "ptr_gg->get_non_k_dim( " << Mat::M.name[emat_x] << " )  < at ( "
                    << Mat::M.name[emat_x] << " ).macro_tile_length, this means the tile is too big "
                                           "to work with  "
                    << Mat::M.name[emat_x] << " . not considering this kernel. ";
    }
  }

  // check -1 : enough workspace memory
  if (ptr_gg->workspace_size < required_workspace)
  {
    set_status_ss << "ptr_gg->workspace_size ( " << ptr_gg->workspace_size
                  << " ) is less then the required workspace ( " << required_workspace << " ). ";
  }

  if (set_status_ss.str() != "")
  {
    return std::make_tuple(false, set_status_ss.str());
  }

  // check 1 : n_work_items_per_workgroup divides n_elements_in_unroll for a and b  */

  auto is_div = [&set_status_ss, this](Mat::E emat_x, std::string which, size_t val) {

    if (at(emat_x).n_elements_in_unroll % val != 0)
    {
      set_status_ss << "this is not supported: " << which << " (" << val
                    << ") is not a factor of n_elements_in_(" << Mat::M.name[emat_x] << ")_unroll ("
                    << at(emat_x).n_elements_in_unroll << "). \n"
                    << "Consider rounding unroll up. ";
      return std::make_tuple<bool, std::string>(false, set_status_ss.str());
    }
    else
    {
      return std::make_tuple<bool, std::string>(true, {});
    }
  };

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    auto tup = is_div(emat_x, "main_n_work_items_per_workgroup", main_n_work_items_per_workgroup);
    if (std::get<0>(tup) == false)
    {
      return tup;
    }

    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      auto tup_cw =
        is_div(emat_x, "at(emat_x).cw2_local_work_size", at(emat_x).cw2_local_work_size);
      if (std::get<0>(tup_cw) == false)
      {
        return tup_cw;
      }
    }
  }

  // check 2 : tileability
  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    auto tup = tiling::get_tileability(at(emat_x).macro_tile_length,
                                       ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR],
                                       at(emat_x).main_n_elements_to_load_per_workitem);
    if (std::get<0>(tup) == false)
    {
      return tup;
    }

    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      tup = tiling::get_tileability(at(emat_x).macro_tile_length,
                                    ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR],
                                    at(emat_x).cw2_n_elements_to_load_per_workitem);
      if (std::get<0>(tup) == false)
      {
        return tup;
      }
    }
  }

  if (ptr_hp->at(Mat::E::C).vs[NonChi::E::UFO] == Binary::E::YES)
  {
    if (ptr_gg->k <= ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR])
    {
      return std::make_tuple(false, "UFO = yes, so UNR must be greater that k");
    }
  }

  // ran the gauntlet, returning deriveable is true
  return std::make_tuple(true, "");
}

void DerivedParams::initialise_chis()
{
  chis.resize(2);
  if (Mat::E::A > 2 || Mat::E::B > 2)
  {
    throw miog_error("In DeriverParams constructor, enums too large (strange)");
  }

  chis[Mat::E::A] = &adps;
  chis[Mat::E::B] = &bdps;
}


std::string get_tint(size_t memsize){
  
  std::string tint;
  if (memsize < std::pow(2, 16)){ // 2 bytes = 16 bits. 
    tint = "ushort";
  }
  
  else if (memsize < std::pow(2, 32)){ // 4 bytes = 32 bits. 
    tint = "unsigned";
  }
  
  else{
    tint = "size_t";
  }
  
  return tint;
  
}



DerivedParams::DerivedParams(const HyperParams& hp_, const Geometry& gg_)
  : ptr_hp(&hp_), ptr_gg(&gg_)
{

  initialise_chis();

  auto tup = set_fragile();

  if (std::get<0>(tup) == false)
  {
    throw miog_error("Failure to construct DerivedParams. Problem caught in set_fragile. It "
                     "is "
                     "recommended to run function ` derivable ' to check that a valid "
                     "DerivedParams can be constructed. The message returned in set_fragile "
                     "is : "
                     " " +
                     std::get<1>(tup));
  }

  // do the tiling
  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    tiling::set_tile_dimensions(at(emat_x).main_micro_tile_perp_unroll,
                                at(emat_x).main_micro_tile_pll_unroll,
                                at(emat_x).macro_tile_length,
                                ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR],
                                at(emat_x).main_n_elements_to_load_per_workitem,
                                ptr_hp->at(emat_x).vs[Chi::E::PLU] == 0);

    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      tiling::set_tile_dimensions(at(emat_x).cw2_micro_tile_perp_unroll,
                                  at(emat_x).cw2_micro_tile_pll_unroll,
                                  at(emat_x).macro_tile_length,
                                  ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR],
                                  at(emat_x).cw2_n_elements_to_load_per_workitem,
                                  at(emat_x).cw2_load_pll_to_unroll == 0);
    }
  }

  main_split_on_k      = ptr_hp->at(Mat::E::C).vs[NonChi::E::ICE] == 1 ? 0 : 1;
  main_does_beta_c_inc = main_split_on_k == 1 ? 0 : 1;

  if (ptr_hp->at(Mat::E::C).vs[NonChi::E::ICE] == 1)
  {
    infa = "n_work_items_per_c_elm is 1, should not be using atomics";
    fati = "n_work_items_per_c_elm is 1, should not be using atomics";
  }

  else
  {
    infa = ptr_gg->derived.float_size_bits == 32 ? "uint" : "ulong";
    fati = ptr_gg->derived.float_size_bits == 32 ? "atomic_cmpxchg" : "atom_cmpxchg";
  }

  pragma_unroll_string = ptr_hp->at(Mat::E::C).vs[NonChi::E::PUN] == 1 ? "#pragma unroll\n" : "";

  effective_k_varies_string = ptr_hp->at(Mat::E::C).vs[NonChi::E::UFO] == 0 ? "__K__" : "k_plus_offset";
  t_float                   = ptr_gg->derived.float_size_bits == 32 ? "float" : "double";

  main_n_work_groups = ptr_hp->at(Mat::E::C).vs[NonChi::E::ICE] *
                       ((ptr_gg->m / at(Mat::E::A).macro_tile_length) +
                        (ptr_gg->m % at(Mat::E::A).macro_tile_length != 0)) *
                       ((ptr_gg->n / at(Mat::E::B).macro_tile_length) +
                        (ptr_gg->n % at(Mat::E::B).macro_tile_length != 0));

  main_global_work_size = main_n_work_groups * main_n_work_items_per_workgroup;

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    at(emat_x).main_n_micro_in_macro =
      at(emat_x).macro_tile_length / ptr_hp->at(emat_x).vs[Chi::E::MIC];
    at(emat_x).main_n_micro_tiles_pll_unroll =
      ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR] / at(emat_x).main_micro_tile_pll_unroll;
    at(emat_x).main_c_interweave_stride =
      ptr_hp->at(emat_x).vs[Chi::E::MIW] == 0 ? 1 : at(emat_x).main_n_micro_in_macro;

    if (ptr_hp->at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      at(emat_x).cw2_n_micro_tiles_pll_unroll =
        ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR] / at(emat_x).cw2_micro_tile_pll_unroll;
      at(emat_x).cw2_n_micro_tiles_perp_unroll =
        at(emat_x).macro_tile_length / at(emat_x).cw2_micro_tile_perp_unroll;
    }
  }

  if (ptr_hp->at(Mat::E::C).vs[NonChi::E::GAL] == 3)
  {
    reset_ga3_params();
  }

  // these are hyper params, with a check if not optional ?
  main_use_edge_trick = (ptr_gg->m % at(Mat::E::A).macro_tile_length == 0 &&
                         ptr_gg->n % at(Mat::E::B).macro_tile_length == 0)
                          ? 0
                          : 1;
  main_final_fractional_unroll = (ptr_hp->at(Mat::E::C).vs[NonChi::E::UFO] == 1 ||
                                  ptr_gg->k % ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR] != 0)
                                   ? 1
                                   : 0;

  // set tints. TODO here.
  tints[Mem::E::A] = get_tint(ptr_gg->get_uncoal(Mat::E::A)*(ptr_gg->ldX[Mat::E::A])); // TODO : does UFO need increase here
  tints[Mem::E::B] = get_tint(ptr_gg->get_uncoal(Mat::E::B)*(ptr_gg->ldX[Mat::E::B]));
  tints[Mem::E::C] = get_tint(ptr_gg->get_uncoal(Mat::E::C)*(ptr_gg->ldX[Mat::E::C])); 
  tints[Mem::E::W] = get_tint(ptr_gg->workspace_size);
  tintk = get_tint(ptr_gg->k + 2*ptr_hp->at(Mat::E::C).vs[NonChi::E::UNR]); // TODO : make this tight and prove correct. 
  tshort = "ushort";
  

  tints[Mem::E::A] = "size_t";
  tints[Mem::E::B] = "size_t";
  tints[Mem::E::C] = "size_t";
  tints[Mem::E::W] = "size_t";
  tintk = "size_t";
  tshort = "size_t";
    
}




/* TODO : move to hyper params */
void DerivedParams::set_should_be_hyperparams()
{

  betac_local_work_size = 256;
  betac_work_per_thread = 2;

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    at(emat_x).cw1_local_work_size    = 256;
    at(emat_x).cw1_work_per_thread    = 2;
    at(emat_x).cw2_load_pll_to_unroll = 0;
    at(emat_x).cw2_local_work_size    = 64;
  }
}


size_t DerivedParams::get_stride(Mat::E emat_x,
                                   bool       pll_k,
                                   bool       is_macro,
                                   size_t   workspace_type_) const
{

  if (workspace_type_ == 0)
  {
    return get_stride_cw0(emat_x, pll_k);
  }

  else if (workspace_type_ == 1)
  {
    return get_stride_cw1(emat_x, pll_k);
  }

  else if (workspace_type_ == 2)
  {
    return get_stride_cw2(emat_x, pll_k, is_macro);
  }
  else
    throw miog_error("unrecognised workspace_type in get_strinde in derivedparams");
}

size_t DerivedParams::get_stride_cw0(Mat::E emat_x, bool pll_k) const
{
  return ptr_gg->coal_is_pll_k(emat_x) == pll_k ? 1 : ptr_gg->ldX.at(emat_x);
}

size_t DerivedParams::get_stride_cw1(Mat::E emat_x, bool pll_k) const
{
  return ptr_gg->coal_is_pll_k(emat_x) == pll_k ? 1 : at(emat_x).cw1_target_ldx;
}

size_t DerivedParams::get_stride_cw2(Mat::E emat_x, bool pll_k, bool is_macro) const
{
  if (is_macro == false)
  {
    return pll_k == true ? at(emat_x).macro_tile_length : 1;
  }
  else
  {
    return pll_k == true ? at(emat_x).macro_tile_length : ptr_gg->k;
  }
}
//}
}
