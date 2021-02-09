/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <miopengemm/alphagenerator.hpp>
#include <miopengemm/basegenerator.hpp>

/* TODO : reconsider half threads loading A to LDS, others B to LDS, even though
 * will only work in unusual cirmumstances (strides same, or tiles parallel
 * to contiguous). With normal form matrices, it should work in all situations  */
/* TODO : remove final barrier, not nec */
/* TODO : beta = 1 optimisation */
/* TODO : volatile int id to prevent unrolling (idea from website of Nugteren) */
/* TODO : play with restrict keyword */
/* TODO : consider idea of localA, localB being contiguous (float1)
 * and then "localC" can be within or an extension of this. then,
 * when writing to C, first write to localC, then make a contiguous write to
 * C from localC */
// TODO : experiment with this https://community.amd.com/thread/192119
// (to see when CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE works).

namespace MIOpenGEMM
{
namespace alphagen
{

class AlphaGenerator : public basegen::BaseGenerator
{

  private:
  // TODO : move to derived maybe
  std::vector<Mat::E> mata_matb;

  virtual void set_usage() override final
  {

    u_a     = (hp.sus[Mat::E::A].vs[Chi::E::WOS] == Scratch::E::UNUSED) ? true : false;
    u_b     = (hp.sus[Mat::E::B].vs[Chi::E::WOS] == Scratch::E::UNUSED) ? true : false;
    u_c     = true;
    u_w     = (not u_a or not u_b);
    u_alpha = true;
    u_beta  = dp.main_does_beta_c_inc;
  }

  public:
  AlphaGenerator(const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_)
    : basegen::BaseGenerator(hp_, gg_, dp_)
  {

    if (hp.sus[Mat::E::C].vs[NonChi::E::AFI] == Binary::E::YES)
    {
      mata_matb = {Mat::E::A, Mat::E::B};
    }
    else
    {
      mata_matb = {Mat::E::B, Mat::E::A};
    }
  }

  private:
  void append_group_allocation_string(std::stringstream& ss)
  {
    if (hp.sus[Mat::E::C].vs[NonChi::E::GAL] == GroupAllocation::E::BYCOL)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */
const TINTA group_id_a = group_id_xy % N_GROUPS_A;
const TINTB group_id_b = group_id_xy / N_GROUPS_A;
)";
    }

    else if (hp.sus[Mat::E::C].vs[NonChi::E::GAL] == GroupAllocation::E::BYROW)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */
const TINTA group_id_a = group_id_xy / N_GROUPS_B;
const TINTB group_id_b = group_id_xy % N_GROUPS_B;
)";
    }

    else if (hp.sus[Mat::E::C].vs[NonChi::E::GAL] == GroupAllocation::E::SUCOL)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 3 : allocation examples
* (if SUPER_COLUMN_WIDTH is 8, m = 3, and N_WORK_ITEMS_PER_C_ELM is 1) is done as follows
* |0   1  2  3  4  5  6  7| 24 25 26
* |8   9 10 11 12 13 14 15| 27 28 29
* |16 17 18 19 20 21 21 23| 30 31 32
*              
* if SUPER_COLUMN_WIDTH is 2 and N_WORK_ITEMS_PER_ELM is 3 it is done as follows
* | (0,   1,  2)  (3,  4,  5 )    |    
* | (6,   7,  8)  (9,  10, 11)    |    ...
* | (12, 13, 14)  (15, 16, 17)    |
*                .
*                .
* where the integers are work group numbers
* */  
TINTB group_id_b;
TINTA group_id_a;
TINTC wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_A);
)";

      std::string full_SUCOL_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_A;
)";

      std::string partial_SUCOL_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy  - (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A) / LAST_SUPER_COLUMN_WIDTH;
)";

      // super column width perfectly fits across B
      if (dp.ga3_last_super_column_width == 0)
      {
        ss << full_SUCOL_string;
      }

      else
      {

        // there is just one column
        if (dp.ga3_last_super_column_width == dp.at(Mat::E::B).n_groups)
        {
          ss << partial_SUCOL_string;
        }

        else
        {
          ss << '\n'
             << "if (group_id_xy < (N_GROUPS_B - "
                "LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A){";
          ss << full_SUCOL_string << "}\n";

          ss << "else{";
          ss << partial_SUCOL_string << "}\n";
        }
      }
    }

    else
    {
      std::stringstream err_ss;
      err_ss << "Invalid group_allocation parameter : " << hp.sus[Mat::E::C].vs[NonChi::E::GAL]
             << ". It should be one of 1/2/3.";
      throw miog_error(err_ss.str());
    }
  }

  void append_super_column_width_defn(std::stringstream& ss)
  {

    if (hp.sus[Mat::E::C].vs[NonChi::E::GAL] == 3)
    {

      ss << "\n\n"
         << "/* This variable defines the width of super-columns (we have "
            "GROUP_ALLOCATION 3). It "
            "is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) "
            "*/\n"
         << "#define SUPER_COLUMN_WIDTH " << dp.ga3_super_column_width;
      ss << "\n/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_B % SUPER_COLUMN_WIDTH  "
            "*/";
      ss << "\n#define LAST_SUPER_COLUMN_WIDTH " << dp.ga3_last_super_column_width;
    }
  }

  void append_split_on_k_vardecl_write_string(std::stringstream& ss)
  {
    if (dp.main_split_on_k != 0)
    {
      ss <<
        R"(
/* the following variables are used in implementing a basic atomic increment */
global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
TFLOAT previous_value; )"
         << '\n'
         << dp.infa << " newVal;\n"
         << dp.infa << " prevVal;"
         << "\n\n";
    }
  }

  void append_loop_var_bound_incr(std::stringstream& ss,
                                  std::string        varname,
                                  std::string        bound_string,
                                  std::string        increment_string,
                                  Mat::E             emat_x)
  {
    ss << "for (TINT" << Mat::M().name[emat_x] << ' ' << varname << " = 0; " << varname << " < "
       << bound_string << "; " << increment_string << ")";
  }

  void append_load_for_perp(Mat::E emat_x, std::stringstream& ss)
  {

    char X = Mat::M().name[emat_x];

    std::string bound_string = hp.sus[emat_x].vs[Chi::E::LIW] == 0
                                 ? std::string("MICRO_") + X + "_TILE_PERP_UNROLL"
                                 : std::string("MACRO_TILE_LENGTH_") + X;

    bound_string += "/VEW_";
    bound_string += X;

    std::stringstream incr_liw;
    incr_liw << "mu_perp_i += MACRO_TILE_LENGTH_" << X << "/MICRO_" << X << "_TILE_PERP_UNROLL";
    std::string increment_string =
      hp.sus[emat_x].vs[Chi::E::LIW] == 0 ? "++mu_perp_i" : incr_liw.str();

    append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string, emat_x);
  }

  void append_load_for_pll(Mat::E emat_x, std::stringstream& ss)
  {

    std::string bound_string =
      hp.sus[emat_x].vs[Chi::E::LIW] == 0
        ? std::string("MICRO_") + Mat::M().name[emat_x] + "_TILE_PLL_UNROLL"
        : "UNROLL";
    std::string increment_string =
      hp.sus[emat_x].vs[Chi::E::LIW] == 0
        ? "++mu_pll_i"
        : std::string("mu_pll_i += UNROLL/MICRO_") + Mat::M().name[emat_x] + "_TILE_PLL_UNROLL";
    append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string, emat_x);
  }

  void append_final_write_element(std::stringstream& ss,
                                  size_t             atomic_increment,
                                  size_t             with_beta_scaling,
                                  size_t             with_alpha_increment)
  {

    std::string dima_index = hp.sus[Mat::E::A].vs[Chi::E::MIW] == 0
                               ? "dima"
                               : "(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v";  //
    std::string dimb_index = hp.sus[Mat::E::B].vs[Chi::E::MIW] == 0
                               ? "dimb"
                               : "(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v";

    // a good place to break kernel to check error checking.
    // make this* 1.11101242345 for example

    std::string alpha_scaled = "alpha*rC[" + dima_index + "][" + dimb_index + "]";
    ss << "\nindex =  STRIDE_PLL_M_C*(write_start_a + dima) + STRIDE_PLL_N_C*(write_start_b + "
          "dimb) ;\n";

    if (with_beta_scaling != 0)
    {
      ss << "if (beta >= 0 && beta <= 0){\nc[index] = 0; \n}\n"
         << "else {\nc[index] *= beta;}\n";
    }

    if (with_alpha_increment != 0)
    {
      ss << '\n';
      if (atomic_increment == 0)
      {
        ss << "c[index] += " << alpha_scaled << +";\n";
      }

      else
      {
        ss << "ptr_to_c_elm = c + index;\n"
           << "do {\n"
           << "previous_value = *ptr_to_c_elm;\n"
           << "prevVal = as_" << dp.infa << "(previous_value);\n"
           << "newVal = as_" << dp.infa << "(" << alpha_scaled << " + previous_value);\n"
           << "} while (" << dp.fati << "(( __global " << dp.infa
           << "*)(ptr_to_c_elm), prevVal, newVal) != prevVal);";
      }
    }
  }

  void append_for_loops_for_c_write_open(std::stringstream& ss)
  {

    ss << "\n/* loops for writing to c */\n";

    for (auto emat : mata_matb)
    {
      std::string X(1, Mat::M().name[emat]);
      char        x     = Mat::M().lcase_name[emat];
      std::string dimxi = "dim" + std::string(1, x) + "i";

      ss << dp.pragma_unroll_string;
      append_loop_var_bound_incr(
        ss,
        dimxi,
        hp.sus[emat].vs[Chi::E::MIW] == 0 ? "MICRO_TILE_LENGTH_" + X + "/VEW_" + X
                                          : "MACRO_TILE_LENGTH_" + X + "/VEW_" + X,
        hp.sus[emat].vs[Chi::E::MIW] == 0 ? "++" + dimxi : dimxi + " += N_MICRO_IN_MACRO_" + X,
        Mat::E::A);
      ss << " {\n";
    }

    for (auto emat : mata_matb)
    {
      std::string X(1, Mat::M().name[emat]);
      char        x     = Mat::M().lcase_name[emat];
      std::string dimxi = "dim" + std::string(1, x) + "i";

      ss << dp.pragma_unroll_string;
      append_loop_var_bound_incr(ss, dimxi + "_v", "VEW_" + X, "++" + dimxi + "_v", emat);
      ss << " {\n";
    }

    ss << "TINTB dimb = dimbi*VEW_B + dimbi_v;\n";
    ss << "TINTA dima = dimai*VEW_A + dimai_v;\n";
  }

  void append_for_loops_for_c_write_close(std::stringstream& ss) { ss << "\n}\n}\n}\n}\n"; }

  void append_check_wrapped_if_clause_open(std::stringstream& ss)
  {
    ss <<
      R"(
/* catching the write cases */
if (



/* B overflow, but not A edge */
((write_start_b + dimb >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1)) && group_id_a  !=  (N_GROUPS_A - 1 )) ||

/* A overflow, but not B edge */
((write_start_a + dima >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)) && group_id_b  !=  (N_GROUPS_B - 1 )) ||


/* A edge and B edge and A overflow and B overflow */
(
group_id_a == (N_GROUPS_A - 1)   && 
group_id_b == (N_GROUPS_B - 1)   && 
write_start_b + dimb >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1) && 
write_start_a + dima >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)
)){
)";
  }

  void append_check_wrapped_if_clause_close(std::stringstream& ss) { ss << "\n}"; }

  void append_checked_wrapped_loops_from_bools(std::stringstream& ss,
                                               size_t             with_check,
                                               size_t             atomic_increment,
                                               size_t             with_beta_scaling,
                                               size_t             with_alpha_increment)
  {

    append_for_loops_for_c_write_open(ss);
    if (with_check != 0)
    {
      append_check_wrapped_if_clause_open(ss);
      append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
      append_check_wrapped_if_clause_close(ss);
    }

    else
    {
      append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
    }
    append_for_loops_for_c_write_close(ss);
  }

  void append_final_write_loops(std::stringstream& ss, size_t with_check)
  {
    if (dp.main_split_on_k == 0)
    {
      append_checked_wrapped_loops_from_bools(ss, with_check, 0, 1, 1);
    }

    else
    {
      append_checked_wrapped_loops_from_bools(ss, with_check, 1, 0, 1);
    }
  }

  void append_final_write_loops_no_check(std::stringstream& ss) { append_final_write_loops(ss, 0); }

  void append_final_write_loops_with_check(std::stringstream& ss)
  {
    append_final_write_loops(ss, 1);
  }

  void append_k_remaining_string(std::stringstream& ss)
  {
    ss << '\n' << "TSHORT k_remaining = " << dp.effective_k_varies_string << " % UNROLL;";
  }

  // simple for loops. Could consider unrolling like Cobalt, but for the moment
  // I use the optional pragma unroll
  void append_load_ab_into_LDS_string(std::stringstream& ss,
                                      size_t             final_unroll,
                                      size_t             special_first_unroll)
  {

    for (auto emat : mata_matb)
    {
      append_load_into_LDS_string(emat, ss, final_unroll, special_first_unroll);
    }

    ss <<
      R"(
/* make sure all loads from LDS memory have completed */
barrier(CLK_LOCAL_MEM_FENCE); )";
  }

  // simple for loops. Could consider unrolling like Cobalt,
  // but for the moment I use the optional pragma unroll
  void append_load_into_LDS_string(Mat::E             emat_x,
                                   std::stringstream& ss,
                                   size_t             final_unroll,
                                   size_t             special_first_unroll)
  {

    char X = Mat::M().name[emat_x];
    char x = Mat::M().lcase_name[emat_x];

    std::string n_jumps_string =
      (dp.main_split_on_k == 0 || (hp.sus[Mat::E::C].vs[NonChi::E::IWI] == Binary::E::NO))
        ? "UNROLL"
        : "G_UNROLL";

    if (final_unroll != 0 && special_first_unroll != 0)
    {
      throw miog_error("From get_load_ab_into_LDS_string > It is not possible "
                       "for this to be both "
                       "a `special_first_unroll' and a `final_unroll'. This is "
                       "a logic error, "
                       "broken alg, come and sort it out");
    }

    std::stringstream ss_value_to_get;
    std::stringstream ss_comment;

    std::stringstream basic_to_get_ss;
    basic_to_get_ss << x << "_vec[(mu_pll_i*STRIDE_PLL_K_" << X << " + VEW_" << X
                    << "*mu_perp_i*STRIDE_PERP_K_" << X << ")/VEW_" << X << "]";

    if (final_unroll == 1 || special_first_unroll == 1)
    {
      std::string condition       = final_unroll == 1 ? " < k_remaining " : " >= unroll_offset";
      std::string special_comment = final_unroll == 1 ? "(ignoring tail)" : "(ignoring prepend)";
      ss_comment << "/* load final bit of data from " << x << " into LDS, less than a full unroll "
                 << special_comment << " */";
      ss_value_to_get << "(" << x << "_offset_pll_unroll + mu_pll_i) " << condition << " ? "
                      << basic_to_get_ss.str() << " : 0;";
    }

    else
    {
      ss_comment << "/* load data from " << x << " into LDS */";
      ss_value_to_get << basic_to_get_ss.str() << ';';
    }

    ss << '\n' << ss_comment.str() << '\n' << dp.pragma_unroll_string;
    append_load_for_perp(emat_x, ss);
    ss << " {\n" << dp.pragma_unroll_string;
    append_load_for_pll(emat_x, ss);
    ss << " {\n"
       << "local" << X << "[MACRO_TILE_LENGTH_" << X << "_AND_PAD/VEW_" << X << "*(" << x
       << "_offset_pll_unroll + mu_pll_i) + " << x << "_offset_perp_unroll_v + mu_perp_i] = \n"
       << ss_value_to_get.str() << '\n'
       << "}\n"
       << "}\n";

    if (final_unroll == 0)
      ss << x << "_vec += "
         << "(STRIDE_PLL_K_" << X << "*" << n_jumps_string << ")/VEW_" << X << ";\n";

    ss << '\n';
  }

  std::string get_c_work_item_next(Mat::E emat_x)
  {

    return (hp.sus[emat_x].vs[Chi::E::MIW] != 0)
             ? std::string("VEW_") + Mat::M().name[emat_x]
             : (std::string("MICRO_TILE_LENGTH_") + Mat::M().name[emat_x]);
  }

  // We previously had a variable unroll_the_math_section = False.
  // Experiments with unroll_the_math_section suggest that it's a bad idea.
  void append_math_section(std::stringstream& ss, size_t use_k_remaining)
  {

    std::string number_of_unrolls = use_k_remaining == 0 ? "UNROLL" : "k_remaining";
    ss << "\nfor (TSHORT u = 0; u < " << number_of_unrolls << "; ++u){\n";

    for (Mat::E emat : mata_matb)
    {
      append_load_to_register_string(emat, ss);
    }

    ss << '\n';
    append_compute_string(ss);

    ss << "}\n";
  }

  void append_relocate_load_math_string(std::stringstream& ss,
                                        size_t             final_unroll,
                                        size_t             special_first_unroll)
  {
    if (final_unroll != 0 && special_first_unroll != 0)
    {
      throw miog_error("From get_relocate_load_math_string : It is not "
                       "possible for this to be "
                       "both a `special_first_unroll' and a `final_unroll'. "
                       "This is a logic error, "
                       "broken alg, come and sort it out");
    }

    append_load_ab_into_LDS_string(ss, final_unroll, special_first_unroll);

    ss << '\n';
    for (Mat::E emat_x : mata_matb)
    {
      char X = Mat::M().name[emat_x];
      char x = Mat::M().lcase_name[emat_x];

      ss << '\n'
         << "l" << X << " = local" << X << " + micro_id_" << x << "*"
         << get_c_work_item_next(emat_x) << "/VEW_" << X << ";";
    }

    ss << '\n';

    append_math_section(ss, final_unroll);
    ss <<
      R"(
/* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
barrier(CLK_LOCAL_MEM_FENCE); )";
  }

  void append_final_unroll_string(std::stringstream& ss)
  {

    if (dp.main_split_on_k == 0)
    {
      ss << '\n';
      append_relocate_load_math_string(ss, 1, 0);
      ss << '\n';
    }
    else
    {

      ss <<
        R"(
/* There is one workgroup which will process the remainder (less that UNROLL) */)";

      if (hp.sus[Mat::E::C].vs[NonChi::E::IWI] == Binary::E::YES)
      {
        ss << R"(
/* With ICE interwoven (IWI is YES), this workgroup is the last with 1 more */
if (group_id_z == n_work_groups_with_1_more && k_remaining > 0){
    )";
      }
      else
      {
        ss << R"(
/* With ICE not-interwoven (IWI is NO), this is the last group */
if ((group_id_z == N_WORK_ITEMS_PER_C_ELM - 1) && k_remaining > 0){
    )";
      }

      append_relocate_load_math_string(ss, 1, 0);
      ss << "\n}\n";
    }
  }

  void append_first_unroll_block(std::stringstream& ss)
  {
    if (hp.sus[Mat::E::C].vs[NonChi::E::UFO] != 0)
    {
      ss << "\n\n/* This is where the first unroll will be performed. "
            "Identical to what is in the "
            "main while, but with zero buffering.  */";
      if (dp.main_split_on_k == 0)
      {
        ss << '\n';
        append_relocate_load_math_string(ss, 0, 1);
        ss << "\n--n_unrolls_remaining;\n";
      }
      else
      {
        ss << "\nif (group_id_z == 0){\n";
        append_relocate_load_math_string(ss, 0, 1);
        ss << "\n--n_unrolls_remaining;\n}";
      }
    }
  }

  void append_compute_string(std::stringstream& ss)
  {

    for (auto emat : mata_matb)
    {
      char x = Mat::M().lcase_name[emat];
      char X = Mat::M().name[emat];
      ss << dp.pragma_unroll_string << "for (TSHORT dim" << x << " = 0; dim" << x
         << " < MICRO_TILE_LENGTH_" << X << "; ++dim" << x << "){\n";
    }

    if (hp.sus[Mat::E::C].vs[NonChi::E::MAD] == Binary::E::NO)
    {
      ss << "rC[dima][dimb] += rA[dima]*rB[dimb];   \n}\n}\n";
    }
    else
    {
      ss << "rC[dima][dimb] = mad(rA[dima], rB[dimb], rC[dima][dimb]);    \n}\n}\n";
    }
  }

  void append_load_to_register_string(Mat::E emat_x, std::stringstream& ss)
  {
    char X = Mat::M().name[emat_x];

    ss << '\n' << dp.pragma_unroll_string;
    ss << "for (TSHORT i = 0; i < MICRO_TILE_LENGTH_" << X << "/VEW_" << X << "; ++i){\n";

    if (hp.sus[emat_x].vs[Chi::E::VEW] != 1)
    {
      for (unsigned j = 0; j < hp.sus[emat_x].vs[Chi::E::VEW]; ++j)
      {
        ss << "r" << X << "[VEW_" << X << "*i + " << j << "] = l" << X << "["
           << "i*"
           << "C_INTERWEAVE_STRIDE_" << X << "].s" << j << ";\n";
      }
    }
    else
    {
      ss << "r" << X << "[i] = l" << X << "[i*C_INTERWEAVE_STRIDE_" << X << "];\n";
    }
    ss << "}\n";

    ss << "l" << X << " += MACRO_TILE_LENGTH_" << X << "_AND_PAD/VEW_" << X << ";\n";
  }

  void append_group_allocation_defn_string(std::stringstream& ss)
  {
    ss << "#define GROUP_ALLOCATION " << hp.sus[Mat::E::C].vs[NonChi::E::GAL] << '\n';
    if (hp.sus[Mat::E::C].vs[NonChi::E::GAL] == 3)
    {
      ss << "/* this variable is declared because we have GROUP_ALLOCATION "
            "type 3. */\n";
      ss << "/* It should define how many workgroups we expect to have active "
            "simulantaneuosly. "
            "*/\n";
      ss << "#define N_TARGET_ACTIVE_WORKGROUPS " << hp.sus[Mat::E::C].vs[NonChi::E::NAW] << '\n';
    }
  }

  void append_final_write_all(std::stringstream& ss)
  {

    if (dp.main_use_edge_trick == 0)
    {
      ss << '\n';
      append_final_write_loops_no_check(ss);
    }

    else
    {

      std::array<std::string, Mat::E::N> cond_ab;
      size_t nconds = 0;
      for (Mat::E emat : mata_matb)
      {
        char X        = Mat::M().name[emat];
        char x        = Mat::M().lcase_name[emat];
        cond_ab[emat] = "";
        if (dp.at(emat).preshift_final_tile != dp.at(emat).macro_tile_length)
        {
          std::stringstream soo;
          soo << "(group_id_" << x << " != N_GROUPS_" << X << " - 1)";
          cond_ab[emat] = soo.str();
          ++nconds;
        }
      }

      if (nconds == 0)
      {
        append_final_write_loops_no_check(ss);
      }

      else
      {

        if (dp.main_use_edge_trick == 0)
        {
          throw miog_error("in alphagenerator, dp.main_use_edge_trick == 0. "
                           "however, non-perfectly tilable");
        }

        ss << "\n/* the case where this is not an edge tile : will write to "
              "all cells */ \n";
        if (nconds == 2)
        {
          ss << "if (" << cond_ab[Mat::E::B] << " && " << cond_ab[Mat::E::A] << "){ \n";
        }

        else
        {
          for (auto emat : mata_matb)
          {
            if (cond_ab[emat] != "")
            {
              ss << "if  " << cond_ab[emat] << "{ \n";
            }
          }
        }

        append_final_write_loops_no_check(ss);
        ss << "\n}";

        ss << "\n\nelse{";
        append_final_write_loops_with_check(ss);
        ss << "\n}";
      }
    }
  }

  void append_split_on_k_defns_string(std::stringstream& ss)
  {
    if (dp.main_split_on_k != 0)
    {
      ss <<
        R"(/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL )"
         << hp.sus[Mat::E::C].vs[NonChi::E::ICE] * hp.sus[Mat::E::C].vs[NonChi::E::UNR]
         << " // N_WORK_ITEMS_PER_C_ELM*UNROLL";
    }
  }

  void append_group_id_defns(std::stringstream& ss)
  {
    if (dp.main_split_on_k == 0)
    {
      ss << "\nconst TINTC group_id_xy = get_group_id(0);\n";
    }
    else
    {
      ss <<
        R"(
const TINTC group_id = get_group_id(0);
const TINTC group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
const TSHORT group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;
)";
    }
  }

  void append_stride_c_defn(std::stringstream& ss)
  {

    size_t transposed_xor_is_col_major = (gg.tX[Mat::E::C] + gg.isColMajor) % 2;
    ss << "#define STRIDE_PLL_M_C " << (transposed_xor_is_col_major == 1 ? 1 : gg.ldX[Mat::E::C])
       << '\n';
    ss << "#define STRIDE_PLL_N_C " << (transposed_xor_is_col_major == 0 ? 1 : gg.ldX[Mat::E::C])
       << '\n';
  }

  void append_n_unrolls_remaining_string(std::stringstream& ss)
  {

    if (dp.main_split_on_k == 0)
    {
      ss << "\nint n_unrolls_remaining = " << dp.k_effective_div_UNROLL << ";";
    }

    else
    {
      ss << "\n/* a certain number of work groups process one more unroll. "
            "Note that with UFO = 1, "
            "this depends on column */";
      ss << "\nconst int n_work_groups_with_1_more = (" << dp.k_effective_mod_G_UNROLL
         << ") / UNROLL; \n";
      ss << "\n/* branching between work groups : some wgs have 1 more unroll "
            "to process. */\n";
      ss << "int n_unrolls_remaining = (" << dp.k_effective_div_G_UNROLL;
      ss << ") ";

     // To avoid the compiler outsmarting us when n_work_groups_with_1_more is zero 
     // [tautological-unsigned-zero-compare], we avoid checking if unsigned integers
     // are less than zero. 
     if (dp.k_effective_mod_G_UNROLL > 0){
      ss << " +  (group_id_z < n_work_groups_with_1_more)";
     }
     ss << ";";
    }
  }

  void append_c_offset_string(std::stringstream& ss)
  {

    ss << R"(

/* In OpenCL, host code does not have access to raw data pointers. */
/* Host code works with cl_mem objects, which encapsulate and hide raw points. */
/* For this reason, host code CANNOT simply increment pointers to data, */
/* as one can do with pointers for CPU gemm, or cublas gemm.*/

c += c_offset;
)";
  }

  void append_id_string_nonsym(std::stringstream& ss)
  {
    ss << "const TSHORT local_id = (TSHORT)(get_local_id(0));\n";
    append_group_id_defns(ss);

    ss << "/* Define which part of the C macro-tile this thread will process: "
       << MicroAllocation::M().name[hp.sus[Mat::E::C].vs[NonChi::E::MIA]] << "*/\n";

    if (hp.sus[Mat::E::C].vs[NonChi::E::MIA] == MicroAllocation::E::BYA)
      ss << R"(
const TSHORT micro_id_a = local_id % N_MICRO_IN_MACRO_A;
const TSHORT micro_id_b = local_id / N_MICRO_IN_MACRO_A;

)";

    else if (hp.sus[Mat::E::C].vs[NonChi::E::MIA] == MicroAllocation::E::BYB)
    {
      ss << R"(
const TSHORT micro_id_b = local_id % N_MICRO_IN_MACRO_B;
const TSHORT micro_id_a = local_id / N_MICRO_IN_MACRO_B;

)";
    }

    else
    {
      throw miog_error("unrecognised MicroAllocation");
    }

    append_group_allocation_string(ss);

    if (hp.sus[Mat::E::C].vs[NonChi::E::UFO] != 0)
    {
      ss <<
        R"(
/* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
TSHORT unroll_offset = (13*group_id_a + 7*group_id_b)%UNROLL;
TINTK k_plus_offset = KV__ + unroll_offset;
)";
    }
  }

  void append_id_string_sym(std::stringstream& ss, Mat::E emat_x)
  {

    char X = Mat::M().name[emat_x];
    char x = Mat::M().lcase_name[emat_x];

    ss << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* LDS memory */\n";
    ss << "__local "
       << "TVFLOAT" << X << " local" << X << "[N_ELEMENTS_IN_PADDED_" << X << "_UNROLL"
       << "/VEW_" << X << "];\n";
    if (emat_x == Mat::E::A)
      ss << "/* jumping pointer to locate the LDS to load into register memory "
            "*/\n";
    ss << "__local const TVFLOAT" << X << " * l" << X << ";\n";
    if (emat_x == Mat::E::A)
      ss << "/* register memory */ \n";
    ss << "TFLOAT r" << X << "[MICRO_TILE_LENGTH_" << X << "];\n";
    if (emat_x == Mat::E::A)
      ss << "/* Define which part of the C macro-tile this thread will process "
            "(% / or / % ? "
            "doesn't seem to make much difference) */\n";
    ss << "TINT" << X << " write_macro_tile_start_" << x << " = group_id_" << x
       << "*MACRO_TILE_LENGTH_" << X << "; \n";
    if (dp.main_use_edge_trick != 0)
    {
      if (emat_x == Mat::E::A)
        ss << "/* tile on edge : pulling it in so no C overflow */\n";
      ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
      ss << "write_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X
         << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
      ss << "}\n";
    }
    ss << "const TINT" << X << " write_start_" << x << " = write_macro_tile_start_" << x
       << " + micro_id_" << x << "*(" << get_c_work_item_next(emat_x) << "/1);\n";

    ss << "\n\n\n";

    if (hp.sus[emat_x].vs[Chi::E::WOS] == Scratch::E::COPY ||
        hp.sus[emat_x].vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      if (emat_x == Mat::E::A)
        ss << "/* from workspace */\n";
      ss << "const TFLOAT * restrict " << x << " = w + w_offset + GLOBAL_OFFSET_" << X << ";\n";
    }

    else
    {
      ss << x << " += " << x << "_offset;\n";
    }

    if (emat_x == Mat::E::A)
      ss << "/* Define what of A this thread will load from unroll tile in "
            "global to LDS (% / or / "
            "% ? looks like no difference ) */\n";
    ss << "const TINT" << X << " pll_unroll_" << x << "_load_id = local_id % N_MICRO_" << X
       << "_TILES_PLL_UNROLL;\n";
    ss << "const TINT" << X << " perp_unroll_" << x << "_load_id = local_id / N_MICRO_" << X
       << "_TILES_PLL_UNROLL;\n";

    if (emat_x == Mat::E::A)
      ss << "/* Define which part of A this thread will read from (% / "
            "or / % ? doesn't "
            "seem to make much difference) */\n";
    ss << "TINT" << X << " read_macro_tile_start_" << x << " = group_id_" << x
       << "*MACRO_TILE_LENGTH_" << X << "; \n";
    if (dp.main_use_edge_trick != 0 && hp.sus[emat_x].vs[Chi::E::WOS] != Scratch::E::NFORM)
    {
      if (emat_x == Mat::E::A)
        ss << "/* tile on edge and A is not normal form: pulling in read zone "
              "so no C overflow */\n";
      ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
      ss << "read_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X
         << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
      ss << "}\n";
    }

    if (emat_x == Mat::E::A)
      ss << "/* move to corner of the region required by the macro tile */\n";
    ss << x << " += read_macro_tile_start_" << x << "*MACRO_STRIDE_PERP_K_" << X << ";\n";

    if (dp.main_split_on_k != 0)
    {
      if (emat_x == Mat::E::A)
        ss << R"(/* a points to top left of region required, but this work group  */
/* might not process the whole of a. So turn 90 and move to the start for this wg */
)";
      if (hp.sus[Mat::E::C].vs[NonChi::E::IWI] == Binary::E::NO)
      {
        if (emat_x == Mat::E::A)
        {
          ss << "/* IWI is NO, ICE is not interwoven */\n";
        }
        ss << x << " += (1 + (" << dp.k_effective_div_G_UNROLL
           << "))*UNROLL*group_id_z*STRIDE_PLL_K_" << X << ";\n";
        if (emat_x == Mat::E::A)
        {
          ss << "/*The last couple of groups (large group_id_z) will process 1 fewer unroll */\n";
        }
        ss << "\nif (group_id_z >  n_work_groups_with_1_more){\n";
        ss << x << " -= UNROLL*(group_id_z - n_work_groups_with_1_more)*STRIDE_PLL_K_" << X
           << ";\n}\n";
      }
      else if (hp.sus[Mat::E::C].vs[NonChi::E::IWI] == Binary::E::YES)
      {
        if (emat_x == Mat::E::A)
        {
          ss << "/* IWI is YES, ICE is interwoven */\n";
        }
        ss << x << " += UNROLL*group_id_z*STRIDE_PLL_K_" << X << ";\n";
      }
      else
      {
        std::stringstream errm;
        errm << NonChi::M().name[NonChi::E::IWI] << " should be NO (0) or YES (1), not "
             << hp.sus[Mat::E::C].vs[NonChi::E::IWI] << '.';
        throw miog_error(errm.str());
      }
    }

    if (hp.sus[Mat::E::C].vs[NonChi::E::UFO] != 0)
    {
      if (emat_x == Mat::E::A)
      {
        ss << "/* UFO != 0, so offsetting the unroll */\n";
      }

      ss << x << " -= unroll_offset*STRIDE_PLL_K_" << X << ";\n";
    }

    std::string str_n_pll("");
    std::string str_n_perp("");
    std::string str_n_perp_v("");
    if (hp.sus[emat_x].vs[Chi::E::LIW] == 0)
    {
      str_n_pll    = std::string("MICRO_") + X + "_TILE_PLL_UNROLL *";
      str_n_perp   = std::string("MICRO_") + X + "_TILE_PERP_UNROLL *";
      str_n_perp_v = std::string("MICRO_") + X + "_TILE_PERP_UNROLL/VEW_" + X + " *";
    }
    if (emat_x == Mat::E::A)
      ss << "/* make the micro adjustments (A) for the thread, getting ready "
            "to load */\n";
    ss << "const TINT" << X << " " << x << "_offset_pll_unroll = " << str_n_pll << " pll_unroll_"
       << x << "_load_id;\n";

    if (emat_x == Mat::E::A)
    {
      ss << "/* the offset in vector-floats perp to unroll */\n";
    }

    ss << "const TINT" << X << " " << x << "_offset_perp_unroll_v = " << str_n_perp_v
       << " perp_unroll_" << x << "_load_id;\n";

    ss << x << " += "
       << "STRIDE_PLL_K_" << X << " * " << x << "_offset_pll_unroll;\n";

    if (emat_x == Mat::E::A)
    {
      ss << "/* vectorised version of a */\n";
    }

    ss << "const __global TVFLOAT" << X << " * " << x << "_vec = (const __global TVFLOAT" << X
       << " * )" << x << ";\n";

    ss << x << "_vec += "
       << "STRIDE_PERP_K_" << X << " * " << x << "_offset_perp_unroll_v;\n";

    ss << '\n';
  }

  void append_transpose_note(std::stringstream& ss)
  {
    ss << R"(
/* A note on how transposes isColMajor  effect the kernel generated: very little. */
/* just via STRIDE_PLL_K_{A,B}, STRIDE_PERP_K_{A,B}, STRIDE_PLL_M_C, STRIDE_PERP_M_C */
)";
  }

  void add_predefine_chiral(Mat::E emat_x, std::stringstream& ss)
  {

    char x = Mat::M().name[emat_x];

    auto defcom = [emat_x, &ss](std::string&& comment) {
      if (emat_x == Mat::E::A)
        ss << "/*"
           << " " << comment << " : */\n";
    };

    bool withcomments = emat_x == Mat::E::A;

    bool with_x_in_name = true;
    append_unroll_block_geometry(emat_x, ss, withcomments, with_x_in_name);

    append_stride_definitions(
      emat_x, ss, hp.sus[emat_x].vs[Chi::E::WOS], withcomments, "", with_x_in_name);

    if (emat_x == Mat::E::A)
      ss << "/* vector float type */\n";
    ss << "#define TVFLOAT" << x << " " << dp.t_float;
    if (hp.sus[emat_x].vs[Chi::E::VEW] != 1)
      ss << hp.sus[emat_x].vs[Chi::E::VEW];
    ss << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* vector width */\n";
    ss << "#define VEW_" << x << "  " << hp.sus[emat_x].vs[Chi::E::VEW];
    ss << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* micro tiles define the pattern of C that individual threads "
            "process */\n";
    ss << "#define MICRO_TILE_LENGTH_" << x << " " << hp.sus[emat_x].vs[Chi::E::MIC] << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* the amount of padding of " << x
         << " in LDS (local) memory, to avoid bank comflicts */\n";
    ss << "#define PAD_LDS_" << x << "  " << hp.sus[emat_x].vs[Chi::E::PAD] << '\n';
    if (emat_x == Mat::E::A)
      ss << "/* whether loading of " << x << " from global should try to be long in direction of "
                                             "unroll (1) or perpendicular to it (0) */\n";
    ss << "#define WORK_ITEM_LOAD_" << x << "_PLL_TO_UNROLL " << hp.sus[emat_x].vs[Chi::E::PLU]
       << '\n';
    defcom("MACRO_TILE_LENGTH_A + PAD_LDS_A");
    ss << "#define MACRO_TILE_LENGTH_" << x << "_AND_PAD "
       << dp.at(emat_x).main_macro_tile_length_and_pad << '\n';

    defcom("MACRO_TILE_LENGTH_A_AND_PAD * UNROLL");
    ss << "#define N_ELEMENTS_IN_PADDED_" << x << "_UNROLL "
       << dp.at(emat_x).main_n_elements_in_padded_unroll << '\n';

    defcom("N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP");
    ss << "#define N_ELEMENTS_OF_" << x << "_TO_LOAD_PER_WORKITEM "
       << dp.at(emat_x).main_n_elements_to_load_per_workitem << '\n';
    defcom("MACRO_TILE_LENGTH_A / MICRO_TILE_LENGTH_A");
    ss << "#define N_MICRO_IN_MACRO_" << x << "  " << dp.at(emat_x).main_n_micro_in_macro << '\n';
    defcom("MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = "
           "N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM");
    ss << "#define MICRO_" << x << "_TILE_PLL_UNROLL " << dp.at(emat_x).main_micro_tile_pll_unroll
       << " \n";
    ss << "#define MICRO_" << x << "_TILE_PERP_UNROLL " << dp.at(emat_x).main_micro_tile_perp_unroll
       << '\n';

    defcom("MACRO_TILE_LENGTH_A / MICRO_A_TILE_PLL_UNROLL");
    ss << "#define N_MICRO_" << x << "_TILES_PLL_UNROLL "
       << dp.at(emat_x).main_n_micro_tiles_pll_unroll << " \n";
    if (emat_x == Mat::E::A)
      ss << "/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if "
            "the load tiles are "
            "truly contiguous tiles (0) */\n";
    ss << "#define LOAD_TO_LDS_INTERWOVEN_" << x << " " << hp.sus[emat_x].vs[Chi::E::LIW] << '\n';
    if (emat_x == Mat::E::A)
      ss << "/* Whether micro tile being processed by a compute item is "
            "interwoven with other "
            "micro tiles (ala Cobalt, (1)) or if the micro tiles are "
            "contiguous in C */\n";
    ss << "#define C_MICRO_TILES_INTERWOVEN_" << x << " " << hp.sus[emat_x].vs[Chi::E::MIW] << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* depending on whether loads to c are interwoven, set as MIW == "
            "0 ? 1 : "
            "N_MICRO_IN_MACRO_A */\n";
    ss << "#define C_INTERWEAVE_STRIDE_" << x << " " << dp.at(emat_x).main_c_interweave_stride
       << '\n';

    if (hp.sus[emat_x].vs[Chi::E::WOS] != Scratch::E::UNUSED)
    {
      if (emat_x == Mat::E::A)
        ss << "/* global memory offset, depends on type of copy of both a,b "
              "*/\n";
      ss << "#define GLOBAL_OFFSET_" << x << " " << dp.at(emat_x).cw_global_offset;
    }

    ss << '\n';
  }

  public:
  // the "main" kernel
  virtual KernBlob get_kernelstring() override final
  {

    std::stringstream ss;
    ss << get_time_string();
    ss << "\n\n";
    ss << "/* this kernel was generated for starting geometry : */\n";
    ss << "/* " << gg.get_string() << "*/\n";
    ss << "#define KV__ " << gg.k << '\n';
    ss << "#define TFLOAT  " << dp.t_float << '\n';
    ss << "#define DOES_BETA_C_INC " << dp.main_does_beta_c_inc << '\n';
    ss << "#define DOES_ALPHA_A_B_INC 1" << '\n';

    append_transpose_note(ss);

    ss <<
      R"(

)";

    for (auto emat_x : mata_matb)

    {
      ss << "\n/* ********************************** specific to " << Mat::M().name[emat_x]
         << " *************************************** */";
      add_predefine_chiral(emat_x, ss);
    }

    ss << "\n/* integer types for navigating each of the memory buffers */\n";
    for (size_t i = 0; i < Mem::E::N; ++i)
    {
      ss << "#define TINT" << Mem::M().name[i] << " " << dp.tints[i] << '\n';
    }
    ss << "\n/* type for integer in inner most loops (probably inlined anyway)  */\n";
    ss << "#define TSHORT " << dp.tshort << '\n';
    ss << "\n/* type for integers which never exceeds KV__ + UNROLL (for UFO case) */\n";
    ss << "#define TINTK " << dp.tintk << '\n';

    ss << "\n/* ********************************** common to A and B "
          "*************************************** */";
    ss << "\n/* whether or not to shimmy the starting k, in an attempt to "
          "avoid cache line overuse "
          "for cases where lda/ldb are powers of 2 */\n";
    ss << "/* if 0, no shimmying. if 1, instead of starting at k = 0 "
          "workgroups start at some "
          "negative offset dependent on work group id */\n";
    ss << "/* in the same way as the final unroll populates LDS with zeros in "
          "k mod UNROLL != 0, "
          "the initial negative indices here populate with 0 */\n";
    ss << "#define UNROLL_FOR_OFFSET " << hp.sus[Mat::E::C].vs[NonChi::E::UFO] << '\n';

    ss << "/* How much a workgroup loads (global -> LDS) in the k-direction at "
          "each iteration of "
          "the outer-most loop */\n";
    ss << "#define UNROLL " << hp.sus[Mat::E::C].vs[NonChi::E::UNR] << '\n';
    ss << "/* whether or not this kernel uses the edge trick (SC17 submission) "
          "*/\n";
    ss << "/* this precompiler defn has no direct influence on the running the "
          "kernel, "
          "implementation already done in make_kernel.py */\n";
    ss << "#define EDGETRICK " << dp.main_use_edge_trick << '\n';
    ss << "/* the number of work items working on the same c element. if this "
          "is 1, there will be "
          "just one thread doing all k multiply-adds, */\n";
    ss << "/* otherwise if it is greater than 1, each thread will be computing "
          "~ k / "
          "N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added "
          "at the end */ \n";
    ss << "#define N_WORK_ITEMS_PER_C_ELM " << hp.sus[Mat::E::C].vs[NonChi::E::ICE] << '\n';

    ss << R"(/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
)";

    append_group_allocation_defn_string(ss);

    ss << "/* Whether to use the unroll pragma to encourage the compiler to "
          "unroll certain loops "
          "*/\n";
    ss << "/* Included here for user, in practice it has no direct effect on "
          "this kernel, as the "
          "relevent implementation has been done in make_kernel.py */\n";
    ss << "#define PRAGMA_UNROLL_FORLOOPS " << hp.sus[Mat::E::C].vs[NonChi::E::PUN] << '\n';
    ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many "
          "steps ahead are we "
          "reading into registers, as compared to doing the math. */\n";
    ss << "/* This should be the domain of the compiler, and currently "
          "(26/08/2016, Catalyst) it "
          "seems that leaving this as 0 is best.  */\n";
    ss << "#define N_PREFETCH_FOR_REGISTER_LOAD " << 0 << '\n';
    ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many "
          "steps ahead are we "
          "reading into LDS, as compared to the unroll loop */\n";
    ss << "/* This should be the domain of the compiler, and currently "
          "(26/08/2016, Catalyst) it "
          "seems that leaving this as 0 is best.  */\n";
    ss << "#define N_PREFETCH_FOR_LDS_LOAD " << 0 << '\n';
    ss << "#define MACRO_TILE_AREA " << dp.main_macro_tile_area << '\n';
    ss << "#define MICRO_TILE_AREA " << dp.main_micro_tile_area << '\n';
    ss << "#define N_WORK_ITEMS_PER_WORKGROUP  " << dp.main_n_work_items_per_workgroup << '\n';
    ss << "/* two more parameters, which do dot have an effect the running of "
          "this kernel (used in "
          "enqueuing) */\n";
    ss << "/* the total number of work groups this kernel will use (recall "
          "m,n,k are fixed) */ \n";
    ss << "/* N_WORK_ITEMS_PER_C_ELM * ((M/MACRO_TILE_LENGTH_A) + "
          "(M%MACRO_TILE_LENGTH_A != 0)) * "
          "((N/MACRO_TILE_LENGTH_B) + (N%MACRO_TILE_LENGTH_B != 0)) */ \n";
    ss << "#define N_WORK_GROUPS " << dp.main_n_work_groups << '\n';
    ss << "/* the global work size, ie the total mumber of work items "
          "(threads) which will run */\n ";
    ss << "/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ \n";
    ss << "#define GLOBAL_WORK_SIZE " << dp.main_global_work_size << '\n';

    append_stride_c_defn(ss);
    append_split_on_k_defns_string(ss);
    append_super_column_width_defn(ss);

    ss << "\n\n\n__attribute__((reqd_work_group_size(" << dp.main_n_work_items_per_workgroup
       << ",1, 1)))\n";
    ss << "__kernel void ";
    ss << kernelname;
    append_fargs(ss);

    ss << "\n{\n\n";

    append_c_offset_string(ss);

    append_id_string_nonsym(ss);

    append_n_unrolls_remaining_string(ss);

    for (auto emat : mata_matb)
    {
      ss << "\n\n/* ************* " << Mat::M().name[emat] << " setup *************** */";
      append_id_string_sym(ss, emat);
    }

    ss << "\n\n\n";

    ss << "/* register memory for C */\n ";

    ss << "TFLOAT rC[MICRO_TILE_LENGTH_A][MICRO_TILE_LENGTH_B] = {{0.}};\n";

    append_first_unroll_block(ss);

    ss << "\n\nwhile (n_unrolls_remaining > 0){\n";
    append_relocate_load_math_string(ss, 0, 0);
    ss << "\n--n_unrolls_remaining;\n}\n";

    if (dp.main_final_fractional_unroll == 1)
    {
      ss << "\n/* *********** processing the tail *************** */\n";
      append_k_remaining_string(ss);
      append_final_unroll_string(ss);
      ss << "\n/* *********************************************** */\n\n";
    }

    ss << "\n\n";
    ss << "TINTC index;\n";

    append_split_on_k_vardecl_write_string(ss);
    append_final_write_all(ss);

    ss << "\n}\n";

    return {get_ktype(),
            {u_a, u_b, u_c, u_w, u_alpha, u_beta},
            ss.str(),
            kernelname,
            dp.main_global_work_size,
            dp.main_n_work_items_per_workgroup};
  }

  virtual size_t get_local_work_size() override final { return dp.main_n_work_items_per_workgroup; }

  virtual size_t get_n_work_groups() override final { return dp.main_n_work_groups; }

  virtual void set_type() override final
  {
    type = dp.main_does_beta_c_inc ? "betac_alphaab" : "alphaab";
  }

  virtual void setup_final() override final {}

  virtual KType::E get_ktype() override final { return KType::E::MAIN; }
};

KernBlob get_alpha_kernelstring(const HyPas& hp, const Geometry& gg, const DerivedParams& dp)
{
  AlphaGenerator ag(hp, gg, dp);
  ag.setup();
  return ag.get_kernelstring();
}
}
}
