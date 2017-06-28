/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#ifndef GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP
#define GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP

#include <string>
#include <vector>

namespace MIOpenGEMM
{

namespace nsHP
{
enum eMat
{
  matA,
  matB,
  matC,
  nMats
};
}

// maps eMats to characters
extern std::vector<char> matChars;

// offsets and tails
class Offsets
{
  public:
  // offsets of a,b,c and workspace
  unsigned oa;
  unsigned ob;
  unsigned oc;
  unsigned oworkspace;

  // tails of a,b and c. why no workspace?
  unsigned tail_off_a;
  unsigned tail_off_b;
  unsigned tail_off_c;

  Offsets(unsigned oa,
          unsigned ob,
          unsigned oc,
          unsigned oworkspace,
          unsigned tail_off_a,
          unsigned tail_off_b,
          unsigned tail_off_c);

  const unsigned& operator[](char c) const;
};

class GeometryDerived
{
  public:
  unsigned float_size_bits;
  unsigned float_size_bytes;
  void reset(char floattype);
};

class Geometry
{

  private:
  void initialise(bool     isColMajor_,
                  bool     tA_,
                  bool     tB_,
                  bool     tC_,
                  unsigned lda_,
                  unsigned ldb_,
                  unsigned ldc_,
                  unsigned m_,
                  unsigned n_,
                  unsigned k_,
                  unsigned workspace_size_,
                  char     floattype_);

  public:
  bool isColMajor;

  // indexed by eMat  (for A, B and C)
  std::vector<bool> tX;

  // indexed by eMat (for A, B and C)
  std::vector<unsigned> ldX;

  unsigned m;
  unsigned n;
  unsigned k;

  // the usable amount of workspace
  unsigned workspace_size;

  // 'f' : 32-bit single precision or 'd' : 64-bit double precision
  char floattype;

  GeometryDerived derived;

  Geometry(bool     isColMajor,
           bool     tA,
           bool     tB,
           bool     tC,
           unsigned lda,
           unsigned ldb,
           unsigned ldc,
           unsigned m,
           unsigned n,
           unsigned k,
           unsigned workspace_size,
           char     floattype);

  Geometry() = default;

  Geometry(const Geometry&) = default;

  Geometry(std::string geometry_string);

  Geometry& operator=(const Geometry&) = default;

  unsigned get_padless_dim(nsHP::eMat emat_x, bool isCoal) const;

  unsigned get_coal(nsHP::eMat emat_x) const;

  unsigned get_uncoal(nsHP::eMat emat_x) const;

  unsigned get_non_k_dim(nsHP::eMat emat_x) const;

  bool coal_is_pll_k(nsHP::eMat emat_x) const;

  std::string get_string() const;

  std::string get_networkconfig_string() const;

  void check_ldx_consistent() const;
};
}

#endif
