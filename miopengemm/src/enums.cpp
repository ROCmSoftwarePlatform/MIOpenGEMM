/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <miopengemm/enums.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

// as declared in outputwriter.hpp:
owrite::Endline Endl;
owrite::Flusher Flush;

namespace Floating
{

const double& get_default_alpha()
{
  static const double default_alpha = 0.415693029182345929;
  return default_alpha;
}

const double& get_default_beta()
{
  static const double default_beta = -0.31415011010010110;
  return default_beta;
}

MFType::MFType(double v) : v_d(v), v_f(static_cast<float>(v)) {}
const void* MFType::operator[](char floattype) const
{
  return floattype == 'd' ? static_cast<const void*>(&v_d) : static_cast<const void*>(&v_f);
}

const MFType& get_m_alpha()
{
  const static MFType m_alpha(get_default_alpha());
  return m_alpha;
}

const MFType& get_m_beta()
{
  const static MFType m_beta(get_default_beta());
  return m_beta;
}
}

template <typename T>
T unfilled();

template <>
char unfilled()
{
  return '?';
}

template <>
std::string unfilled()
{
  return "?";
}

template <typename T>
void confirm(const std::vector<T>& X, std::string enum_name)
{
  for (auto& x : X)
  {
    if (x == unfilled<T>())
    {
      throw miog_error("unpopulated element of vector for " + enum_name + ".");
    }
  }
}

template <typename T>
std::unordered_map<T, size_t> get_val(const std::vector<T>& a)
{
  std::unordered_map<T, size_t> X;
  for (size_t i = 0; i < a.size(); ++i)
  {
    X[a[i]] = i;
  }
  return X;
}

template <typename T>
T aslower(T X)
{
  (void)X;
}

template <>
char aslower(char x)
{
  char y = std::tolower(static_cast<unsigned char>(x));
  return y;
}

template <>
std::string aslower(std::string X)
{
  std::string Y = X;
  std::transform(X.begin(), X.end(), Y.begin(), ::tolower);
  return Y;
}

template <typename T>
EnumMapper<T>::EnumMapper(const std::vector<T>& name_)
  : N(name_.size()), name(name_), all_enum(name.size()), val(get_val<T>(name))
{

  std::iota(all_enum.begin(), all_enum.end(), 0);

  lcase_name.resize(name.size());
  for (size_t i = 0; i < name.size(); ++i)
  {
    lcase_name[i] = aslower<T>(name[i]);
  }
}

template <typename T>
EnumMapper<T> get_enum_mapper(const std::vector<T>& name_, std::string enum_name)
{
  confirm<T>(name_, enum_name);
  return EnumMapper<T>(name_);
}

namespace KType
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::WSA]   = "WSA";
  X[E::WSB]   = "WSB";
  X[E::BETAC] = "BETAC";
  X[E::MAIN]  = "MAIN";
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "KType");
  return em;
}
}

namespace SummStat
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::MEAN]   = "MEAN";
  X[E::MEDIAN] = "MEDIAN";
  X[E::MAX]    = "MAX";
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "SummStat");
  return em;
}
}

namespace Xtr
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::MIN] = "MIN";
  X[E::MAX] = "MAX";
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "Xtr");
  return em;
}
}

namespace MicroAllocation
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::BYA] = "BYA";
  X[E::BYB] = "BYB";
  return X;
}
const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em =
    get_enum_mapper<std::string>(get_name(), "MicroAllocation");
  return em;
}
}

std::vector<int> get_priority_confirmed(std::vector<int> X, size_t target_size)
{
  if (X.size() != target_size)
  {
    throw miog_error("priority vector not of correct size");
  }

  for (auto& x : X)
  {
    if (x == std::numeric_limits<int>::max())
    {
      throw miog_error("it appears as though a priority has not been set");
    }
  }
  return X;
}

namespace Chi
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::MIC] = "MIC";
  X[E::PAD] = "PAD";
  X[E::PLU] = "PLU";
  X[E::LIW] = "LIW";
  X[E::MIW] = "MIW";
  X[E::WOS] = "WOS";
  X[E::VEW] = "VEW";
  return X;
}

std::vector<int> get_priority_basic()
{
  std::vector<int> X(E::N, std::numeric_limits<int>::max());
  X[E::MIC] = 1;
  X[E::PAD] = 0;
  X[E::PLU] = 0;
  X[E::LIW] = 0;
  X[E::MIW] = 0;
  X[E::WOS] = 0;
  X[E::VEW] = 0;
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "Chi");
  return em;
}

const std::vector<int>& get_priority()
{
  const static std::vector<int> priority = get_priority_confirmed(get_priority_basic(), E::N);
  return priority;
}
}

namespace OutPart
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::MAI] = "MAI";
  X[E::TRA] = "TRA";
  X[E::DEP] = "DEP";
  X[E::ACC] = "ACC";
  X[E::WRN] = "WRN";
  X[E::CCH] = "CCH";
  X[E::BEN] = "BEN";
  X[E::MER] = "MER";
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "OutPart");
  return em;
}
}

namespace Ver
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::SILENT]       = "SILENT";
  X[E::TERMINAL]     = "TERMINAL";
  X[E::TERMWITHDEPS] = "TERMWITHDEPS";
  X[E::SPLIT]        = "SPLIT";
  X[E::TOFILE]       = "TOFILE";
  X[E::TRACK]        = "TRACK";
  X[E::STRACK]       = "STRACK";
  X[E::ACCURACY]     = "ACCURACY";
  X[E::MULTIBENCH]   = "MULTIBENCH";
  X[E::MERGE]        = "MERGE";
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "Ver");
  return em;
}
}

namespace NonChi
{
std::vector<std::string> get_name()
{
  std::vector<std::string> X(E::N, unfilled<std::string>());
  X[E::UNR] = "UNR";
  X[E::GAL] = "GAL";
  X[E::PUN] = "PUN";
  X[E::ICE] = "ICE";
  X[E::NAW] = "NAW";
  X[E::UFO] = "UFO";
  X[E::MAC] = "MAC";
  X[E::SKW] = "SKW";
  X[E::IWI] = "IWI";
  X[E::SZT] = "SZT";
  X[E::MAD] = "MAD";
  X[E::AFI] = "AFI";
  X[E::MIA] = "MIA";
  return X;
}

std::vector<int> get_priority_basic()
{
  std::vector<int> X(E::N, std::numeric_limits<int>::max());
  X[E::UNR] = 1;
  X[E::GAL] = 0;
  X[E::PUN] = 0;
  X[E::ICE] = 1;
  X[E::NAW] = 0;
  X[E::UFO] = 0;
  X[E::MAC] = 1;
  X[E::SKW] = 0;
  X[E::IWI] = 0;
  X[E::MAD] = -1;
  X[E::AFI] = -1;
  X[E::MIA] = -1;
  X[E::SZT] = -1;
  return X;
}

const EnumMapper<std::string>& M()
{
  static const EnumMapper<std::string> em = get_enum_mapper<std::string>(get_name(), "NonChi");
  return em;
}

const std::vector<int>& get_priority()
{
  const static std::vector<int> prty = get_priority_confirmed(get_priority_basic(), E::N);
  return prty;
}
}

namespace Mat
{
std::vector<char> get_name()
{
  std::vector<char> X(E::N, unfilled<char>());
  X[E::A] = 'A';
  X[E::B] = 'B';
  X[E::C] = 'C';
  return X;
}

const EnumMapper<char>& M()
{
  static const EnumMapper<char> em = get_enum_mapper<char>(get_name(), "Mat");
  return em;
}

const EnumMapper<std::string>* mat_to_xchi(Mat::E emat)
{
  switch (emat)
  {
  case Mat::E::A: return &Chi::M();
  case Mat::E::B: return &Chi::M();
  case Mat::E::C: return &NonChi::M();
  case Mat::E::N: throw miog_error("unrecognised Mat::E (N) in mat_to_xchi");
  }
  throw miog_error("failed in mat_to_xchi");
}

const std::vector<int>* mat_to_priority(Mat::E emat)
{
  switch (emat)
  {
  case Mat::E::A: return &Chi::get_priority();
  case Mat::E::B: return &Chi::get_priority();
  case Mat::E::C: return &NonChi::get_priority();
  case Mat::E::N: throw miog_error("unrecognised Mat::E (N) in mat_to_priority");
  }
  throw miog_error("failed in mat_to_priority");
}

Mat::E mem_to_mat(Mem::E emat)
{
  switch (emat)
  {
  case Mem::E::A: return Mat::E::A;
  case Mem::E::B: return Mat::E::B;
  case Mem::E::C: return Mat::E::C;
  case Mem::E::W: throw miog_error("no mat enum for supposed ::W");
  case Mem::E::N: throw miog_error("no mat enum for supposed ::N");
  }
  throw miog_error("failed in mem_to_mat");
}
}

namespace Mem
{
std::vector<char> get_name()
{
  std::vector<char> X(E::N, unfilled<char>());
  X[E::A] = 'A';
  X[E::B] = 'B';
  X[E::C] = 'C';
  X[E::W] = 'W';
  return X;
}

const EnumMapper<char>& M()
{
  static const EnumMapper<char> em = get_enum_mapper<char>(get_name(), "Mem");
  return em;
}

Mem::E mat_to_mem(Mat::E emat)
{
  if (emat == Mat::E::A)
  {
    return Mem::E::A;
  }

  else if (emat == Mat::E::B)
  {
    return Mem::E::B;
  }

  else if (emat == Mat::E::C)
  {
    return Mem::E::C;
  }

  else
  {
    throw miog_error("no mem enum for supposed mat enum provided");
  }
}
}

namespace KType
{
std::array<std::vector<size_t>, E::N> get_dependencies_basic()
{
  std::vector<size_t> uninitialised_vector{std::numeric_limits<size_t>::max()};
  std::array<std::vector<size_t>, E::N> kdps;
  for (size_t i = 0; i < E::N; ++i)
  {
    kdps[i] = uninitialised_vector;
  }
  kdps[E::WSA]   = {};
  kdps[E::WSB]   = {};
  kdps[E::BETAC] = {};
  kdps[E::MAIN]  = {E::BETAC, E::WSA, E::WSB};

  for (auto& x : kdps)
  {
    if (x == uninitialised_vector)
    {
      throw miog_error("dependencies does not appear to be initialised entirely");
    }
  }
  return kdps;
}

const std::array<std::vector<size_t>, KType::N>& get_dependencies()
{
  static const std::array<std::vector<size_t>, KType::N> dependencies = get_dependencies_basic();
  return dependencies;
}
}

namespace Ver
{

std::array<std::array<bool, OutPart::E::N>, E::N> get_base_toX()
{

  std::array<std::array<bool, OutPart::E::N>, E::N> x;

  for (size_t vi = 0; vi < E::N; ++vi)
  {

    for (size_t op = 0; op < OutPart::E::N; ++op)
    {
      x[vi][op] = false;
    }
  }
  return x;
}

std::array<std::array<bool, OutPart::E::N>, E::N> get_toTerm_basic()
{
  auto x = get_base_toX();

  // all output to terminal, other than tracker
  x[E::TERMINAL][OutPart::E::MAI] = true;
  x[E::TERMINAL][OutPart::E::ACC] = true;
  x[E::TERMINAL][OutPart::E::WRN] = true;

  // copy TERMINAL
  x[E::SPLIT] = x[E::TERMINAL];

  // just tracker output to terminal
  x[E::TRACK][OutPart::E::TRA] = true;
  x[E::TRACK][OutPart::E::WRN] = true;

  // just tracker output to terminal
  x[E::STRACK][OutPart::E::TRA] = true;

  // like tracker, but with accuracy
  x[E::ACCURACY]                  = x[E::TRACK];
  x[E::ACCURACY][OutPart::E::ACC] = true;

  // like terminal, but with dependency of kernels printed
  x[E::TERMWITHDEPS]                  = x[E::TERMINAL];
  x[E::TERMWITHDEPS][OutPart::E::DEP] = true;

  x[E::MULTIBENCH][OutPart::E::BEN] = true;

  x[E::MERGE][OutPart::E::MER] = true;

  return x;
}

const std::array<std::array<bool, OutPart::E::N>, E::N>& get_toTerm()
{
  const static std::array<std::array<bool, OutPart::E::N>, E::N> toTerm = get_toTerm_basic();
  return toTerm;
}

std::array<std::array<bool, OutPart::E::N>, E::N> get_toFile_basic()
{
  auto x = get_base_toX();
  // all output to file, other than tracker
  x[E::TOFILE][OutPart::E::MAI] = true;
  x[E::TOFILE][OutPart::E::ACC] = true;
  x[E::TOFILE][OutPart::E::WRN] = true;

  // copy TOFILE
  x[E::SPLIT] = x[E::TOFILE];

  // copy TOFILE
  x[E::STRACK]                  = x[E::TOFILE];
  x[E::STRACK][OutPart::E::CCH] = true;

  return x;
}

const std::array<std::array<bool, OutPart::E::N>, E::N>& get_toFile()
{
  const static std::array<std::array<bool, OutPart::E::N>, E::N> toFile = get_toFile_basic();
  return toFile;
}

std::array<bool, E::N> get_fileRequired_basic()
{
  std::array<bool, E::N> X;
  X[E::SILENT]       = false;
  X[E::TERMINAL]     = false;
  X[E::TERMWITHDEPS] = false;
  X[E::SPLIT]        = true;
  X[E::TOFILE]       = true;
  X[E::TRACK]        = false;
  X[E::STRACK]       = true;
  X[E::ACCURACY]     = false;
  X[E::MULTIBENCH]   = false;
  X[E::MERGE]        = false;
  return X;
}

const std::array<bool, E::N>& get_fileRequired()
{
  static const std::array<bool, E::N> fileRequired = get_fileRequired_basic();
  return fileRequired;
}
}
}
