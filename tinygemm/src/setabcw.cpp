#include <tinygemm/setabcw.hpp>
#include <tinygemm/error.hpp>

#include <sstream>

namespace setabcw{



template <typename TFloat>
void fill_uni(std::vector<TFloat> & v, unsigned r_small, unsigned r_big){

  if (r_small > r_big){
    std::stringstream ss;
    ss << "in fill_uni, with r_small > r_big. this seems like a strange (incorrect) request ";
    throw tinygemm::tinygemm_error(ss.str());    
  }
  
  
  if (r_small > v.size()){
    throw tinygemm::tinygemm_error("in fill_uni, r_small > v.size()");
  }

  if (r_big > v.size()){
    throw tinygemm::tinygemm_error("in fill_uni, r_big > v.size()");
  }
    
  for (size_t i = 0; i < r_small; ++i){
    v[i] = TFloat(rand() % 1000) / 1000. - 0.5;
  }
  
  for (size_t i = r_small; i < r_big; ++i){
    v[i] = 1e9*(TFloat(rand() % 1000) / 1000. - 0.5);
  }
}


template <typename TFloat>
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff){

  size_t n_a = gg.ldX[tinygemm::nsHP::matA] * (gg.tX[tinygemm::nsHP::matA] == gg.isColMajor ? gg.m : gg.k) + toff.oa + toff.tail_off_a;
  size_t n_b = gg.ldX[tinygemm::nsHP::matB] * (gg.tX[tinygemm::nsHP::matB] == gg.isColMajor ? gg.k : gg.n) + toff.ob + toff.tail_off_b;
  size_t n_c = gg.ldX[tinygemm::nsHP::matC] * (gg.tX[tinygemm::nsHP::matC] == gg.isColMajor ? gg.m : gg.n) + toff.oc + toff.tail_off_c; 


  if (n_a > 20000*10000 || n_b > 20000*10000 || n_c > 20000*10000){
    std::stringstream ss;
    ss << "currently, this code only generates random matrices of size less than 20000*10000. ";
    ss << "the request with geometry " << gg.get_string() << " is therefore rejected.";
    ss << " (n_a = " << n_a << " n_b = " << n_b << " n_c = " << n_c << ")";
    throw tinygemm::tinygemm_error(ss.str());
  } 
  /* fill matrices with random floats. It is important to fill them with random floats, 
   * as if they're integers, the kernel can, and does, cheat! (runs faster) */
  v_a.resize(n_a);
  v_b.resize(n_b);
  v_c.resize(n_c);
   
  fill_uni<TFloat>(v_a, n_a - toff.tail_off_a, n_a);
  fill_uni<TFloat>(v_b, n_b - toff.tail_off_b, n_b);
  fill_uni<TFloat>(v_c, n_c - toff.tail_off_c, n_c);
}



template <typename TFloat>     
void set_abcw(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, std::vector<TFloat> & v_workspace, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff){
    
  set_abc<TFloat>(v_a, v_b, v_c, gg, toff);

  size_t n_workspace = gg.workspace_size + toff.oworkspace;

  v_workspace.resize(n_workspace);
  fill_uni(v_workspace, n_workspace, n_workspace);
}


template void set_abc(std::vector<double> & v_a, std::vector<double> & v_b, std::vector<double> & v_c, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);

template void set_abc(std::vector<float> & v_a, std::vector<float> & v_b, std::vector<float> & v_c, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);

template void set_abcw(std::vector<double> & v_a, std::vector<double> & v_b, std::vector<double> & v_c, std::vector<double> & v_workspace, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);

template void set_abcw(std::vector<float> & v_a, std::vector<float> & v_b, std::vector<float> & v_c, std::vector<float> & v_workspace, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);

}
