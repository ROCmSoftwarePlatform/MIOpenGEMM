#ifndef BASICFIND_HPP
#define BASICFIND_HPP



namespace tinygemm{
  
  
tinygemm::TinyGemmSolution basicfind(const tinygemm::TinyGemmGeometry & geometry, const tinygemm::TinyGemmOffsets & toff, const tinygemm::FindParams & find_params, bool verbose, std::string logfile, std::string constraints_string, unsigned n_postfind_runs, bool do_cpu_test);


}

#endif
