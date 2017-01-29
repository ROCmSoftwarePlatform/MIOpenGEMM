#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/mapkeycheck.hpp>


namespace tinygemm{
namespace hyperparams{


HyperParamList::HyperParamList() {


shortkeys = {"Y", "X", "y", "x", "U", "P", "GA", "APLU", "BPLU", "PU", "LIW", "MIW", "ICE", "NAW", "UFO"};

map_shortkey_to_key =  {

  {"Y",  "macro_tile_height"}, 
  {"X",  "macro_tile_width"},
  {"y",  "micro_tile_height"},
  {"x",  "micro_tile_width"},
  {"U",  "unroll"},  
  {"P",  "pad"},
  {"GA",  "group_allocation"},
  {"APLU",  "work_item_load_a_pll_to_unroll"},
  {"BPLU",  "work_item_load_b_pll_to_unroll"},
  {"PU",  "unroll_pragma"},
  {"LIW",  "load_to_lds_interwoven"},
  {"MIW",  "c_micro_tiles_interwoven"},
  {"ICE",  "n_work_items_per_c_elm"},
  {"NAW",  "n_target_active_workgroups"},
  {"UFO",  "unroll_for_offset"}
  
  };

  for(auto const & v: shortkeys){
    keys.push_back(map_shortkey_to_key.at(v));
  }
}

HyperParamList hpl;


std::string HyperParamList::get_key_from_shortkey(const std::string & shortkey){
  if (hpl.map_shortkey_to_key.count(shortkey) == 0){
    std::stringstream ss;
    ss << "The shortkey `" << shortkey << "', does not appear as a key in map_shortkey_to_key. \n";
    throw tinygemm_error(ss.str());
  }
  
  else{
    return hpl.map_shortkey_to_key.at(shortkey);
  } 
}

unsigned HyperParams::get_val_from_shortkey(const std::string & shortkey) const{
  auto key = hpl.get_key_from_shortkey(shortkey);
  return *(pval_from_key.at(key));
}


/* take in hyper-parameter string and get a HyperParam object from it */
HyperParams get_hp(const std::string & hyperstring){
 
  auto frags = stringutil::split(hyperstring, "_");

  std::map<std::string, unsigned> hpdict;
  std::string shortkey;
  unsigned val;

  for (auto & x : frags){
    std::tie(shortkey, val) = stringutil::splitnumeric(x);
    hpdict[hpl.get_key_from_shortkey(shortkey)] = val;
  }
  
  auto newhp = HyperParams(hpdict);
  
  return newhp;
}



    




HyperParams get_default_small(bool enforce_deterministic){
  
  std::map<std::string, unsigned> params;
  
  params["micro_tile_width"] = 1;
  params["micro_tile_height"] = 1;
  params["macro_tile_width"] = 8;
  params["macro_tile_height"] = 8; 
  params["unroll"] = 16;

  params["pad"] = 1;    
  params["group_allocation"] = 1;
  params["work_item_load_a_pll_to_unroll"] = 0;
  params["work_item_load_b_pll_to_unroll"] = 1;
  params["unroll_pragma"] = 1;

  params["load_to_lds_interwoven"] = 0;
  params["c_micro_tiles_interwoven"] = 1;
  params["n_work_items_per_c_elm"] = (enforce_deterministic == false) ? 3 : 1;
  params["n_target_active_workgroups"] = 64;
  params["unroll_for_offset"] = 0;

  return HyperParams(params);
}

HyperParams get_default_tiniest(bool enforce_deterministic){
  
  std::map<std::string, unsigned> params;

  params["micro_tile_width"] = 1;
  params["micro_tile_height"] = 1;
  params["macro_tile_width"] = 1;
  params["macro_tile_height"] = 1; 
  params["unroll"] = 16;

  params["pad"] = 1;    
  params["group_allocation"] = 1;
  params["work_item_load_a_pll_to_unroll"] = 0;
  params["work_item_load_b_pll_to_unroll"] = 1;
  params["unroll_pragma"] = 1;

  params["load_to_lds_interwoven"] = 0;
  params["c_micro_tiles_interwoven"] = 1;
  params["n_work_items_per_c_elm"] = (enforce_deterministic == false) ? 3 : 1;
  params["n_target_active_workgroups"] = 64;
  params["unroll_for_offset"] = 0;

  return HyperParams(params);  
}



/* Find the nearest geometry in the cache, and take its hyper params */
HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic){
  
  
  


  /* The case  of  (gg.m < 8 || gg.n < 8) */  
  if (gg.m < 8 || gg.n < 8) {
    return get_default_tiniest(enforce_deterministic);
  }
  
  
  
  
  tinygemm::TinyGemmGeometry nearestgeometry;
  HyperParams best_hp = get_default_small(enforce_deterministic);
  

  float min_distance = std::numeric_limits<float>::max();
  
  tinygemm::TinyGemmGeometry geo;
  std::string hpstring;
  
  for (auto geohyp : HyperParams::kernel_cache){
    std::tie(geo, hpstring) = geohyp; 
    
    float new_distance = gg.get_distance(geo);
    if (new_distance < min_distance){
      nearestgeometry = geo;
      best_hp = get_hp(hpstring);
      min_distance = new_distance;
    }
  }

  if (enforce_deterministic == true){
    best_hp.n_work_items_per_c_elm = 1;
  }
    
  return best_hp;
}



void HyperParams::add_parameter_in_constructor(unsigned & up, const std::string & sp, const std::map<std::string, unsigned> & params){
  up = params.at(sp);
  pval_from_key[sp] = &up;


}
  


HyperParams::HyperParams(const std::map<std::string, unsigned> & params){


  mapkeycheck::check_map_keys(params, hpl.keys, "HyperParams constructor, params against keys");

  add_parameter_in_constructor(micro_tile_width, "micro_tile_width", params);
  
  //std::cout << "(constructor) " << this << "   " << &micro_tile_width << "  " << pval_from_key["micro_tile_width"] << "  " << *pval_from_key["micro_tile_width"] << std::endl;
  
  add_parameter_in_constructor(micro_tile_height, "micro_tile_height", params);
  add_parameter_in_constructor(macro_tile_width, "macro_tile_width", params);
  add_parameter_in_constructor(macro_tile_height, "macro_tile_height", params); 
  add_parameter_in_constructor(unroll, "unroll", params);
  add_parameter_in_constructor(pad, "pad", params);
  add_parameter_in_constructor(group_allocation, "group_allocation", params);
  add_parameter_in_constructor(work_item_load_a_pll_to_unroll, "work_item_load_a_pll_to_unroll", params);
  add_parameter_in_constructor(work_item_load_b_pll_to_unroll, "work_item_load_b_pll_to_unroll", params);
  add_parameter_in_constructor(unroll_pragma, "unroll_pragma", params);
  add_parameter_in_constructor(load_to_lds_interwoven, "load_to_lds_interwoven", params);
  add_parameter_in_constructor(c_micro_tiles_interwoven, "c_micro_tiles_interwoven", params);
  add_parameter_in_constructor(n_work_items_per_c_elm, "n_work_items_per_c_elm", params);
  add_parameter_in_constructor(n_target_active_workgroups, "n_target_active_workgroups", params);
  add_parameter_in_constructor(unroll_for_offset, "unroll_for_offset", params);

}

std::map<std::string, unsigned> HyperParams::get_map(){
  std::map<std::string, unsigned> params;
  
  //std::cout << "\n\n" << std::endl;
  
  //std::cout << "(constructor) " << this << "   " << &micro_tile_width << "  " << pval_from_key["micro_tile_width"] << "  " << *pval_from_key["micro_tile_width"] << std::endl;
    
  //std::cout << &pad << std::endl;
  for (auto & x : hpl.keys){
    params[x] = *(pval_from_key.at(x));
    //std::cout << "" << x << " --------------- " << pval_from_key.at(x) << "   " << params[x] << std::endl;
  }
  
  //std::abort();
  return params;
}




  

unsigned HyperParams::get_workgroup_size(){
  return (macro_tile_height * macro_tile_width) / (micro_tile_height * micro_tile_width);
}

unsigned HyperParams::get_nwitems_h(){
  return macro_tile_height / micro_tile_height;
}

unsigned HyperParams::get_nwitems_w(){
  return macro_tile_width / micro_tile_width;
}

bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string();
}
  
    
void HyperParams::add_hyperparam(const std::string & hyperstring, std::vector<HyperParams> & one_aways){
  one_aways.push_back(get_hp(hyperstring));
}
    
  
std::vector<HyperParams> HyperParams::get_one_aways(const tinygemm::TinyGemmGeometry & gg){
  
  if (gg.m < 8 || gg.n < 8){
    throw tinygemm_error("Currently, if matrix C has a dimension less that 16, it is not supported. If you are seeing this, please remind to fix it via github ");
  }
 
  std::vector<HyperParams> one_aways;
  
  size_t n_h0 = get_nwitems_h();
  size_t n_w0 = get_nwitems_w();

  
  /* *****************  micro tile sizes ***************************** 
   * We form a superset of the micro-tile to micro-tile one-step edges
   * as the cartesian product, micro_tile_step ^ 2. 
   * to view the micro_tile_step edge "graph", take a look at one_away_generator.py 
   * */
  std::map <unsigned, std::vector<unsigned> > micro_tile_step;
  micro_tile_step[1] = {1,2};
  micro_tile_step[2] = {1,2,3,4};
  micro_tile_step[3] = {2,3,4};
  micro_tile_step[4] = {2,3,4,5,6};
  micro_tile_step[5] = {4,5,6,8};
  micro_tile_step[6] = {4,5,6,8};
  micro_tile_step[8] = {6,8};

  /* we now form the one-step micro tile edges by pruning the product */
  std::map< std::array<unsigned, 2>, std::vector< std::array<unsigned, 2> > > micro_tile_edges;
  std::vector<unsigned> CC {1,2,3,4,5,6,8};
  for (auto & x: CC){
    for (auto & y : CC){
      for (auto & nx : micro_tile_step[x]){
        for (auto & ny : micro_tile_step[y]){
          /* eliminate type 1 skinny micro-tiles */
          bool not_too_skinny = (std::abs(int(nx) - int(ny)) <= 4);
          
          float delta_ratio = (float(x)/float(y)) / (float(nx)/float(ny));
          
          /* eliminate too dramatic changes is skinniness */
          bool skininess_change_good = (delta_ratio < 2.01 && delta_ratio > 0.499);
          float delta_volume = (float(x)*float(y)) / (float(nx)*float(ny));
          
          /* eliminate too dramatic changes in volume unless going to an `even hub' */
          bool volumn_change_good = ((nx%2 == 0 and ny%2 == 0) || (delta_volume <= 2.01 and delta_volume > 0.499));
          /* the only way to get to 5,8 is from 4,8 */
          bool condition_on_58 = ((x == 4 && y == 8) || (x == 8 && y == 4) || (!(nx == 5 && ny == 8) && !(nx == 8 && ny == 5)));
          if (not_too_skinny and skininess_change_good and volumn_change_good and condition_on_58){
            std::array<unsigned, 2> key  {{x,y}};
            std::array<unsigned, 2> value {{nx, ny}};
            micro_tile_edges[ key ].push_back( value );
          }
        }
      }
    }
  }
  
  for (auto & micro_tile : micro_tile_edges[ {{micro_tile_height , micro_tile_width }} ]){
    
    auto micro_h = micro_tile[0];
    auto micro_w = micro_tile[1];
    
    /* To each of the one-step micro tile edges, we can also (with p = 0.333) 
     * change n_work_items_per_c_elm in the same step, provided that n_work_items_per_c_elm 
     * increases by 1 if the area of the micro tile decreases, and v.v. */
    std::vector<unsigned> k_splits_to_consider;
    if (micro_h*micro_w < micro_tile_height*micro_tile_width && micro_tile_height*micro_tile_width < 36){
      if (rand()%3 == 0){ //TODO : is using rand bad?
        k_splits_to_consider = {n_work_items_per_c_elm, n_work_items_per_c_elm + 1};
      }
      else{
        k_splits_to_consider = {n_work_items_per_c_elm};
      }
    }
    
    else if (micro_h*micro_w > micro_tile_height*micro_tile_width && n_work_items_per_c_elm > 1 ){
      if (rand()%3 == 0){
        k_splits_to_consider = {n_work_items_per_c_elm, n_work_items_per_c_elm - 1};
      }
      else{
        k_splits_to_consider = {n_work_items_per_c_elm};
      }
    }
    
    else{
      k_splits_to_consider = {n_work_items_per_c_elm};
    }
    
    for (auto k_split : k_splits_to_consider){
      HyperParams hp(get_map());
      hp.micro_tile_height = micro_h;
      hp.micro_tile_width = micro_w;
      hp.macro_tile_height = micro_h*n_h0;
      hp.macro_tile_width = micro_w*n_w0;
      hp.n_work_items_per_c_elm = k_split;
      
      /* Observations suggest that k-split > 1 does not work very well with ufo. */
      if (k_split > 1){
        hp.unroll_for_offset = 0;
      }
       
      one_aways.push_back(hp);
    }
  }


  /* ***************** n_work_items_per_c_elms ************************
   * These hold the tile size constant, and explore just n_work_items_per_c_elm
   */
    
  std::vector<int> delta_k_split = {-4, -2, -1, 1, 2, 4, 8};
  for (auto & dx : delta_k_split){
    int old_k_split = static_cast<int>(n_work_items_per_c_elm);
    int new_k_split =  old_k_split + dx;
    if (new_k_split > 0 &&  (new_k_split / old_k_split <= 2)){
      HyperParams hp(get_map());
      hp.n_work_items_per_c_elm = new_k_split;
      /* Observations suggest that k-split > 1 does not work very well with ufo. */
      if (new_k_split > 1){
        hp.unroll_for_offset = 0;
      }
      one_aways.push_back(hp);
    }
  }
  


  /* ***************** macro tile sizes *************************************** */
  /* For Nvidia, where a wavefront (warp) is 32, should this be different? TODO */
  /* The standard 8x8 and 16x16 tiling schemes. *********************************/
  std::vector<unsigned> wg_hw_s = {8, 16};
  for (auto & wg_hw : wg_hw_s){
    HyperParams hp(get_map());
    hp.macro_tile_height = wg_hw*hp.micro_tile_height;
    hp.macro_tile_width = wg_hw*hp.micro_tile_width;          
    one_aways.push_back(hp);
  }
  
  /* Currently, if C has m or n less than 16, an error is thrown. */
  /* **************** unrolls **********************************  */
  
  std::vector<int> delta_unrolls = {-16, -8, +8, +16};
  for (auto & d_unroll : delta_unrolls){
    int old_unroll = unroll;
    int new_unroll = old_unroll + d_unroll;
    if (new_unroll > 0 && new_unroll <= 60){
      HyperParams hp(get_map());
      hp.unroll = new_unroll;
      /* (weak) observations suggest that unroll > 8 does not work well with ufo. */
      if (new_unroll > 8){
        hp.unroll_for_offset = 0;
      }
      one_aways.push_back(hp);
    }
  }
    
  
  
  if (n_work_items_per_c_elm >= 4){ //if n_work_items_per_c_elm is 4,5,6,7,8,9,10 ... consider making it 2,2,2,2,4,4,4,4 ... with an "increase" in unroll_map
    HyperParams hp_2(get_map());
    hp_2.unroll = 16*(hp_2.unroll/16 + 1); //= unroll_map.at(unroll).back();
    hp_2.n_work_items_per_c_elm = 2*(n_work_items_per_c_elm/4);
    //hp_2.unroll_for_offset = 0; /* this is a mere whim */
    one_aways.push_back(hp_2);
  }  
  
  /* **************** pads *****************************************
   * considering any pad other than 1 is  probably a waste of time, 
   * as I've never seen any pad significantly outperform 1. 
   * */
  std::vector<unsigned> pads = {1}; //2
  for (auto & pad_ : pads){
    HyperParams hp(get_map());
    hp.pad = pad_;
    one_aways.push_back(hp);
  }

  /* **************** group allocation *****************************
   * I have seen all of 2 (row-wise) , 1 (column-wise) and
   * 3 (column-wise within row-wise), performing strictly than the
   * other 2
   * */
  std::vector<unsigned> group_allocations = {1,2,3};
  for (auto & group_allocation_ : group_allocations){
    HyperParams hp(get_map());
    hp.group_allocation = group_allocation_;
    one_aways.push_back(hp);
  }

  /* ************** work_item_load_a_pll_to_unrolls ****************
   * TODO : reference to a base explanantion of meta params
  */
  std::vector<unsigned> work_item_load_a_pll_to_unrolls = {0,1};
  for (auto & work_item_load_a_pll_to_unroll_ : work_item_load_a_pll_to_unrolls){
    HyperParams hp(get_map());
    hp.work_item_load_a_pll_to_unroll = work_item_load_a_pll_to_unroll_;
    one_aways.push_back(hp);
  }

  /* ************** work_item_load_b_pll_to_unrolls ****************
  */
  std::vector<unsigned> work_item_load_b_pll_to_unrolls = {0,1};    
  for (auto & work_item_load_b_pll_to_unroll_ : work_item_load_b_pll_to_unrolls){
    HyperParams hp(get_map());  
    hp.work_item_load_b_pll_to_unroll = work_item_load_b_pll_to_unroll_;
    one_aways.push_back(hp);
  }
  
  /* ************** unroll_pragmas **************************************
   * probably not important. I have seen to be important in combo with UFO
   * I do recollect seeing this make a difference at some point in the past
   * Moreover, who knows what the next generation of compilers will do
  */
  std::vector<unsigned> unroll_pragmas = {0,1};
  for (auto & unroll_pragma_ : unroll_pragmas){
    HyperParams hp(get_map());
    hp.unroll_pragma = unroll_pragma_;
    one_aways.push_back(hp);
  }
  
  /* *******************load_to_lds_interwovens **************************
   * definitely an important parameter. 
   * So far, I've seen 0 beating 1
   * */    
  std::vector<unsigned> load_to_lds_interwovens = {0,1};
  for (auto & load_to_lds_interwoven_ : load_to_lds_interwovens){
    HyperParams hp(get_map());
    hp.load_to_lds_interwoven = load_to_lds_interwoven_;
    one_aways.push_back(hp);
  }
  
  /* ******************* c_micro_tiles_interwovens ************************
   * An important parameter. 
   * So far, I've seen 1 (ala cobalt) beating 0
   * */
  std::vector<unsigned> c_micro_tiles_interwovens = {0,1};
  for (auto & c_micro_tiles_interwoven_ : c_micro_tiles_interwovens){
    HyperParams hp(get_map());
    hp.c_micro_tiles_interwoven = c_micro_tiles_interwoven_;
    one_aways.push_back(hp);
  }
  
  /* ******************* unroll_for_offset ******************************************
   * This can improve performance by a significant amount.
   * Strangely, I have only seen it helping when combined with unroll_pragma true. 
   * My hypothesis for the above is that the compiler is relucant to unroll loops
   * with the additional complexity added by unroll_for_offset = 1.
   * */
  std::vector<unsigned> unroll_for_offsets = {0,1};
  for (auto & unroll_for_offset_ : unroll_for_offsets){
    HyperParams hp2(get_map());
    hp2.unroll_for_offset = unroll_for_offset_;
    hp2.unroll_pragma = true;
    one_aways.push_back(hp2);
  }
  
  
  bool add_custom_edges = true;
  if (add_custom_edges == true){
    
    /* ************************ custom edges ***************************************************
     * This is the place to add some edges experience shows can tunnel out of local minima, or
     * lead to some kernels which have found to be good on some problems  */
    
    auto add_hps = [this, & one_aways](std::string hparamstring){
      add_hyperparam(hparamstring, one_aways); //*this, 
    };
    
    if (micro_tile_height*micro_tile_width <=4 ){
      add_hps("Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE6_NAW64_UFO0");
    }

    if (micro_tile_height*micro_tile_width <=16 ){
      add_hps("Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE5_NAW64_UFO0");
    }

    if (micro_tile_height*micro_tile_width <=20 ){
      add_hps("Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ICE4_NAW64_UFO0");
    }
    
    if (micro_tile_height*micro_tile_width >=16 ){
      add_hps("Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0");
    }
    
    if (micro_tile_height*micro_tile_width >=8 ){
      add_hps("Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0");
    }

    if (micro_tile_height >= micro_tile_width &&  micro_tile_height*micro_tile_width >=10){
      add_hps("Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0");
    }

    if ((micro_tile_height == 8 || micro_tile_height == 4) && micro_tile_width ==  4){
      add_hps("Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0");
    }
    
    if ((micro_tile_height == 8 || micro_tile_height == 4) && micro_tile_width ==  4){    
      add_hps("Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0");
    }
    
    if ((micro_tile_height*micro_tile_height == 24) && n_work_items_per_c_elm >  1){    
      add_hps("Y48_X64_y3_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0");
    }
  
    if (micro_tile_height == 3 && micro_tile_height < micro_tile_width){    
      add_hps("Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0");
    }

    if (micro_tile_height*micro_tile_width > 5 && micro_tile_height*micro_tile_width < 48 && n_work_items_per_c_elm >  1){    
      add_hps("Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0");
    }

    if (gg.m*gg.n < 64*64 && gg.k > 20000){    
      add_hps("Y16_X32_y1_x2_U48_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE32_NAW64_UFO0");
    }
    
    if (gg.m*gg.n > 2000*2000){
      add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0");
    }    
    
    
    if (gg.tA == gg.isColMajor && gg.tB != gg.isColMajor && micro_tile_height*micro_tile_height == 64){
      add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1");
      add_hps("Y128_X128_y8_x8_U8_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1");
    }

    else if (gg.tA != gg.isColMajor && gg.tB == gg.isColMajor && micro_tile_height*micro_tile_height == 64){
      add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1");
      add_hps("Y128_X128_y8_x8_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1");
    }
    
    
    
  }
  
  /* shuffle them, which bounds the expected time to finding an improvement 
   * (prevents pathological case of all improving kernels at end of vector) 
   * currently, we shuffle after adding custom edges, might consider shuffling
   * before adding, to prevent getting trapped in previously found minima.*/
  std::random_device rd;
  std::default_random_engine default_engine(rd());
  std::shuffle(one_aways.begin(), one_aways.end(), default_engine);
  return one_aways;


}
  
  
std::vector<HyperParams> HyperParams::get_two_aways(const tinygemm::TinyGemmGeometry & gg){
  std::vector<HyperParams> two_aways;
  std::vector<HyperParams> one_aways = get_one_aways(gg);
  for (auto & hp : one_aways){
    std::vector<HyperParams> two_aways_via = hp.get_one_aways(gg);
    for (auto & hp2 : two_aways_via){
      auto blip = std::find(two_aways.begin(), two_aways.end(), hp2);
      if (blip == two_aways.end()) {
        two_aways.push_back(hp2);
      }
      else{
      }
    }
  }

  std::random_device rd;
  std::default_random_engine default_engine(rd());
  std::shuffle(two_aways.begin(), two_aways.end(), default_engine);    
  return two_aways;
}



  
/* TODO is this function ever used ? */
std::string HyperParams::get_string() const{

  std::stringstream ss;
  
  unsigned i_key;
  
  for (i_key = 0; i_key < hpl.shortkeys.size()  - 1; ++i_key){
    ss << hpl.shortkeys[i_key] << get_val_from_shortkey(hpl.shortkeys[i_key]) << "_";
  }
  ss << hpl.shortkeys[i_key] << get_val_from_shortkey(hpl.shortkeys[i_key]);
  return ss.str();
}


bool HyperParams::can_be_used_on(const tinygemm::TinyGemmGeometry & gg){
  unsigned m = gg.m;
  unsigned n = gg.n;
  bool tC = gg.tC;
  
  if (macro_tile_height < tC ? n : m){
    return false;
  }
  
  if (macro_tile_width < tC ? n : m){
    return false;
  }
  
  return true;
}



//std::make_tuple(100, 32, 26939, 26939, 26939, 100, true, false), 


/* see dev/python/deepbench/deepbench_results.py : this is generated by get_kernel_cache_string, based on results running find with allotted_time =  30 seconds per problem, with three starting kernels for
 * small, medium, large: On a Fiji! 
 * TODO : regenerate with longer runs and more problems.
 * TODO : should not be a single vector, this has linear find time. At least seperate out isColMajor, tA, tB  
 * TODO : figure out how to make cache contain only reduced problems.... very important! */
std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string> > 
HyperParams::kernel_cache = {                             /* colMaj tA    tB     tC     lda   ldb   ldc   m     n    k     ao bo co                                                                       */
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 128, 3072, 0, 0, 0} , 
  "Y96_X64_y6_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW1_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU0_PU1_LIW1_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 32, 1024, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 16, 7680, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW0_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2560, 5124, 5124, 9124, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU1_BPLU0_PU0_LIW1_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y2_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2560, 7133, 2560, 2560, 7133, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 0, 0} , "Y8_X16_y1_x2_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U16_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2560, 2560, 8457, 35, 0, 0, 0} , "Y48_X48_y3_x3_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 0, 0} , "Y16_X16_y2_x2_U8_P1_GA2_APLU0_BPLU1_PU1_LIW1_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7680, 5481, 7680, 7680, 5481, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2048, 2048, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA1_APLU1_BPLU1_PU0_LIW1_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 1760, 1760, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2048, 35, 35, 8457, 2048, 0, 0, 0} , "Y24_X24_y3_x3_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 4096, 5124, 5124, 9124, 4096, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 0, 0} , "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU1_LIW1_MIW1_ICE5_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2560, 35, 35, 8457, 2560, 0, 0, 0} , "Y24_X40_y3_x5_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 32, 7680, 0, 0, 0} , "Y48_X32_y3_x2_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 4096, 4096, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1760, 7133, 1760, 1760, 7133, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4096, 7133, 4096, 4096, 7133, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 5124, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 128, 1024, 0, 0, 0} , "Y32_X64_y2_x4_U48_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW1_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 64, 2560, 0, 0, 0} , "Y96_X64_y6_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 128, 1024, 0, 0, 0} , "Y96_X32_y6_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 64, 3072, 0, 0, 0} , "Y32_X64_y2_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 35, 35, 8457, 2560, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 0, 0} , "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU1_LIW1_MIW1_ICE9_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 0, 0} , "Y32_X16_y2_x1_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 4096, 35, 35, 8457, 4096, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 32, 1024, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 1760, 35, 35, 8457, 1760, 0, 0, 0} , "Y24_X48_y3_x6_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU1_BPLU0_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 4096, 4096, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 0, 0} , "Y48_X32_y3_x2_U64_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ICE8_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 35, 35, 8457, 4096, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 64, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA3_APLU0_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 16, 1024, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE9_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 5124, 5124, 9124, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 35, 35, 8457, 2048, 0, 0, 0} , "Y24_X16_y3_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2560, 2560, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 16, 3072, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW0_ICE9_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2048, 2048, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 16, 1024, 0, 0, 0} , "Y48_X16_y3_x1_U32_P1_GA1_APLU0_BPLU1_PU1_LIW1_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3072, 7435, 3072, 3072, 7435, 1024, 0, 0, 0} , "Y96_X96_y6_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 64, 7680, 0, 0, 0} , "Y80_X32_y5_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE9_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 1760, 1760, 8457, 35, 0, 0, 0} , "Y80_X80_y5_x5_U16_P1_GA1_APLU0_BPLU1_PU0_LIW1_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 128, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 5124, 5124, 9124, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 64, 1024, 0, 0, 0} , "Y32_X32_y2_x2_U48_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 32, 3072, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU1_BPLU0_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE8_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 5124, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 0, 0} , "Y16_X32_y2_x4_U32_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE8_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 16, 2560, 0, 0, 0} , "Y40_X16_y5_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U32_P1_GA1_APLU0_BPLU0_PU0_LIW1_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 1760, 5124, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 32, 2560, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2048, 7133, 2048, 2048, 7133, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 35, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 64, 1024, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 128, 7680, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2048, 5124, 5124, 9124, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 128, 3072, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE8_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 32, 1024, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 16, 7680, 0, 0, 0} , "Y24_X16_y3_x2_U64_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW0_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2567, 5137, 5124, 9124, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2565, 7140, 2573, 2560, 7133, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2573, 2560, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U8_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 0, 0} , "Y32_X16_y4_x2_U32_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7685, 5488, 7693, 7680, 5481, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2061, 2048, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 1773, 1760, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2055, 48, 35, 8457, 2048, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 4103, 5137, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2567, 48, 35, 8457, 2560, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 32, 7680, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 4109, 4096, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1765, 7140, 1773, 1760, 7133, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4101, 7140, 4109, 4096, 7133, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 5137, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 128, 1024, 0, 0, 0} , "Y48_X64_y3_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW1_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 64, 2560, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 128, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U8_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 64, 3072, 0, 0, 0} , "Y32_X64_y2_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 48, 35, 8457, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 32, 2560, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 0, 0} , "Y8_X16_y1_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW0_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 32, 1024, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 1767, 48, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U8_P1_GA1_APLU1_BPLU1_PU1_LIW1_MIW0_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 4103, 48, 35, 8457, 4096, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 128, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 48, 35, 8457, 4096, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 64, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE3_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU0_PU0_LIW1_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 16, 1024, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW0_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 5137, 5124, 9124, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 48, 35, 8457, 2048, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2573, 2560, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 16, 3072, 0, 0, 0} , "Y32_X16_y2_x1_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2061, 2048, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 16, 1024, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3077, 7442, 3085, 3072, 7435, 1024, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 64, 7680, 0, 0, 0} , "Y128_X64_y8_x4_U32_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 48, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 4109, 4096, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 5137, 5124, 9124, 2048, 0, 0, 0} , "Y96_X128_y6_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 64, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 32, 3072, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 16, 2560, 0, 0, 0} , "Y48_X16_y3_x1_U48_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE8_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 0, 0} , "Y24_X16_y3_x2_U48_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW0_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 5137, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 16, 2560, 0, 0, 0} , "Y40_X16_y5_x2_U8_P1_GA1_APLU1_BPLU0_PU0_LIW1_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 1767, 5137, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 32, 2560, 0, 0, 0} , "Y48_X32_y3_x2_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2053, 7140, 2061, 2048, 7133, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 1773, 1760, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 64, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 128, 7680, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE4_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2055, 5137, 5124, 9124, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  
  
  //some back conv problems : 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1000, 1000, 16, 16, 16, 1000, 0, 0, 0} ,   "Y8_X8_y1_x1_U40_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5760, 5760, 144, 144, 32, 5760, 0, 0, 0} ,   "Y8_X8_y1_x1_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 23040, 23040, 9, 9, 16, 23040, 0, 0, 0} ,   "Y8_X8_y1_x1_U48_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW0_ICE37_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 12544, 12544, 147, 147, 64, 12544, 0, 0, 0} ,   "Y32_X64_y2_x4_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE24_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 26939, 26939, 100, 100, 32, 26939, 0, 0, 0} ,   "Y16_X32_y1_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW0_ICE18_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2916, 2916, 27, 27, 64, 2916, 0, 0, 0} ,   "Y8_X8_y1_x1_U32_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE12_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 50176, 50176, 27, 27, 64, 50176, 0, 0, 0} ,   "Y16_X32_y1_x2_U48_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW0_ICE32_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 832, 832, 256, 49, 0, 0, 0} ,   "Y32_X32_y2_x2_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 192, 192, 64, 784, 0, 0, 0} ,   "Y16_X16_y1_x1_U16_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2916, 2916, 576, 576, 64, 2916, 0, 0, 0} ,   "Y64_X32_y4_x2_U16_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ICE7_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 6308, 6308, 1600, 1600, 32, 6308, 0, 0, 0} ,   "Y64_X32_y4_x2_U40_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1440, 1440, 288, 288, 64, 1440, 0, 0, 0} ,   "Y32_X32_y2_x2_U32_P1_GA1_APLU1_BPLU0_PU0_LIW0_MIW1_ICE6_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 4608, 4608, 512, 49, 0, 0, 0} ,   "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 2304, 2304, 512, 49, 0, 0, 0} ,   "Y32_X32_y2_x2_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 512, 512, 192, 196, 0, 0, 0} ,   "Y32_X16_y2_x1_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 20800, 20800, 128, 49, 0, 0, 0} ,   "Y32_X32_y2_x2_U8_P1_GA3_APLU1_BPLU0_PU1_LIW0_MIW0_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 360, 360, 576, 576, 128, 360, 0, 0, 0} ,   "Y16_X16_y2_x2_U24_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW0_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 4608, 4608, 512, 196, 0, 0, 0} ,   "Y96_X64_y6_x4_U16_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 12800, 12800, 48, 196, 0, 0, 0} ,   "Y32_X48_y2_x3_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3136, 3136, 1152, 1152, 256, 3136, 0, 0, 0} ,   "Y80_X64_y5_x4_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 1152, 1152, 256, 196, 0, 0, 0} ,   "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ICE1_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 12544, 12544, 576, 576, 128, 12544, 0, 0, 0} ,   "Y96_X128_y6_x8_U32_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ICE20_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 4800, 4800, 32, 784, 0, 0, 0} ,   "Y32_X32_y2_x2_U32_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ICE5_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 729, 729, 1152, 1152, 128, 729, 0, 0, 0} ,   "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ICE2_NAW64_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 2304, 2304, 512, 784, 0, 0, 0} ,   "Y96_X64_y6_x4_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ICE1_NAW64_UFO0" ), 




};


 
}
} // namespace




