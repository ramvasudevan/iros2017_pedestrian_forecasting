#include <iostream>
#include "ioc.hpp"
using namespace std;

int main (int argc, char * const argv[]) 
{
    String name = argv[1];
	string basenames_txt_path		 = name + "/walk_basenames.txt";
	string demontraj_txt_path_prefix = name + "/walk_traj/";
	string feat_maps_xml_path_prefix = name + "/walk_feat/";
	string rect_imag_jpg_path_prefix = name + "/walk_imag/";
	string output_params_path		 = name + "/walk_output/walk_reward_params.txt";
	
	IOC model;
	model.name=name;
	
	model.loadBasenames		(basenames_txt_path);
	model.loadDemoTraj		(demontraj_txt_path_prefix);
	model.loadFeatureMaps	(feat_maps_xml_path_prefix);
	model.loadImages		(rect_imag_jpg_path_prefix); 

	model.initialize		(true,false);
	
	model.computeEmpiricalStatistics();
	
	bool converge = 0;
	while(!converge)
	{
		model.backwardPass();
		model.forwardPass ();
		model.gradientUpdate();
		model.saveParameters(output_params_path);
		converge = model._converged;
	}
	
	
	
    return 0;
}