/*
 *  ioc.cpp
 *  IOC_DEMO
 *
 *  Created by Kris Kitani on 11/28/12.
 *  Copyright 2012. All rights reserved.
 *
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ioc.hpp"
#include <assert.h> 

void IOC::loadBasenames	(string input_filename)
{
	cout << "\nLoadBasenames()\n";
	ifstream fs;
	fs.open(input_filename.c_str());
	if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	string str;
	while(fs >> str){
		if(str.find("#")==string::npos) _basenames.push_back(str);
	}
	_nd = (int)_basenames.size();
	cout << VERBOSE << endl;
	if(VERBOSE) cout << "  Number of basenames loaded:" << _nd << endl;
}


void IOC::loadDemoTraj	(string input_file_prefix)
{
	cout << "\nLoadDemoTraj()\n";
	for(int d=0;d<_nd;d++)
	{
		_trajgt.push_back(vector<cv::Point>(0));
		_trajob.push_back(vector<cv::Point>(0));
		
		string input_filename = input_file_prefix + _basenames[d]; 
		ifstream fs(input_filename.c_str());
		if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
		float val[5];
		int k=0;
		while(fs >> val[k++]){
			if(k==5)
			{
				_trajgt[d].push_back(cv::Point(val[1],val[2]));
				_trajob[d].push_back(cv::Point(val[3],val[4]));
				k=0;
			}
		}
		_start.push_back( _trajgt[d][0] );					// store start state
		_end.push_back  ( _trajgt[d][_trajgt[d].size()-1] );     	// store end state
		

		if(VERBOSE) printf("  %s: trajectory length is %d\n",_basenames[d].c_str(), (int)_trajgt[d].size());
	}
}

void IOC::loadFeatureMaps(string input_file_prefix)
{
	cout << "\nLoadFeatures()\n";
	
	for(int i=0;i<_nd;i++)
	{
		_featmap.push_back(vector<cv::Mat>(0));
		
		string input_filename = name + "/walk_feat/feature_map.xml";
		FileStorage fs(input_filename.c_str(), FileStorage::READ);
		if(!fs.isOpened()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
		for(int j=0;true;j++)
		{
			stringstream ss;
			ss << "feature_" << j;
			Mat tmp;
			fs[ss.str()] >> tmp;
			if(!tmp.data) break;
			_featmap[i].push_back(tmp+0.0);
		}
		_nf = (int)_featmap[i].size();
		_size = _featmap[i][0].size();
		if(VERBOSE) printf("  %s: Number of features loaded is %d\n",_basenames[i].c_str(), _nf);
	}
}


void IOC::loadImages		(string input_file_prefix)
{
	cout << "\nLoadImages()\n";
	
	for(int i=0;i<_nd;i++)
	{
		string input_filename = name+ "/walk_imag/topdown.jpg";
		Mat im = imread(input_filename);
		if(!im.data){cout << "ERROR: Opening:" << input_filename << endl; exit(1);}
		if(VERBOSE) cout << "  Loading: " << input_filename << endl;
		resize(im,im,_featmap[0][0].size());

		_image.push_back(im);
	}
	if(VERBOSE) cout << "  Number of images loaded: " << _image.size() << endl;
}


void IOC::computeEmpiricalStatistics()
{
	cout << "\nComputeEmpiricalStatistics()\n";
	
	
	
	for(int i=0;i<_nd;i++)
	{
		if(VERBOSE) cout << "  add feature counts for " << _basenames[i] << endl;
		
		for(int j=0;j<_trajgt[i].size();j++)
		{
			accumilateEmpiricalFeatureCounts(i,_trajgt[i][j]);
		}
	}
	
	for(int f=0;f<_nf;f++) _f_empirical[f] /= (float)_nd;
	
	cout << "  Mean empirical feature count:\n  ";
	for(int f=0;f<_nf;f++) cout << _f_empirical[f] << " "; cout << endl;
	
	
}

void IOC::initialize(bool verbose, bool visualize)
{
	cout << "\nInitialize()\n";
	
	VISUALIZE = visualize;						// visualization flag
	VERBOSE = verbose;							// print out intermediate status
	DELTA = 0.01;								// minimum improvement in loglikelihood
	_lambda = 0.01;								// initial step size
	
	_w = vector<float>	(_nf,0.5);				// initial parameter values
	
	_R = vector<Mat>	(_nd);
	for(int d=0;d<_nd;d++) 
		_R[d] = Mat::zeros(_size,CV_32FC1);
	
	_V = vector<Mat>	(_nd);
	for(int d=0;d<_nd;d++) 
		_V[d] = Mat::zeros(_size,CV_32FC1);
	
	_na = 9;
	_pax = vector< vector<Mat> > (_nd);
	for(int d=0;d<_nd;d++){
		_pax[d] = vector<Mat> (_na);
		for(int a=0;a<_na;a++)
			_pax[d][a] = Mat::zeros(_size,CV_32FC1);
	}
	
	_f_empirical = vector<float>(_nf,0);		// allocate memory
	_f_expected  = vector<float>(_nf,0);		// allocate memory
	_f_gradient  = vector<float>(_nf,0);		// allocate memory
	
	_minloglikelihood = -FLT_MAX;
	
}

void IOC::backwardPass()
{
	cout << "\nBackwardsPass()\n";
	
	_error = 0;
	_loglikelihood = 0;
	
	for(int d=0;d<_nd;d++)
	{
		cout << "TRAJECTORY: " << d << "\n";
		cout << _start[d] <<  "\n";
		cout << _end[d] << "\n";
		computeRewardFunction(d);
		computeSoftValueFunc(_R[d],_end[d],_image[d],_V[d]);
		computePolicy(_pax[d], _V[d], _na);
		computeTrajLikelihood(_pax[d],_trajgt[d],_loglikelihood);
		
		if(_loglikelihood <= -FLT_MAX) break;
	}
	
	cout << "  LogLikelihood SUM: " << _loglikelihood << endl;
}


void IOC::forwardPass ()
{
	
	cout << "\nForwardPass()\n";
	if(_error || (_loglikelihood < _minloglikelihood))
	{ 
		cout << "  Skip.\n";
		return;
	}
	
	
	for(int d=0;d<_nd;d++)
	{
		Mat D;
		computeStateVisDist(_pax[d],_start[d],_end[d],_image[d], D);
		accumilateExpectedFeatureCounts(D,_featmap[d]);
	}
	
	
	cout << "  Mean expected feature count:\n  ";
	for(int f=0;f<_nf;f++) cout << _f_expected[f] << " "; cout << endl;
}

void IOC::computeStateVisDist(vector<Mat> pax,Point start,Point end,Mat img, Mat &D)
{
	if(VERBOSE) cout << "  computeStateVisDist()\n";
	
	Mat N[2];
	N[0] = Mat::zeros(_size,CV_32FC1);
	N[1] = Mat::zeros(_size,CV_32FC1);
	N[0].at<float>(start.y,start.x) = 1.0;					// initialize start
	
	D = Mat::zeros(_size,CV_32FC1);
	
	D += N[0];
	
	Mat dsp;
	
	int n=0;
	
	while(1)
	{
		N[1] *= 0.0;
		for(int col=0;col<N[0].cols;col++)
		{
			for(int row=0;row<N[0].rows;row++)
			{
				if(row==end.y && col == end.x) continue;		// absorbsion state
				
				if(N[0].at<float>(row,col) > (FLT_MIN))			// ignore small probabilities
				{
					int col_1 = N[1].cols-1;
					int row_1 = N[1].rows-1;
					
					if(col>0	 && row>0	 )	N[1].at<float>(row-1,col-1) += N[0].at<float>(row,col) * pax[0].at<float>(row,col);	// NW
					if(				row>0	 )	N[1].at<float>(row-1,col-0) += N[0].at<float>(row,col) * pax[1].at<float>(row,col);	// N
					if(col<col_1 && row>0	 )	N[1].at<float>(row-1,col+1) += N[0].at<float>(row,col) * pax[2].at<float>(row,col);	// NE
					if(col>0				 )  N[1].at<float>(row-0,col-1) += N[0].at<float>(row,col) * pax[3].at<float>(row,col);	// W
					if(col<col_1             )	N[1].at<float>(row-0,col+1) += N[0].at<float>(row,col) * pax[5].at<float>(row,col);	// E
					if(col>0	 && row<row_1)	N[1].at<float>(row+1,col-1) += N[0].at<float>(row,col) * pax[6].at<float>(row,col);	// SW
					if(			    row<row_1)	N[1].at<float>(row+1,col-0) += N[0].at<float>(row,col) * pax[7].at<float>(row,col);	// S
					if(col<col_1 && row<row_1)	N[1].at<float>(row+1,col+1) += N[0].at<float>(row,col) * pax[8].at<float>(row,col);	// SE
				}
			}
		}
		N[1].at<float>(end.y,end.x) = 0.0;				// absorption state
		
		swap(N[0],N[1]);
		
		D += N[0];								// update state visitation distribution

		if(VISUALIZE)
		{
			colormap_CumilativeProb(D,dsp);
			img.copyTo(dsp,dsp<1);
			addWeighted(dsp,0.5,img,0.5,0,dsp);
			imshow("Forecast Distribution",dsp);
			waitKey(1);
		}
		
		if(n++>300) break;
	}
	return;
}

void IOC::accumilateExpectedFeatureCounts(Mat D, vector<Mat> featmaps)
{
	if(VERBOSE) cout << "  AccumilateExpectedFeatureCounts()\n";
	for(int f=0;f<_nf;f++)
	{
		Mat F = D.mul(featmaps[f]);
		_f_expected[f] += sum(F)(0)/(float)_nd;
	}
}

/*
 void IOC::computeLikelihood	()
 {
 if(VERBOSE) cout << "\nComputeLogLikelihood()\n";
 
 _loglikelihood = 0;
 
 for(int d=0;d<_nd;d++)
 {
 computeTrajLikelihood(_pax[d],_trajgt[d],_loglikelihood);
 }
 
 cout << "  LogLikelihood SUM: " << _loglikelihood << endl;
 
 }
 */

void IOC::computeTrajLikelihood	(vector<Mat> pax, vector<Point> trajgt, float &loglikelihood)
{
	if(VERBOSE) cout << "  ComputeTrajLikelihood()\n";
	
	float ll = 0;
	cout << "size of trajectory: " << trajgt.size() << "\n";
	
	for(int t=0;t<(int)trajgt.size()-1;t++)
	{
		int dx = trajgt[t+1].x - trajgt[t].x ;
		int dy = trajgt[t+1].y - trajgt[t].y ;
		
		int a = -1;
		if( dx==-1 && dy==-1 ) a = 0;
		if( dx== 0 && dy==-1 ) a = 1;
		if( dx== 1 && dy==-1 ) a = 2;
		
		if( dx==-1 && dy== 0 ) a = 3;
		if( dx== 0 && dy== 0 ) a =-1;	// stopping prohibited
		if( dx== 1 && dy== 0 ) a = 5;
		
		if( dx==-1 && dy== 1 ) a = 6;
		if( dx== 0 && dy== 1 ) a = 7;
		if( dx== 1 && dy== 1 ) a = 8;
		
		if(a<0)
		{
			printf("ERROR: Invalid action %d(%d,%d)\n" ,t,dx,dy);
			printf("Preprocess trajectory data properly.\n");
			exit(1);
		}
		
		float val = log(pax[a].at<float>(trajgt[t].y,trajgt[t].x));
		
		if(val < -FLT_MAX || isnan(val))
		{
			val = 0;
		}
		ll += val;
		
	}
	
	if(VERBOSE) cout << "    loglikelihood: " << ll << endl;
	
	loglikelihood += ll;
}

void IOC::gradientUpdate()
{
	_secondStep = true;
	cout << "\nGradientUpdate()\n";
	if(_error)
	{
		cout << "  ERROR. Increase step size.\n";
		for(int f=0;f<_nf;f++) _w[f] *= 2.0; 
	}

	float improvement = _loglikelihood - _minloglikelihood;	
	cout << "improvement " << improvement << "\n";
	if(improvement > DELTA) _minloglikelihood = _loglikelihood;
	else if(improvement < DELTA && improvement >= 0) improvement = 0;
	if(VERBOSE) cout << "  Improved by: " << improvement << endl;
	
	
	
	// ===== UPDATE PARAMETERS (STANDARD LINE SEARCH) ===== //
	
	if(improvement<0)
	{
		cout << "  ===> NO IMPROVEMENT: decrease step size and redo.\n";
		_lambda = _lambda * 0.5;
		for(int f=0;f<_nf;f++) _w[f] = _w_best[f] * exp( _lambda *  _f_gradient[f] );
	}
	else if(improvement>0)
	{
		cout << "  ***> IMPROVEMENT: increase step size.\n";
		_w_best = _w;
		_lambda = _lambda * 2.0;
		for(int f=0;f<_nf;f++) _f_gradient[f] = _f_empirical[f] - _f_expected[f];
		for(int f=0;f<_nf;f++) _w[f] = _w_best[f] * exp( _lambda * _f_gradient[f] );
		
	}
	else if(improvement==0)
	{
		cout << "  CONVERGED.\n";
		_converged = 1;		// converged.
	}
	
	if(VERBOSE)
	{
		cout << "  _lambda: " << _lambda << endl;
		
		//cout << "  _w_best: ";
		//for(int f=0;f<_nf;f++) cout << _w_best[f] << " ";
		//cout << endl;
		
		//cout << "  _w: ";
		//for(int f=0;f<_nf;f++) cout << _w[f] << " ";
		//cout << endl;
		
		cout << "  _f_empirical: ";
		for(int f=0;f<_nf;f++) cout << _f_empirical[f] << " ";
		cout << endl;
		
		cout << "  _f_expected: ";
		for(int f=0;f<_nf;f++) cout << _f_expected[f] << " ";
		cout << endl;
		cout << "  _f_gradient: ";
		for(int f=0;f<_nf;f++) cout << _f_gradient[f] << " ";
		cout << endl;
		cout << " _w ";
		for(int f=0;f<_nf;f++) cout << _w[f] << " ";
		cout << endl;
		cout << " _w_best ";
		for(int f=0;f<_nf;f++) cout << _w_best[f] << " ";
		cout << endl;
		
	}
	
}

void IOC::saveParameters(string output_filename)
{
	cout << "SAVING" << "\n";
	ofstream fs(output_filename.c_str());
	if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
	for(int f=0;f<(int)_w_best.size();f++){
		cout << _w_best[f] << "\n";
		fs << _w_best[f] << endl;
	}
}


void IOC::accumilateEmpiricalFeatureCounts(int data_i, cv::Point pt)
{
	for(int f=0;f<(int)_featmap[0].size();f++)
		_f_empirical[f] += _featmap[data_i][f].at<float>(pt.y,pt.x);
}


void IOC::computeRewardFunction(int data_i)
{
	//ssert (!_error);
	if(_error) return;
	if(VERBOSE) cout << "  ComputeRewardFunction()\n";
	_R[data_i] *= 0;

	for(int f=0;f<_nf;f++) {_R[data_i] += _w[f] * _featmap[data_i][f];}
	
	if(VISUALIZE)
	{
		Mat dst;
		colormap(_R[data_i],dst);
		addWeighted(_image[data_i],0.5,dst,0.5,0,dst);
		imshow("Reward Function",dst);
		waitKey(1);
	}
}


void IOC::computeSoftValueFunc(Mat R, Point end, Mat img, Mat &VF)
{
	//assert (!_error);
	if(_error) return;
	
	if(VERBOSE) cout << "  ComputeSoftValueFunc()\n";
	
	Mat V[2];
	V[0] = Mat::ones(R.size(),CV_32FC1) * -FLT_MAX;
	V[1] = Mat::ones(R.size(),CV_32FC1) * -FLT_MAX;
	
	int n = 0;
	float prev = FLT_MAX;
	
	while(1)
	{
		Mat V_padded;
		Mat v = V[0] * 1.0;
		copyMakeBorder(v,V_padded,1,1,1,1,BORDER_CONSTANT,Scalar::all(-FLT_MAX));
		V_padded *= 1.0;
		
		for(int col=0;col<(V_padded.cols-2);col++)
		{
			for(int row=0;row<(V_padded.rows-2);row++)
			{
				Mat sub = V_padded(Rect(col,row,3,3));
				double minVal, maxVal;
				minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
				
				if(maxVal==-FLT_MAX) continue;						// entire region has no data
				float softmax;
				float minv;
				float maxv;
				float subat;
				// ===== SOFTMAX ===== //
				for(int y=0;y<3;y++)								// softmax over actions (hard-coded)
				{
					for(int x=0;x<3;x++)							// softmax over actions
					{
						if(y==1 && x==1) continue;					// stopping prohibited
						
						minv = MIN(V[0].at<float>(row,col),sub.at<float>(y,x));
						maxv = MAX(V[0].at<float>(row,col),sub.at<float>(y,x));
						subat = sub.at<float>(y,x);
						/*cout << "calc softmax: " << maxv + log(1.0 + exp(minv-maxv)) << endl;
						cout << "subat " << subat << endl;*/

						if (_secondStep){
							/*cout << "minv: " << minv << endl;
							cout << "maxv: " << maxv << endl;
							cout << "softmax: " << softmax << endl;
							cout << "calc softmax: " << maxv + log(1.0 + exp(minv-maxv)) << endl;
							cout << "subat " << subat << endl;
							cout << "v[0] " << V[0].at<float>(row,col) << endl;*/
						}
						
						softmax = maxv + log(1.0 + exp(minv-maxv));
						if (softmax > 0){
							softmax = MAX(maxv, minv);
						}
						V[0].at<float>(row,col) = softmax;
					}
				}
				//imshow("test", V[0]);
				//waitKey(1);
				V[0].at<float>(row,col) += R.at<float>(row,col);	// asyncronus updates

				if(V[0].at<float>(row,col)>0)
				{
						double min, max;
						cv::minMaxLoc(V[0], &min, &max);

						cout << "Min: " << min << endl;
						cout << "Max: " << max << endl;
					cout << "row: " << row << endl;
					cout << "col: " << col << endl;
					cout << V[0].at<float>(row,col) << endl;
					cout << V[0].at<float>(row,col) - R.at<float>(row,col) << endl;
					cout << "minv: " << minv << endl;
					cout << "maxv: " << maxv << endl;
					cout << "softmax: " << softmax << endl;
					cout << "calc softmax: " << maxv + log(1.0 + exp(minv-maxv)) << endl;
					cout << "subat " << subat << endl;
					_error=1;
					return;											// error has occurred
				}
			}
		}
		V[0].at<float>(end.y,end.x) = 0.0;							// reset goal value to 0
		
		
		// ==== CONVERGENCE CRITERIA ==== //
		Mat residual;
		double minVal, maxVal;
		absdiff(V[0],V[1],residual);
		minMaxLoc(residual,&minVal,&maxVal,NULL,NULL);
		V[0].copyTo(V[1]);
		if(maxVal<0.9) break;
		else if (prev - maxVal < .01 && maxVal < 3) break;
		prev = maxVal;
		
		if(VISUALIZE)
		{
			Mat dst;
			colormap(V[0],dst);
			addWeighted(img,0.5,dst,0.5,0,dst);
			//imshow("MaxEnt Value Function",dst);
			//waitKey(1);
		}
		
		n++;
		if (n>2000) assert (false);
		if(n>2000){cout << "ERROR: Max number of iterations." << endl;_error=1;return;}
	}
	
	double min, max;
	cv::minMaxLoc(V[0], &min, &max);

	cout << "Min: " << min << endl;
	cout << "Max: " << max << endl;

	V[0].copyTo(VF);
	
	return;
}

void IOC::computePolicy(vector<Mat> &pax, Mat VF, int na)
{
	assert (!_error);
	if(_error) return;
	
	if(VERBOSE) cout << "  ComputePolicy()\n";
	
	double minVal, maxVal;
	Mat V_padded;
	copyMakeBorder(VF,V_padded,1,1,1,1,BORDER_CONSTANT,Scalar(-INFINITY));
	
	for(int col=0;col<V_padded.cols-2;col++)
	{
		for(int row=0;row<V_padded.rows-2;row++)
		{
			Rect r(col,row,3,3);
			Mat sub = V_padded(r);
			minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
			Mat p = sub - maxVal;				// log rescaling
			exp(p,p);							// Z(x,a) - probability space
			p.at<float>(1,1) = 0;				// zero out center
			Scalar su = sum(p);					// sum (denominator)
			if(su.val[0]>0) p /= su.val[0];		// normalize (compute policy(x|a))
			else p = 1.0/(na-1.0);				// uniform distribution
			p = p.reshape(1,1);					// vectorize
			for(int a=0;a<na;a++) pax[a].at<float>(row,col) = p.at<float>(0,a); // update policy
		}
	}
	for (int f = 0; f < pax.size(); f++){
		double min, max;
		cv::minMaxLoc(pax[f], &min, &max);

		cout << "Min " << f << ": " << min << endl;
		cout << "Max " << f << ": " << max << endl;
	}
}

void IOC::colormap(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	compare(src,-FLT_MAX,isInf,CMP_GT);
	threshold(src,src,-FLT_MAX,0,THRESH_TOZERO);
	minMaxLoc(src,&minVal,NULL,NULL,NULL);
	Mat im = (src-minVal)/(maxVal-minVal) * 255.0;
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);
	I3[0] = U8 * 0.85;
	I3[1] = isInf;
	I3[2] = isInf;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
}

void IOC::colormap_CumilativeProb(Mat src, Mat &dst)
{
	if(src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	
	Mat im;
	src.copyTo(im);
	
	double minVal = 1e-4;
	double maxVal = 0.2;
	threshold(im,im,minVal,0,THRESH_TOZERO);
	
	im = (im-minVal)/(maxVal-minVal)*255.0;
	
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);	
	I3[0] = U8*1.0;									// Hue
	
	Mat pU;
	U8.convertTo(pU,CV_64F,-1.0/255.0,1.0);
	pow(pU,0.5,pU);									
	pU.convertTo(U8,CV_8UC1,255.0,0);					
	I3[1] = U8*1.0;									// Saturation
	
	Mat isNonZero;
	compare(im,0,isNonZero,CMP_GT);
	I3[2] = isNonZero;								// Value
	
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);				// Convert to RGB
}
