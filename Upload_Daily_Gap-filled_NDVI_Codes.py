# coding:utf-8
# @Time    : 2023/10/25
# @Author  : Huiwen Li
# @FileName: Daily_Gap-filled_NDVI_Codes.py

import os
import time
import numpy as np
from osgeo import osr,ogr,gdal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
import scipy.interpolate as spi
from scipy.interpolate import interp1d
import scipy as sc
from scipy.optimize import minimize
from multiprocessing import Pool
import copy
from numba import jit
import numba

warnings.filterwarnings("ignore")

def read_all_files_inside_filepath(filepath,filetype,final_file_list):
	pathDir=os.listdir(filepath)
	for each in pathDir:
		newDir=os.path.join(filepath,each)
		if os.path.isfile(newDir):
			if os.path.splitext(newDir)[1]==filetype:
				final_file_list.append(newDir.replace("\\","/"))			
		else:
			read_all_files_inside_filepath(newDir,filetype,final_file_list)

def get_all_type_files(filepath,filetype):
	final_file_list=[]
	read_all_files_inside_filepath(filepath,filetype,final_file_list)
	return final_file_list

def read_raster_line_allcell_value(input_rasterfile,start_line_index,im_width,line_num):
	dataset=gdal.Open(input_rasterfile)
	line_allcell=dataset.ReadAsArray(0,start_line_index,im_width,line_num)
	line_allcell_setnull=np.around(np.array(line_allcell),decimals=3)
	line_allcell_setnull[line_allcell_setnull==-9999.0]=np.nan
	line_allcell_setnull=np.around(np.array(line_allcell_setnull*1000),decimals=0)
	return line_allcell_setnull.astype(str)

def get_processing(i,list_length):
	if int(i) in [int(list_length*0.1),int(list_length*0.2),int(list_length*0.3),int(list_length*0.4),int(list_length*0.5),int(list_length*0.6),int(list_length*0.7),int(list_length*0.8),int(list_length*0.9),list_length-1]:
		print(str(i)+', Progress:'+str(round((int(i)+1)*100.0/list_length,0))+'%')

def read_raster_ts_line_allcell(img_path_list,start_line_index,im_width,line_num):
	line_allcell_ts_value=np.full([line_num*len(img_path_list),im_width+2],'n').astype('U11')
	for i in img_path_list:
		i_index=img_path_list.index(i)
		cell_time=i.split('/')[-1][:8]
		line_allcell_setnull=read_raster_line_allcell_value(i,start_line_index,im_width,line_num)
		line_allcell_ts_value[i_index*line_num:(i_index+1)*line_num,0]=cell_time
		line_allcell_ts_value[i_index*line_num:(i_index+1)*line_num,1]=np.arange(start_line_index,start_line_index+line_num).astype(str)
		line_allcell_ts_value[i_index*line_num:(i_index+1)*line_num,2:]=line_allcell_setnull
		get_processing(img_path_list.index(i),len(img_path_list))
	line_allcell_ts_value[line_allcell_ts_value=='nan']='n'
	return line_allcell_ts_value

@jit()
def change_line_all_nan(list_2dim):
	for line in range(len(list_2dim)):
		if np.unique(list_2dim[line][2:]).shape[0]==1 and list_2dim[line][2]=='n':
			list_2dim[line]=[list_2dim[line][0],list_2dim[line][1],'ALLN']
		else:
			pass
	return list_2dim

def write_2dim_list_to_txt_file(csv_file,listdata):
	f=open(csv_file,'w')
	for eachi in listdata[:-1]:
		if len(eachi)==1:
			f.write(eachi[0]+'\n')
		else:
			for eachj in eachi[:-1]:
				f.write(eachj+',')
			f.write(eachi[-1]+'\n')
	for lasti in listdata[-1][:-1]:
		f.write(lasti+',')
	f.write(listdata[-1][-1])
	f.close()

def read_2dim_txt(txt_file):
	f=open(txt_file,'r')
	c=f.read()
	c_all=c.split('\n')
	all_data_list=[]
	all_time_list=[]
	max_column_num=0
	for i in range(len(c_all)):
		column_data=c_all[i].split(',')
		all_data_list.append(column_data)
		all_time_list.append(column_data[0])
		if max_column_num<len(column_data):
			max_column_num=len(column_data)
	f.close()
	return all_data_list,all_time_list,max_column_num

@jit()
def generate_line_all_line_array(thisline_all_data_list):
	this_line_all_column_array=np.full([len(thisline_all_data_list),7200],np.nan)
	for i in range(len(thisline_all_data_list)):
		if thisline_all_data_list[i][2]=='ALLN':
			pass
		else:
			initial_data=np.array(thisline_all_data_list[i][2:])
			initial_data[initial_data=='n']=np.nan
			this_line_all_column_array[i,:]=np.round(initial_data.astype(float)/1000,decimals=4)
	return this_line_all_column_array

def get_all_usefull_data_jit(daily_data,pd_time,threshold):
	first_year=int(pd_time[0,1])
	last_year=int(pd_time[-1,1])
	first_year_month=int(pd_time[0,2])
	last_year_month=int(pd_time[-1,2])
	month_all_max_value,all_max_v_moth_time=jit_get_usefull_data(first_year,last_year,daily_data,pd_time,first_year_month,last_year_month,threshold)
	usefull_ndvi_time_DF=pd.DataFrame(data=month_all_max_value,index=pd.to_datetime(all_max_v_moth_time,format='%Y%m%d'),columns=['NDVI'])
	sorted_usefull_ndvi_time_DF=usefull_ndvi_time_DF.sort_values(by='NDVI',ascending=False)
	if sorted_usefull_ndvi_time_DF.shape[0]>5:
		mean_ndvi_2_4=np.nanmean(sorted_usefull_ndvi_time_DF.iloc[np.arange(1,5)]['NDVI'])
	else:
		pass
	# If the maximum value in the entire time series is 39% greater than the mean of adjacent 4 data points, delete this outlier.
	return sorted_usefull_ndvi_time_DF[((sorted_usefull_ndvi_time_DF['NDVI']-mean_ndvi_2_4)/mean_ndvi_2_4<0.39) & (sorted_usefull_ndvi_time_DF['NDVI']>0)]

@jit()
def jit_get_usefull_data(first_year,last_year,daily_data,pd_time,first_year_month,last_year_month,threshold):
	month_all_max_value=[1]
	all_max_v_moth_time=[1]
	for year in range(first_year,last_year+1):
		if year == first_year:
			start_mon=first_year_month
			end_mon=12
		elif year== last_year:
			start_mon=1
			end_mon=last_year_month
		else:
			start_mon=1
			end_mon=12
		for m in range(start_mon,end_mon+1):   
			this_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m)]
			this_m_time=pd_time[(pd_time[:,1]==year) & (pd_time[:,2]==m)][:,0] 
			sorted_index=np.argsort(this_m_value)	
			# If the monthly max value is 20% larger than any other values in this month, delete this outlier.
			if this_m_value[sorted_index[-1]]/this_m_value[sorted_index[-2]]>1.2:
				this_m_value=np.delete(this_m_value,sorted_index[-1])
				this_m_time=np.delete(this_m_time,sorted_index[-1])			
			this_m_max_value=np.nanmax(this_m_value)
			if year!=first_year and year != last_year:
				if m>1 and m<12:
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)	
				elif m==1:
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year-1) & (pd_time[:,2]==12)]
					last_m_max_value=np.nanmax(last_m_value)	
				elif m==12:
					next_m_value=daily_data[(pd_time[:,1]==year+1) & (pd_time[:,2]==1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)
				this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value,next_m_max_value])
				this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value,next_m_max_value])
				month_all_max_value=np.hstack([month_all_max_value,this_m_value[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
				all_max_v_moth_time=np.hstack([all_max_v_moth_time,this_m_time[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
			elif year==first_year:
				if m==first_year_month:		
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					this_next_m_max_value_mean=np.nanmean([this_m_max_value,next_m_max_value])
					this_next_m_max_value_min=np.nanmin([this_m_max_value,next_m_max_value])
				elif m<12:
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)
					this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value,next_m_max_value])
					this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value,next_m_max_value])
				elif m==12:
					next_m_value=daily_data[(pd_time[:,1]==year+1) & (pd_time[:,2]==1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)
					this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value,next_m_max_value])
					this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value,next_m_max_value])
				month_all_max_value=np.hstack([month_all_max_value,this_m_value[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
				all_max_v_moth_time=np.hstack([all_max_v_moth_time,this_m_time[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
			elif year==last_year:
				if m==1 and m!=last_year_month:
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year-1) & (pd_time[:,2]==12)]
					last_m_max_value=np.nanmax(last_m_value)
					this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value,next_m_max_value])
					this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value,next_m_max_value])
				elif m<12 and m!=last_year_month:
					next_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m+1)]
					next_m_max_value=np.nanmax(next_m_value)
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)
					this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value,next_m_max_value])
					this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value,next_m_max_value])
				elif m==last_year_month:
					last_m_value=daily_data[(pd_time[:,1]==year) & (pd_time[:,2]==m-1)]
					last_m_max_value=np.nanmax(last_m_value)
					this_next_m_max_value_mean=np.nanmean([last_m_max_value,this_m_max_value])
					this_next_m_max_value_min=np.nanmin([last_m_max_value,this_m_max_value])
				month_all_max_value=np.hstack([month_all_max_value,this_m_value[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
				all_max_v_moth_time=np.hstack([all_max_v_moth_time,this_m_time[(this_m_value/this_next_m_max_value_mean>=threshold) | (this_m_value/this_next_m_max_value_min>=threshold)].tolist()])
	
	return month_all_max_value[1:],all_max_v_moth_time[1:]

def fit_data_with_annual_piecewise_fitting_nppolyfit(pd_time,usefull_ndvi_time_DF):
	all_data=np.full(pd_time.shape[0],fill_value=-1.0)
	all_years=np.unique(usefull_ndvi_time_DF.index.year)
	for y in all_years:
		if y>all_years[0] and y<all_years[-1]:
			this_year_usefull_data_DF=usefull_ndvi_time_DF[(usefull_ndvi_time_DF.index.year==y) | ((usefull_ndvi_time_DF.index.year==y-1) & (usefull_ndvi_time_DF.index.month>=10)) | ((usefull_ndvi_time_DF.index.year==y+1) & (usefull_ndvi_time_DF.index.month<=3))]
		elif y==all_years[0]:
			this_year_usefull_data_DF=usefull_ndvi_time_DF[(usefull_ndvi_time_DF.index.year==y) | ((usefull_ndvi_time_DF.index.year==y+1) & (usefull_ndvi_time_DF.index.month<=3))]
		else:
			this_year_usefull_data_DF=usefull_ndvi_time_DF[(usefull_ndvi_time_DF.index.year==y) | ((usefull_ndvi_time_DF.index.year==y-1) & (usefull_ndvi_time_DF.index.month>=10))]
		this_year_usefull_data_DF=this_year_usefull_data_DF.sort_index()
		this_year_usefull_time_index=np.array([np.argwhere(pd_time==i)[0,0] for i in this_year_usefull_data_DF.index])+1
		this_year_all_data_time=pd_time[pd_time.year==y]
		this_year_all_data_time_index=np.array([np.argwhere(pd_time==i)[0,0] for i in this_year_all_data_time])+1
		all_data[this_year_all_data_time_index-1]=fit_this_year_data_piecewise_fitting_nppolyfit(y,this_year_usefull_time_index,this_year_usefull_data_DF,this_year_all_data_time_index)
	return all_data

@jit()
def fit_this_year_data_piecewise_fitting_nppolyfit(y,this_year_usefull_time_index,this_year_usefull_data_DF,this_year_all_data_time_index):
	if y==1981 or y==2023:
		polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
		this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)
	else:
		if this_year_usefull_data_DF['NDVI'].idxmax().year==y:
			max_value_index=np.argwhere(this_year_usefull_data_DF.index==this_year_usefull_data_DF['NDVI'].idxmax())[0,0]
			first_this_year_usefull_time_index=this_year_usefull_time_index[:max_value_index+10]
			first_this_year_usefull_data_DF=this_year_usefull_data_DF[:max_value_index+10]
			second_this_year_usefull_time_index=this_year_usefull_time_index[max_value_index-10:]
			second_this_year_usefull_data_DF=this_year_usefull_data_DF[max_value_index-10:]
			if max_value_index+9<this_year_usefull_time_index.shape[0] and max_value_index-10>0:
				if pd.to_datetime(this_year_usefull_data_DF.iloc[max_value_index+9].name).year==y and pd.to_datetime(this_year_usefull_data_DF.iloc[max_value_index-10].name).year==y:
					first_this_year_all_data_time_index=this_year_all_data_time_index[:np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index+9])[0,0]]
					first_overlap_num=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index+9])[0,0]]-this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index])[0,0]]
					second_this_year_all_data_time_index=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index-10])[0,0]:]
					second_overlap_num=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index])[0,0]]-this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[max_value_index-10])[0,0]]
					first_polyfit_fun=np.polyfit(first_this_year_usefull_time_index, first_this_year_usefull_data_DF.values.reshape(-1), 4)
					first_this_year_all_spline_values = np.polyval(first_polyfit_fun,first_this_year_all_data_time_index)
					second_polyfit_fun=np.polyfit(second_this_year_usefull_time_index, second_this_year_usefull_data_DF.values.reshape(-1), 4)
					second_this_year_all_spline_values = np.polyval(second_polyfit_fun,second_this_year_all_data_time_index)
					overlap_sum_num=first_overlap_num+second_overlap_num
					this_year_all_spline_values=np.hstack([first_this_year_all_spline_values[:-overlap_sum_num],np.mean([first_this_year_all_spline_values[-overlap_sum_num:],second_this_year_all_spline_values[:overlap_sum_num]],axis=0),second_this_year_all_spline_values[overlap_sum_num:]])
				elif pd.to_datetime(this_year_usefull_data_DF.iloc[max_value_index+9].name).year!=y or pd.to_datetime(this_year_usefull_data_DF.iloc[max_value_index-10].name).year!=y:
					polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
					this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)
			else:
				polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
				this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)	
		elif this_year_usefull_data_DF['NDVI'].idxmax().year!=y and this_year_usefull_data_DF['NDVI'].idxmin().year==y:			
			min_value_index=np.argwhere(this_year_usefull_data_DF.index==this_year_usefull_data_DF['NDVI'].idxmin())[0,0]
			first_this_year_usefull_time_index=this_year_usefull_time_index[:min_value_index+10]
			first_this_year_usefull_data_DF=this_year_usefull_data_DF[:min_value_index+10]
			second_this_year_usefull_time_index=this_year_usefull_time_index[min_value_index-10:]
			second_this_year_usefull_data_DF=this_year_usefull_data_DF[min_value_index-10:]
			if min_value_index+9<this_year_usefull_time_index.shape[0] and min_value_index-10>0:
				if this_year_usefull_time_index[min_value_index+9]>this_year_all_data_time_index[-1] or this_year_usefull_time_index[min_value_index-10]<this_year_all_data_time_index[0] or min_value_index-10<0:
					polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
					this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)
				else:
					first_this_year_all_data_time_index=this_year_all_data_time_index[:np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index+9])[0,0]]
					first_overlap_num=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index+9])[0,0]]-this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index])[0,0]]
					second_this_year_all_data_time_index=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index-10])[0,0]:]
					second_overlap_num=this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index])[0,0]]-this_year_all_data_time_index[np.argwhere(this_year_all_data_time_index==this_year_usefull_time_index[min_value_index-10])[0,0]]
					first_polyfit_fun=np.polyfit(first_this_year_usefull_time_index, first_this_year_usefull_data_DF.values.reshape(-1), 4)
					first_this_year_all_spline_values = np.polyval(first_polyfit_fun,first_this_year_all_data_time_index)
					second_polyfit_fun=np.polyfit(second_this_year_usefull_time_index, second_this_year_usefull_data_DF.values.reshape(-1), 4)
					second_this_year_all_spline_values = np.polyval(second_polyfit_fun,second_this_year_all_data_time_index)
					overlap_sum_num=first_overlap_num+second_overlap_num
					this_year_all_spline_values=np.hstack([first_this_year_all_spline_values[:-overlap_sum_num],np.mean([first_this_year_all_spline_values[-overlap_sum_num:],second_this_year_all_spline_values[:overlap_sum_num]],axis=0),second_this_year_all_spline_values[overlap_sum_num:]])
		
			else:
				polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
				this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)
		else:
			polyfit_fun=np.polyfit(this_year_usefull_time_index, this_year_usefull_data_DF.values.reshape(-1), 4)
			this_year_all_spline_values = np.polyval(polyfit_fun,this_year_all_data_time_index)
	return this_year_all_spline_values

def get_MSE_RMSE_R2_for_predicted(observations_array,predicted_array):
	mse=mean_squared_error(observations_array,predicted_array)
	rmse=np.sqrt(mse)
	r2=r2_score(observations_array,predicted_array)
	pb=(np.nanmean(predicted_array)-np.nanmean(observations_array))*100.0/np.nanmean(observations_array)
	return rmse,r2,pb

def get_processing_fiting(line,i,list_length,starttime,endtime):
	if int(i) in [int(list_length*0.1),int(list_length*0.2),int(list_length*0.3),int(list_length*0.4),int(list_length*0.5),int(list_length*0.6),int(list_length*0.7),int(list_length*0.8),int(list_length*0.9),list_length-1]:
		print('l:'+str(line)+'_c:'+str(i)+', Progress:'+str(round((int(i)+1)*100.0/list_length,0))+'%: '+str(round((endtime-starttime)/60.0,4))+"  min; ")

def read_2dim_txt_simple(txt_file):
	f=open(txt_file,'r')
	c=f.read()
	c_all=c.split('\n')
	all_data_list=[]
	for i in range(len(c_all)):
		column_data=c_all[i].split(',')
		all_data_list.append(column_data)
	f.close()
	return all_data_list

def write_rasterfile_withnodata(output_rasterfile,im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue):	
	driver=gdal.GetDriverByName("GTiff")
	datasetnew=driver.Create(output_rasterfile,im_width,im_height,im_bands,gdal.GDT_Float32)
	datasetnew.SetGeoTransform(im_geotrans)
	datasetnew.SetProjection(im_proj)
	datasetnew.GetRasterBand(1).SetNoDataValue(float(NoDataValue))
	datasetnew.GetRasterBand(1).WriteArray(im_data)
	del datasetnew

def read_rasterfile(input_rasterfile):
	dataset=gdal.Open(input_rasterfile)
	im_width=dataset.RasterXSize
	im_height=dataset.RasterYSize
	im_bands=dataset.RasterCount
	im_geotrans=dataset.GetGeoTransform()
	im_proj=dataset.GetProjection()
	im_data=dataset.ReadAsArray(0,0,im_width,im_height) 
	NoDataValue=dataset.GetRasterBand(1).GetNoDataValue()
	return [im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]



def write_ts_line_allcell_totxt(avhrr_tif_path,save_txt_filepath,line_index_list):
	starttime=time.time()
	all_daily_avhrr_list=get_all_type_files(avhrr_tif_path,'.tif')
	all_daily_avhrr_list.sort()	
	print('file found, reading...')
	start_line_index=line_index_list[0]
	im_width=7200
	line_num=len(line_index_list)
	line_allcell_ts_value=read_raster_ts_line_allcell(all_daily_avhrr_list,start_line_index,im_width,line_num)
	endtime=time.time()
	print('reading file time: '+str(round((endtime-starttime)/60.0,2))+"  min")
	for i in line_index_list:
		starttime1=time.time()
		save_file=save_txt_filepath+'line_['+str(i)+'].txt'
		if os.path.exists(save_file) and os.path.getsize(save_file) > 2048:
			pass
		else:
			print('data loaded, writing txt...')
			this_line_all_data=line_allcell_ts_value[line_allcell_ts_value[:,1]==str(i)].tolist()
			this_line_all_data=change_line_all_nan(this_line_all_data)
			write_2dim_list_to_txt_file(save_file,this_line_all_data)
		endtime2=time.time()
		print('line: '+str(i) +' write done, consuming time: '+str(round((endtime2-starttime1)/60.0,2))+"  min")

def fit_line_all_column_data_China(txt_filepath):
	all_start_time=time.time()
	threshold=0.8  #1-0.2
	save_txt_filepath="Fitted NDVI txt file path"
	thisline_all_data_list,all_time_list,max_column_num=read_2dim_txt(txt_filepath)
	have_data_pd_time=np.vstack([all_time_list,[i[:4] for i in all_time_list],[i[4:6] for i in all_time_list],[i[6:] for i in all_time_list]]).T
	have_data_pd_time=have_data_pd_time.astype(int)
	line_number=thisline_all_data_list[0][1]
	print('dealing line '+str(line_number)+' txt...')
	start_date=thisline_all_data_list[0][0]
	end_date=thisline_all_data_list[-1][0]
	save_file=save_txt_filepath+'line_['+str(line_number)+'].txt'
	if max_column_num==3:
		print(str(line_number)+' all nan...')
		listdata=[[line_number,start_date,end_date,'ALLN']]
		write_2dim_list_to_txt_file(save_file,listdata)
	else:
		fitting_effect=np.full([3,1401],'n').astype('U11')
		fitting_effect[0,0]='rmse'
		fitting_effect[1,0]='r2'
		fitting_effect[2,0]='pb'
		fitting_effect_save_file="Fitted NDVI effect txt file path/"+'line_['+str(line_number)+'].txt'
		print('fitting line '+str(line_number)+' data...')
		pd_time=pd.date_range(start=start_date,end=end_date,freq='D')
		this_line_save_data=np.full([pd_time.shape[0],1401],'n').astype('U11')
		this_line_save_data[:,0]=pd_time.strftime('%Y%m%d') 
		this_line_all_column_array=generate_line_all_line_array(thisline_all_data_list)
		print('all column data loaded...')
		starttime=time.time()
		for c in range(5000,6400):
			starttime1=time.time()
			if np.isnan(this_line_all_column_array[:,c]).sum()==this_line_all_column_array.shape[0]:
				pass  
				endtime1=time.time()
			else:
				thiscolumn_alldaily_data=this_line_all_column_array[:,c]
				if np.isnan(this_line_all_column_array[:,c]).sum()/this_line_all_column_array.shape[0]>0.3:
					pass 
					endtime1=time.time()
				else:
					usefull_ndvi_time_DF=get_all_usefull_data_jit(thiscolumn_alldaily_data,have_data_pd_time,threshold)
					endtime1=time.time()
					this_line_fitted_data=fit_data_with_annual_piecewise_fitting_nppolyfit(pd_time,usefull_ndvi_time_DF)
					this_line_fitted_data_DF=pd.DataFrame(index=pd_time,data=this_line_fitted_data,columns=['NDVI'])
					rmse,r2,pb=get_MSE_RMSE_R2_for_predicted(usefull_ndvi_time_DF.values.reshape(-1),this_line_fitted_data_DF.loc[usefull_ndvi_time_DF.index].values.reshape(-1))
					fitting_effect[0,c-5000+1]=rmse
					fitting_effect[1,c-5000+1]=r2
					fitting_effect[2,c-5000+1]=pb
					this_line_save_data[:,c-5000+1]=np.round(this_line_fitted_data,decimals=3)
			endtime2=time.time()
			get_processing_fiting(line_number,c-5000,np.arange(5000,6400).shape[0],starttime,endtime2)
		write_2dim_list_to_txt_file(save_file,this_line_save_data.tolist())
		write_2dim_list_to_txt_file(fitting_effect_save_file,fitting_effect.tolist())
		all_endtime=time.time()
		print(str(line_number)+' done: '+str(round((all_endtime-all_start_time)/60.0,4))+"  min")
	
def convert_fitting_results_to_raster(input_filepath,out_put_raster_path):
	RMSE_array=np.full([800,1400],-999.0)
	r2_array=np.full([800,1400],-999.0)
	pb_array=np.full([800,1400],-999.0)
	all_txt_file=get_all_type_files(input_filepath,'.txt')
	for i in range(len(all_txt_file)):
		line_index=int(all_txt_file[i].split('[')[-1].split(']')[0])-700
		this_line_rmse_r2_pb=read_2dim_txt_simple(all_txt_file[i])
		RMSE=np.array(this_line_rmse_r2_pb[0][1:])
		r2=np.array(this_line_rmse_r2_pb[1][1:])
		pb=np.array(this_line_rmse_r2_pb[2][1:])
		RMSE[RMSE=='n']=-999.0
		r2[r2=='n']=-999.0
		pb[pb=='n']=-999.0
		RMSE=RMSE.astype(float)
		r2=r2.astype(float)
		pb=pb.astype(float)
		RMSE_array[line_index]=RMSE
		r2_array[line_index]=r2
		pb_array[line_index]=pb
		get_processing(i,len(all_txt_file))
	nc_ref_data='A reference NOAA CDR NDVI nc file'
	f = Dataset(nc_ref_data)
	v = f.variables
	lon=v['longitude'][:]
	lat=v['latitude'][:]
	lon_resoltion=lon[1]-lon[0]
	lat_resoltion=lat[1]-lat[0]
	im_width=1400
	im_height=800
	im_bands=1
	im_geotrans=[lon[5000],lon_resoltion,0.0,lat[700],0.0,lat_resoltion]
	im_proj='''GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'''
	rmse_file=out_put_raster_path+'RMSE.tif'
	r2_file=out_put_raster_path+'r2.tif'
	pb_file=out_put_raster_path+'pb.tif'
	write_rasterfile_withnodata(rmse_file,RMSE_array,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile_withnodata(r2_file,r2_array,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile_withnodata(pb_file,pb_array,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)

def convert_fitted_txt_to_raster(fitted_txt_fileptah,start_index,end_index,output_raster_path):
	lines_data=np.full([15296,end_index-start_index,1400],-9.0)
	for i in range(start_index,end_index):
		this_line_txtpath=fitted_txt_fileptah+'line_['+str(i)+'].txt'
		line_index=i-start_index
		this_line_all_data=np.array(read_2dim_txt_simple(this_line_txtpath))
		this_line_all_data[this_line_all_data=='n']=-9.0
		if i==start_index:
			start_time=this_line_all_data[0,0]
			end_time=this_line_all_data[-1,0]
			all_time_array=this_line_all_data[:,0]
		lines_data[:,line_index,:]=this_line_all_data[:,1:].astype(float)
		get_processing(i-start_index,end_index-start_index)
	lines_data[(lines_data<-1) | (lines_data>1)]=-9.0
	nc_ref_data='A reference NOAA CDR NDVI nc file'
	f = Dataset(nc_ref_data)
	v = f.variables
	lon=v['longitude'][:]
	lat=v['latitude'][:]
	lon_resoltion=lon[1]-lon[0]
	lat_resoltion=lat[1]-lat[0]
	im_width=1400
	im_height=end_index-start_index
	im_bands=1
	im_geotrans=[lon[5000],lon_resoltion,0.0,lat[start_index],0.0,lat_resoltion]
	im_proj='''GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'''
	print('converting data...')
	for d in range(lines_data.shape[0]):
		year=all_time_array[d][:4]
		fitted_raster_file=output_raster_path+year+'/NDVI_'+str(start_index)+'-'+str(end_index)+'_'+all_time_array[d]+'.tif'
		write_rasterfile_withnodata(fitted_raster_file,lines_data[d],im_width,im_height,im_bands,im_geotrans,im_proj,-9.0)
		get_processing(d,lines_data.shape[0])

def mosaic_rasters2one(input_filepath,output_raster_path):
	all_raster_file=get_all_type_files(input_filepath,'.tif')
	days=int(len(all_raster_file)/6.0)
	[im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]=read_rasterfile(all_raster_file[days*3])
	print(all_raster_file[days*3].split('NDVI_')[-1])
	for i in range(days):
		file_700_840=all_raster_file[i+days*3]
		file_840_980=all_raster_file[i+days*4]
		file_980_1120=all_raster_file[i+days*5]
		file_1120_1260=all_raster_file[i]
		file_1260_1400=all_raster_file[i+days*1]
		file_1400_1500=all_raster_file[i+days*2]
		data_700_840=read_rasterfile(file_700_840)[0]
		data_840_980=read_rasterfile(file_840_980)[0]
		data_980_1120=read_rasterfile(file_980_1120)[0]
		data_1120_1260=read_rasterfile(file_1120_1260)[0]
		data_1260_1400=read_rasterfile(file_1260_1400)[0]
		data_1400_1500=read_rasterfile(file_1400_1500)[0]
		this_year_daily_rater=np.vstack([data_700_840,data_840_980,data_980_1120,data_1120_1260,data_1260_1400,data_1400_1500])
		this_year_data_file=output_raster_path+'CHN_NDVI_'+all_raster_file[i].split('_')[-1]
		im_width=1400
		im_height=1500-700
		im_bands=1
		im_proj='''GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'''
		write_rasterfile_withnodata(this_year_data_file,this_year_daily_rater,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue)
		get_processing(i,days)


def main():
	os.chdir('The project file path')
	'''
	Directly processing all daily-scale raster data would consume a significant amount of computer memory. 
	In this study, data is processed line by line, and all rasters are initially converted into txt files.
	'''
	#1. Store all daily scale raters as txt files in rows
	avhrr_tif_path='NOAA CDR daily NDVI raster file path'
	save_txt_filepath="The conerted txt file path"
	for i in np.arange(0,3600,1):
		line_index_list=np.arange(i,i+1).tolist()		
		write_ts_line_allcell_totxt(avhrr_tif_path,save_txt_filepath,line_index_list)
	#2. Read txt files within China for NDVI reconstruction
	all_txt_file=get_all_type_files("The conerted txt file path",'.txt')
	all_txt_file.sort(key=len)
	# Data range of China: lines: 700~1500; Columns: 5000~6400
	for i in range(700,1501):
		fit_line_all_column_data_China(all_txt_file[i])
	#3. Convert the reconstruction effect txt file to raster file
	input_filepath='Reconstruction effect txt file path'
	out_put_raster_path='Reconstruction effect raster file path'
	convert_fitting_results_to_raster(input_filepath,out_put_raster_path)
	#4. Convert the reconstructed NDVI txt files to raster files
	fitted_txt_fileptah='Fitted NDVI txt file path'
	output_raster_path='Output NDVI raster file path'
	for i in np.linspace(700,1400,6)[:]:
		start_index=int(i)
		if i==1400:
			end_index=int(i+100)
		else:
			end_index=int(i+140)
		convert_fitted_txt_to_raster(fitted_txt_fileptah,start_index,end_index,output_raster_path)
	#5. Mosaic multi-line raster to one complete raster file
	for year in range(1981,2024):
		input_filepath='Input reconstructed multi-line NDVI raster file path/'+str(year)+'/'
		output_raster_path='File path of the Reconstructed NDVI raster in China/'+str(year)+'/'
		mosaic_rasters2one(input_filepath,output_raster_path)
		print(str(year)+' done')


if __name__ == '__main__':
	main()