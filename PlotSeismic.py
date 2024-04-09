# PlotSeismic

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import os
import tkinter as tk
import sys
import time
import shelve
import ast
import pandas as pd
from ipywidgets import interact, widgets
from matplotlib.widgets import Slider
from obspy import read , UTCDateTime , Stream, Trace, Inventory
from obspy.core.inventory import read_inventory 
from obspy.clients.fdsn.client import Client
from obspy.signal.invsim import cosine_taper
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.cross_correlation import correlate
from datetime import datetime , timedelta 
from scipy.signal import spectrogram
from scipy.fft import fft
from threading import Thread
from tkinter import simpledialog , filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


class SeismicConfig:
    def __init__(self):
        self.start_time_global = None
        self.end_time_global = None
        self.network = 'XA'
        self.station = 'S12'
        self.channel = 'MHZ'
        self.location = '*'
        self.pre_filt = [0.0005, 0.005, 0.005, 0.03]
        self.cache_stream = "path/to/your/cache/Cache_seismic/stream.db"
        self.cache_stations = "path/to/your/cache/Cache_stations"
        self.save_folder = "path/to/your/DATA"
        self.CSL_info = None
        self.CSL_stream = None
        self.cache_streams_list = []
        self.current_file_path = None


class SeismicProcessor:
    def __init__(self, config = None,data_file_path = None):
        self.stream = Stream()
        self.config = SeismicConfig()
        self.config = config
        if data_file_path == None:
            self.data_file_path = os.path.dirname(__file__)
        else:
            self.data_file_path = data_file_path
        self.file_name = os.path.basename(self.data_file_path)

    def main(self):
    
        #print("设置缓存文件夹:")
        #print(f'cache_stream : {self.config.cache_stream}')
        #print(f'cache_stations : {self.config.cache_stations}')
        #print('设置缓存文件夹（SAC数据）:')
        #print(f'save_folder = {self.config.save_folder}')
        #print(f"  network={self.config.network}\n  station={self.config.station}\n  channel={self.config.channel}\n  location={self.config.location}")
        #print(f"pre_filt = {self.config.pre_filt}")

        # plot setting
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = 10, 4
        plt.rcParams['lines.linewidth'] = 0.2
        plt.rcParams['font.size'] = 12
        SECONDS_PER_DAY=3600.*24

        # Time Convertion
        #year = 71
        #day_of_year = 107
        #start_time = 704  # 7:04 AM
        #stop_time = 900  # 9:00 AM

        print("Input Date of '0' and input UTC Date")
        input_string = "  71 192 1327 1327"         
        input_string = self.get_user_input("   Input Date of '  Year  Days_of_Year  StartTime  StopTime  '  ",initial_value=input_string)
        self.config.start_time_global, self.config.end_time_global  =  SeismicProcessor.convert_time_format(input_string)
        
        print(f"Date :{self.config.start_time_global}--{self.config.end_time_global}\n")

        ## Set start and end time
        if input_string == "0":
            start_time='1971-03-13T07:30:00.0';end_time='1971-03-13T09:30:00.0'
            initial_values=(start_time,end_time)
            prompt=["  Start Time :  ","  End Time :  "]
            initial_values= self.get_user_inputs(2, prompt , initial_values)
            start_time=initial_values[0]
            end_time = initial_values[1]
            self.config.start_time_global= UTCDateTime(start_time)
            self.config.end_time_global = UTCDateTime(end_time)
            print(f"Date :{self.config.start_time_global}--{self.config.end_time_global}\n")
        
        return self.config
        
    def read_seismic_data(self):

        print(f"Open file path:{self.data_file_path}")
        stime = time.time()

        try:
            self.stream = read(self.data_file_path)
            # print("self.trace_info_list:",self.trace_info_list)
            
            print("文件格式为 ObsPy 的 Stream 格式。")
        except:
            # ObsPy 的 read() 函数抛出异常，说明文件格式不是 Stream 格式
            print("文件格式不是 ObsPy 的 Stream 格式。")
            # use other ways to read file 
            # get filename's suffix
            file_extension = self.data_file_path.split('.')[-1]

            # 根据文件后缀选择相应的读取方法

            if file_extension.lower() == 'ascii':
                self.read_ascii_data()
            elif file_extension.lower() == 'txt':
                self.read_txt_data()
            elif file_extension.lower() == 'csv':
                [data1] = self.read_single_column_from_csv()
        self.config.CSL_info = f"local file {os.path.basename(self.data_file_path)}"
        self.config.CSL_stream = self.stream
        self.read_traceinfo_from_stream()
        etime = time.time()
        execution_time = etime - stime
        print(f"File read time: {execution_time} seconds!\n")

        print(self.stream)

        return self.stream
    

    def read_traceinfo_from_stream(self):
        # 初始化空列表，用于存储当前stream中每个Trace对象的信息
        self.trace_info_list = []
        
        # 遍历当前stream中的每个Trace对象
        for trace in self.stream:
            # 获取当前Trace对象的信息
            trace_info = {
                'start_time_global': trace.stats.starttime,
                'end_time_global': trace.stats.endtime,
                'network': trace.stats.network,
                'station': trace.stats.station,
                'channel': trace.stats.channel,
                'location': trace.stats.location
            }
            # 将当前Trace对象的信息添加到列表中
            self.trace_info_list.append(trace_info)

        if self.trace_info_list:  # 检查列表不为空
            self.config.start_time_global = self.trace_info_list[0]['start_time_global']
            self.config.end_time_global = self.trace_info_list[0]['end_time_global']

        return self.trace_info_list

    # 读取文件
    def read_txt_data(self):
        """
        从文本文件中读取数据并返回一个Stream对象
        """
        traces = []
        with open(self.data_file_path, 'r') as file:
            lines = file.readlines()
            
            trace_data = None
            metadata = {}
            for line in lines:
                if line.startswith("Trace Data:"):
                    trace_data = [int(sample.strip()) for sample in line.split(":")[1].split(",") if sample.strip()]
                elif line.startswith("Stats Information:"):
                    metadata = {}
                elif line.startswith("--------"):
                    if trace_data and metadata:
                        trace = Trace(trace_data)
                        trace.stats = trace.Stats(metadata)
                        traces.append(trace)
                        trace_data = None
                        metadata = {}
                elif ":" in line:
                    key, value = line.split(":")
                    metadata[key.strip()] = value.strip()

        return self.stream(traces=traces)
    
    def read_ascii_data(self):
        with open(self.data_file_path, 'r') as file:
            lines = file.readlines()
            # 提取数据列
            time = []
            trace_data = []
            for line in lines:
                # 如果当前行为空行，则跳过
                if not line.strip():
                    continue
                data = line.strip().split()  # 假设数据列以空格分隔
                time.append(float(data[0]))  # 假设第一列是时间数据
                trace_data.append(float(data[1]))  # 假设第二列是地震强度数据
        # 创建 Trace 对象并将数据添加到其中
        trace = Trace(data=np.array(trace_data))

        # 设置 Trace 的元数据
        filename = os.path.basename(self.data_file_path)
        name_elements = filename.split('.')
        trace.stats.starttime = UTCDateTime(time[100])  # 假设第一列的第一个值是起始时间
        trace.stats.sampling_rate = 1/(time[1] - time[0])
        trace.stats.network = name_elements[0]
        trace.stats.station = name_elements[1]
        trace.stats.channel = name_elements[2]
        trace.stats.location = name_elements[3]

        # 创建 Stream 对象并将 Trace 添加到其中
        self.stream += trace

        return  self.stream
    
    def read_single_column_from_csv(self, column_name = None, plot = False):
        """
        ## 从CSV文件中读取单列数据并绘图。
        Parameters:
            file_path (str): CSV文件的路径。
            column_name (str): 想要绘制的列的名称。如果为None, 则默认使用第一列数据。
        Returns:
            column_data: 单列数据 
        """
        # 读取CSV文件
        data = pd.read_csv(self.data_file_path)

        # 如果column_name为None，则默认使用第一列数据
        if column_name is None:
            column_name = data.columns[0]

        # 提取单列数据
        column_data = data[column_name]
        # 
        if plot:
            pass
        else :
            self.plot_data(column_data,column_name)
        return column_data

    def generate_file_name(self):
        import re
        # 检查并转换 start_time_global
        if not isinstance(self.config.start_time_global, (datetime, UTCDateTime)):
            # 假设 self.config.start_time_global 是一个字符串
            self.config.start_time_global = UTCDateTime(self.config.start_time_global)

        # 检查并转换 end_time_global
        if not isinstance(self.config.end_time_global, (datetime, UTCDateTime)):
            # 假设 self.config.end_time_global 是一个字符串
            self.config.end_time_global = UTCDateTime(self.config.end_time_global)
            
        # 提取时间中的年月日和时分秒
        start_time = self.config.start_time_global.strftime("%Y%m%d_%H%M%S")
        end_time = self.config.end_time_global.strftime("%Y%m%d_%H%M%S")
        
        # 构建文件名
        self.file_name = (f"{self.config.network}.{self.config.station}.{self.config.channel}."
                          f"{start_time}-{end_time}")
        
    def save_data(self):
        self.generate_file_name()
        # 创建文件对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏Tk窗口
        # folder_path = filedialog.askdirectory(title="选择保存文件夹")
        
        # 如果用户选择了文件夹
        default_file_name = f"{self.file_name}"
        
        file_path = filedialog.asksaveasfilename(initialfile=default_file_name,defaultextension=".txt",
                                                 title="保存文件",filetypes=[("Text files", "*.txt"),
                                                                            ("Python files", "*.py"),
                                                                            ("Sac files", "*.sac"),
                                                                            ("Csv files", "*.csv"),
                                                                            ("Ascii files", "*.ascii"),
                                                                            ("All files", "*.*")
                                                                        ])
        if file_path:
            # 提取文件类型和文件扩展名
            file_type = os.path.splitext(file_path)[1][1:].upper()  # 文件类型
            file_extension = os.path.splitext(file_path)[1][1:]  # 文件扩展名
            # 打印文件类型和文件扩展名
            print("文件类型:", file_type)
            print("文件扩展名:", file_extension)

        if file_type == "TXT":
            self.save_data_to_TXT(file_path)
        elif file_type == "SAC":
            self.save_data_to_SAC(file_path)
        elif file_type == "ASCII":
            self.save_data_to_ASCII(file_path)



    def save_data_to_TXT(self,file_path):
        
        file_name, _  = os.path.splitext(os.path.basename(file_path))
        
        for i, trace in enumerate(self.stream):
            file_name = f"{file_name}_{i + 1}.txt"
            output_file_path = os.path.join(os.path.dirname(file_path),file_name)             #os.path.join(self.config.save_folder,'TXT', file_name )
            with open(output_file_path, "w") as file:
                file.write(str(trace))
                file.write("\n--------\n")
                file.write("Stats Information:\n")
                for key, value in trace.stats.__dict__.items():
                    file.write(f"{key}: {value}\n")
                file.write("--------\n")
                file.write("Trace Data:\n")
                for data_point in trace.data:
                    file.write(f"{data_point}, ")
        print("")
        print(f"File <{self.file_name}.txt> has been saved to {os.path.dirname(output_file_path)} !\n")
        return file_name

    def save_data_to_SAC(self,file_path):

        file_name, _  = os.path.splitext(os.path.basename(file_path))

        for i, trace in enumerate(self.stream):
            file_name = f"{file_name}_{i + 1}.sac"
            # Constructing the full file path
            output_file_path = os.path.join(os.path.dirname(file_path), file_name )

            # Check if the file already exists
            if not os.path.exists(output_file_path):
                # Fill masked array with a specified value
                trace.data = np.ma.filled(trace.data, fill_value=np.nan)
                # Writing the trace data to the file in SAC format
                trace.write(output_file_path, format="SAC")
                print(f"File '{file_name}' saved.")
            else:
                print(f"File '{file_name}' already exists. Skipping...")
        print("")
        print(f"The series of File <{file_name}> has been saved to {os.path.dirname(output_file_path)} !\n")
        return file_name
    
    def save_data_to_ASCII(self,file_path):
        file_name, _  = os.path.splitext(os.path.basename(file_path))
        
        for i, trace in enumerate(self.stream):
            file_name = f"{file_name}_{i + 1}.ascii"
            output_file_path = os.path.join(os.path.dirname(file_path),file_name)             #os.path.join(self.config.save_folder,'TXT', file_name )
            with open(output_file_path, "w") as file:
                time=0; delta =trace.stats.delta
                print("delta:",trace.stats.delta)
                for i, data_point in enumerate(trace.data):
                    file.write(f"{round(time+delta*i,2)} {data_point}\n ")
            print(f"File '{file_name}' saved.")
        print("")
        print(f"The series of File <{file_name}> has been saved to {os.path.dirname(output_file_path)} !\n")
        return file_name
    
    def plot_data(self, data, name = None):
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(data, color='blue', linestyle='-')
        plt.title('Plot of {}'.format(name))
        plt.xlabel('Index')
        plt.ylabel(name)
        plt.grid(True)
        plt.show()


    def print_traces(self):
        continue_printing = len(self.stream) >= 10  # 初始化布尔变量，检查 traces 数量是否超过 10
        while continue_printing:
            print("Input 1 to print all Traces, or input 0 to pass this process!\n")
            #A = input("Enter the value of A (0 or 1):\n ")
            A = self.get_user_input("Enter the value of A (0 or 1)")

            if A == "1":
                print(self.stream.__str__(extended=True))
                break
            elif A == "0":
                break
            else:
                print("Invalid input. A should be either 0 or 1.")

    def plot_selected_traces(self):
        #trace_input = input("Enter the index(es) of the trace(s) to plot (e.g., '0' or '0,1,2'): ")
        trace_input = input("Enter the index(es) of the trace(s) to plot (e.g., '0' or '0,1,2'):\n INPUT: ")
        try:
            trace_indices = list(map(int, trace_input.split(',')))
            for index in trace_indices:
                tr = self.stream[index]
                tr.plot()
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid trace index or indices.")

    def plot_seismic_data(self, start_time = None, end_time = None):
        if start_time is None:
            start_time = input("Please input the start_time of the seismogram:\n INPUT:")
        if end_time is None:
            end_time = input("Please input the end_time of the seismogram:\n INPUT:")

        for tr in self.stream:
            tr.plot(starttime=UTCDateTime(start_time), endtime=UTCDateTime(end_time))
        plt.show()
    
   
    def raw_seismogram(self):
        """
        # View a raw seismogram.
        This function retrieves seismic waveform data from a specified network, station, channel,
        and time range. It checks for cached data using a predefined cache key and attempts to
        fetch the data from the cache. If the data is not found in the cache, it downloads the
        waveform data from the IRIS (Incorporated Research Institutions for Seismology) web service.
        
        Returns:
            obspy.core.stream.Stream:
                A stream object containing seismic waveform data. The function also plots the
                waveform data using Matplotlib.
        Raises:
            Exception:
                If there is an error accessing the cache or downloading data from the IRIS service.
        Example:
            stream_from_raw_seismogram = raw_seismogram()
            # Capture the returned stream and optionally use it for further processing.
        """

        if not isinstance(self.config.start_time_global, UTCDateTime):
            self.config.start_time_global = UTCDateTime(self.config.start_time_global)
        if not isinstance(self.config.end_time_global, UTCDateTime):
            self.config.end_time_global = UTCDateTime(self.config.end_time_global)

        print(self.config)
        # 构建缓存键值
        cache_key = (f"{self.config.network}.{self.config.station}.{self.config.channel}.{self.config.location}."
                     f"{self.config.start_time_global}.{self.config.end_time_global}" )

        try:
            # 尝试从缓存中读取数据
            with shelve.open(self.config.cache_stream) as db:
                self.stream = db.get(cache_key)

                if self.stream is None:
                    # 如果缓存中不存在，从服务器下载数据
                    client = Client("IRIS")
                    print(client)

                    self.stream = client.get_waveforms(network=self.config.network, station=self.config.station, 
                                                  channel=self.config.channel, location=self.config.location, 
                                                  starttime=self.config.start_time_global, endtime=self.config.end_time_global)
                    # 将数据写入缓存
                    db[cache_key] = self.stream

        except Exception as e:
            print(f"Error accessing cache: {e}")
            # 如果出现错误，仍然从服务器下载数据
            client = Client("IRIS")
            self.stream = client.get_waveforms(network=self.config.network, station=self.config.station, 
                                            channel=self.config.channel, location=self.config.location, 
                                            starttime=self.config.start_time_global, endtime=self.config.end_time_global)       
        print(self.stream)

        # Save raw stream in cache_streams_list
        self.config.CSL_info = "raw_stream"
        self.config.CSL_stream = self.stream.copy()

        self.stream.plot(equal_scale=False,size=(1000,600),method='full')
    
        return self.stream

    def interpolation(self,trace, method='linear',interpolation_limit=None):
        """Snippet to interpolate missing data.  

        The SHZ traces have missing data samples 3-4 times every 32 samples. 
        Providing the seed data with these missing data would mean using very 
        large files. Instead, we provide the data with -1 replacing the gaps. 
        To change the files to interpolate across the gaps, use this simple method to 
        replace the -1 values. The trace is modified, and a mask is applied at 
        the end if necessary. 

        :type stream: :class:`~obspy.core.Trace` 
        :param trace: A data trace
        :type interpolation_limit: int 
        :param interpolation_limit: Limit for interpolation. Defaults to 1. For
          more information read the options for the `~pandas.Series.interpolate`
          method. 

        :return: original_mask :class:`~numpy.ndarray` or class:`~numpy.bool_`
           Returns the original mask, before any interpolation is made. 

        """
        
        # print("interpolation_limit : ",interpolation_limit)
        trace.data = np.ma.masked_where(trace.data == -1, trace.data)
        original_mask = np.ma.getmask(trace.data)
        data_series = pd.Series(trace.data)
        # data_series.replace(-1.0, pd.NA, inplace=True)
        data_series.interpolate(
            method=method,
            order=3,  # 指定样条的阶数
            axis=0,
            limit=interpolation_limit,
            inplace=True,
            limit_direction=None,
            limit_area='inside')

        data_series.fillna(-1.0, inplace=True)
        trace.data=data_series.to_numpy(dtype=int)
        trace.data = np.ma.masked_where(trace.data == -1, trace.data)
        return original_mask

    def move_response(self,plot_seismogram=True, plot_response=False):
        # Snippet to read in raw seismogram and remove the instrument response for Apollo.

        client = Client("IRIS")

        cache_file_path = os.path.join(self.config.cache_stations, "station_cache.db")

        # 构建缓存键值
        cache_key = (f"{self.config.network}.{self.config.station}.{self.config.channel}.{self.config.location}."
                     f"{self.config.start_time_global}.{self.config.end_time_global}")
        try:
            # 尝试从缓存中读取数据
            with shelve.open(cache_file_path) as db:
                inv = db.get(cache_key)

                if inv is None:
                    # 如果缓存中不存在，从服务器下载数据
                    print("Find no cache, data downlading......")
                    client = Client("IRIS")        
                    inv = client.get_stations(starttime=self.config.start_time_global, endtime=self.config.end_time_global,
                                       network=self.config.network, sta=self.config.station,
                                       loc=self.config.location, channel=self.config.channel,
                                       level="response")

                    # 将数据写入缓存
                    db[cache_key] = inv

        except Exception as e:
            print(f"Error accessing cache: {e}")
            # 如果出现错误，仍然从服务器下载数据
            client = Client("IRIS")
            inv = client.get_stations(starttime=self.config.start_time_global, endtime=self.config.end_time_global,
                                       network=self.config.network, sta=self.config.station,
                                       loc=self.config.location, channel=self.config.channel,
                                       level="response")
        if self.stream is None:
            # If stream is not provided, download it
            self.stream = client.get_waveforms(network=self.config.network, station=self.config.station, 
                                               channel=self.config.channel, location=self.config.location, 
                                               starttime=self.config.start_time_global, endtime=self.config.end_time_global)
        else:
            self.stream.trim(starttime=self.config.start_time_global,endtime=self.config.end_time_global)
        
        for tr in self.stream:
            #interpolate across the gaps of one sample 
            self.interpolation(tr,method="linear",interpolation_limit=1)
        self.stream.merge()
    
        for tr in self.stream:
            # optionally interpolate across any gap 
            # for removing the instrument response from a seimogram, 
            # it is useful to get a mask, then interpolate across the gaps, 
            # then mask the trace again. 
            if tr.stats.channel in ['MH1', 'MH2', 'MHZ']:

                # add linear interpolation but keep the original mask
                original_mask = self.interpolation(tr,interpolation_limit=None)

                # remove the instrument response
                tr.remove_response(inventory=inv, pre_filt=self.config.pre_filt, output="DISP",
                           water_level=None, plot=plot_response)   # output="DISP":displacement;output="VEL":velocity;output="ACC":acceleration.
                if plot_response:
                    plt.show()

                # apply the mask back to the trace 
                tr.data = np.ma.masked_array(tr, mask=original_mask)

            elif tr.stats.channel in ['SHZ']:

                # add linear interpolation but keep the original mask
                original_mask = self.interpolation(tr,interpolation_limit=None)
                # remove the instrument response
                self.config.pre_filt = [1,2,11,13] 
                tr.remove_response(inventory=inv, pre_filt=self.config.pre_filt, output="DISP",
                           water_level=None, plot=plot_response)
                if plot_response:
                    plt.show()
            
                # apply the mask back to the trace 
                tr.data = np.ma.masked_array(tr, mask=original_mask)

        if plot_seismogram:
            #for trace in stream:
            #trace.data = np.ma.filled(trace.data, fill_value=np.nan)
            SeismicProcessor.plot_stream_with_slider(self.stream)
        self.config.CSL_info = "move response"
        self.config.CSL_stream = self.stream.copy()
 
        return self.stream

    def seismic_data_correlation(self,stream_list):
        # seismic data correlation
        # 检查输入是否包含两个 Stream 对象
        if len(stream_list) != 2 or not all(isinstance(st, Stream) for st in stream_list):
            raise ValueError("Input should contain two Stream objects.")

        # 提取两个波形
        waveform1 = stream_list[0][0].data
        waveform2 = stream_list[1][0].data

        ## 对波形进行余弦锥形窗口处理
        #waveform1 *= cosine_sac_taper(len(waveform1), taper_fraction=0.1)
        #waveform2 *= cosine_sac_taper(len(waveform2), taper_fraction=0.1)

        # 对波形应用Hanning窗口
        hanning_window = np.hanning(len(waveform1))
        waveform1 = waveform1.astype('float64')  # 转换为浮点数类型
        waveform1 *= hanning_window

        # 同样的操作应用到 waveform2
        waveform2 = waveform2.astype('float64')
        waveform2 *= hanning_window

        # 计算互相关
        correlation_result = np.correlate(waveform1, waveform2, mode='full')

        # 绘制地震图
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.plot(stream_list[0][0].times(), waveform1, label='Seismic Wave 1',color="black")
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 2, 2)
        plt.plot(stream_list[1][0].times(), waveform2, label='Seismic Wave 2',color="blue")
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # 绘制互相关图
        plt.subplot(2, 1, 2)
        plt.plot(correlation_result,color="red")
        plt.title('Cross-Correlation')
        plt.xlabel('Shift (samples)')
        plt.ylabel('Correlation')

        # 显示图形
        plt.tight_layout()
        plt.show()

        # 返回互相关数据
        return correlation_result

    @staticmethod
    def convert_time_format(input_string):

        if input_string =="0":
            return " ","-> Please Input Date in the diag "
        year, day_of_year, start_time, stop_time = map(int, input_string.split())
        if year < 100:
            year += 1900
        # Convert start_time and stop_time to hours and minutes
        start_hours, start_minutes = divmod(start_time, 100)
        stop_hours, stop_minutes = divmod(stop_time, 100)

        # Calculate start_datetime
        start_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=start_hours, minutes=start_minutes)

        # If stop_time exceeds one day, add the extra days
        stop_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=stop_hours, minutes=stop_minutes)
        if stop_time <= start_time:
            stop_datetime += timedelta(days=1)

        # Format as strings
        start_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        stop_str = stop_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        return start_str, stop_str



    @staticmethod
    def read_sac_file(file_path):
        """
        Read SAC file and return data.

        Parameters:
        - file_path (str): Path to the SAC file.

        Returns:
        - data (numpy.ndarray): NumPy array containing the data.
        """
        # Read SAC file using ObsPy
        st = read(file_path)

        st.plot()
        plt.show()  

        # Check if only one trace is present
        if len(st) != 1:
            raise ValueError("SAC file should contain only one trace.")

        # Extract data from the trace
        data = st[0].data

        return data


    @staticmethod
    def get_fft_values(y_values, N, f_s, fmax=None):
        f_values = np.linspace(0.0, f_s/2.0, N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    
        if fmax is not None:
            # Adjust the values based on the specified maximum frequency
            idx = np.where(f_values <= fmax)[0]
            f_values = f_values[idx]
            fft_values = fft_values[idx]

        return f_values, fft_values

    @staticmethod
    def plot_amplitude_spectrum(stream, fmax):
        # Get amplitude spectrum
        for trace in stream :
            f_s = trace.stats.sampling_rate
            t_values = trace.times()
            y_values = trace.data
            N = len(y_values)
            frequencies, amplitudes = SeismicProcessor.get_fft_values(y_values, N, f_s, fmax)

        # Plot amplitude spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(frequencies, amplitudes,color="red")
        # Set line width
        line.set_linewidth(1)  # 你可以调整这里的值来改变线的宽度

        ax.set_title('Amplitude Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.set_xlim(0, fmax)

        # Add slider for frequency range
        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, 'Frequency Max', 0.0001, fmax*3.0, valinit=fmax)


        def update(val):
            # Update plot based on slider value
            fmax = slider.val
            frequencies, amplitudes = SeismicProcessor.get_fft_values(y_values, N, f_s, fmax)
            line.set_xdata(frequencies)
            line.set_ydata(amplitudes)
            line.set_color('red')
            ax.set_xlim(0, fmax)
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
    


    @staticmethod
    def plot_amplitude_spectrum_s(streams, fmax):
        # Get amplitude spectrum
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, stream in enumerate(streams):
            for trace in stream:
                f_s = trace.stats.sampling_rate
                t_values = trace.times()
                y_values = trace.data
                N = len(y_values)
                frequencies, amplitudes = SeismicProcessor.get_fft_values(y_values, N, f_s, fmax)

                # Plot amplitude spectrum for each trace
                line, = ax.plot(frequencies, amplitudes, label=f'Trace {i + 1}-{trace.id}')

        # Set common properties for the plot
        ax.set_title('Amplitude Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.set_xlim(0, fmax)
        ax.legend()  # 添加图例，显示每个 trace 的标签

        # Add slider for frequency range
        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, 'Frequency Max', 0.0001, fmax * 3.0, valinit=fmax)

        def update(val, f_s=f_s):  # 将f_s作为参数传递给update函数
            # Update plot based on slider value
            fmax = slider.val
            for i, stream in enumerate(streams):
                for trace in stream:
                    y_values = trace.data
                    N = len(y_values)
                    frequencies, amplitudes = SeismicProcessor.get_fft_values(y_values, N, f_s, fmax)
                    # Find the corresponding line by label and update its data
                    line = next(line for line in ax.lines if line.get_label() == f'Trace {i + 1}-{trace.id}')
                    line.set_xdata(frequencies)
                    line.set_ydata(amplitudes)
            ax.set_xlim(0, fmax)
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()


    @staticmethod
    def plot_stream_with_slider(stream):
        """
        Plot ObsPy Stream data with interactive horizontal axis update using sliders.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        """
        # Extract time and amplitude values from the first trace in the stream
        trace = stream[0]
        t_values = trace.times()
        y_values = trace.data

        # 设置全局字体
        mpl.rcParams['font.family'] = 'Arial'#'Times New Roman'

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set color (e.g., 'blue') and linewidth (e.g., 1.5)
        line, = ax.plot(t_values, y_values, color='black', linewidth=1.0)

        # Get the start and end times of the trace
        start_time = UTCDateTime(trace.stats.starttime)
        end_time = UTCDateTime(trace.stats.endtime)

        # Format the start and end times
        formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

        # Set the title with start_time - end_time format
        ax.set_title(f'Interactive Vertical Axis Update\n{formatted_start_time} - {formatted_end_time}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

        # Add annotations for network, station, and channel
        ax.annotate(f"Network: {trace.stats.network}\nStation: {trace.stats.station}\nChannel: {trace.stats.channel}",
                    xy=(0.02, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                    fontsize=10, color='black')

        # Initial y-axis limits
        initial_ylim = ax.get_ylim()

        # Create slider axes for y-axis limits
        dif = initial_ylim[1] - initial_ylim[0]
        ax_slider_ymax = plt.axes([0.2, 0.03, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider_ymax = Slider(ax_slider_ymax, 'Y-Axis Max', initial_ylim[1] - 5 * dif, initial_ylim[1] + 5 * dif,
                             valinit=initial_ylim[1])

        ax_slider_ymin = plt.axes([0.2, 0.005, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider_ymin = Slider(ax_slider_ymin, 'Y-Axis Min', initial_ylim[0] - 5 * dif, initial_ylim[0] + 5 * dif,
                             valinit=initial_ylim[0])

        # Update function for y-axis sliders
        def update_ymin(val):
            new_ylim = (slider_ymin.val, slider_ymax.val)
            ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()

        def update_ymax(val):
            new_ylim = (slider_ymin.val, slider_ymax.val)
            ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()

        # Connect the y-axis sliders to the update functions
        slider_ymin.on_changed(update_ymin)
        slider_ymax.on_changed(update_ymax)

        plt.show()

        return None

    def windowing(self):
        
        pass
    
    def de_pulse(self):
        # de_pulse
        processed_stream = self.stream.copy()  # 保存原始数据的副本 
        print(("Choose methods to clean pluse or outliers (e.g., '1', '1 3'), "
                                   "or enter 'OK' to exit:\n"
                                   "1. Threshold Truncation\n"
                                   "2. MAD-based Cleaning\n"
                                   "3. Moving Average Cleaning\n"
                                   "4. Z-Score Cleaning\n"
                                   "Enter the corresponding numbers separated by space : "))
        while True:
            
            method_choices = self.get_user_input("Choose methods to clean pluse or outliers (e.g., '1', '1 3')","1")

            if method_choices == None:
                break  # 退出循环

            if (method_choices is not None) and (not method_choices.replace(' ', '').isdigit() or any(int(method_choice) not in {1, 2, 3, 4} for method_choice in method_choices.split())):
                print("Invalid choice. Please enter valid numbers between 1 and 4.")
                continue  # 继续下一次循环
            
            if method_choices is not None:
                method_choices_copy = method_choices
                method_choices = [int(choice) for choice in method_choices.split()]
        
                for method_choice in method_choices:
                    if method_choice == 1:
                        amplitude_threshold = str(self.get_user_input("Please input amplitude_threshold: ",""))
                        processed_stream = SeismicProcessor.threshold_truncation(processed_stream, amplitude_threshold=amplitude_threshold)
                        self.config.CSL_info = "threshold_truncatio"
                        self.config.CSL_stream = processed_stream
                        self.plot_stream_with_slider(processed_stream)
                        plt.pause(0.1)
                    elif method_choice == 2:
                        mad_threshold = float(self.get_user_input("Please input MAD threshold(Default: mad_threshold = 3) ","3")) 
                        processed_stream = SeismicProcessor.mad_based_cleaning(processed_stream, mad_threshold=mad_threshold)
                        self.config.CSL_info = "mad_based_cleaning"
                        self.config.CSL_stream = processed_stream
                        self.plot_stream_with_slider(processed_stream)
                        plt.pause(0.1)
                    elif method_choice == 3:
                        window_size = int(self.get_user_input("Please input moving average window size\n (Default :window_size=5): ","5"))
                        amplitude_threshold = float(self.get_user_input("Please input amplitude threshold\n (Default :amplitude_threshold=3): ","3"))
                        processed_stream = SeismicProcessor.moving_average_cleaning(processed_stream, window_size=window_size, amplitude_threshold=amplitude_threshold)
                        self.config.CSL_info = "moving_average_cleaning"
                        self.config.CSL_stream = processed_stream
                        self.plot_stream_with_slider(processed_stream)
                        plt.pause(0.1)
                    elif method_choice == 4:
                        z_threshold = float(self.get_user_input("Please input Z-score threshold\n (Default: z_threshold = 3) : ","3"))
                        processed_stream = SeismicProcessor.z_score_cleaning(processed_stream, z_threshold=z_threshold)
                        self.config.CSL_info = "z_score_cleaning"
                        self.config.CSL_stream = processed_stream
                        self.plot_stream_with_slider(processed_stream)
                        plt.pause(0.1)
                self.config.CSL_info = f"de_pulse_( {method_choices_copy} )"
                self.config.CSL_stream = processed_stream
                plt.show()
    
            self.stream = processed_stream
        # 绘制当前 stream
        #stream.plot(equal_scale=False, size=(1000, 600), method='full')
        #self.config.CSL_info = method_choices_copy
        #self.config.CSL_stream = processed_stream
        #print(self.config.CSL_info, self.config.CSL_stream)

        self.plot_stream_with_slider(self.stream)

        return self.stream

    @staticmethod
    def threshold_truncation(stream, amplitude_threshold):
        """
        Function to truncate data by setting values above a threshold to the threshold.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - amplitude_threshold: float, threshold value for truncation

        Returns:
        - Stream: new Stream object with truncated data
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = trace.data
            if isinstance(amplitude_threshold, (int, float)):
                # 输入是数字时，截断大于等于阈值和小于等于负阈值的数据
                threshold = float(amplitude_threshold)
                mask = (data >= -threshold) & (data <= threshold)
                cleaned_data = np.where(mask, data, -1)
            elif amplitude_threshold.startswith('>'):
                # 输入以">"开头时，截断大于阈值的数据
                threshold = float(amplitude_threshold[1:])
                mask = data <= threshold
                cleaned_data = np.where(mask, data, -1)
            elif amplitude_threshold.startswith('<'):
                # 输入以"<"开头时，截断小于阈值的数据
                threshold = float(amplitude_threshold[1:])
                mask = data >= threshold
                cleaned_data = np.where(mask, data, -1)
            else:
                # 无法识别的输入，不进行截断
                cleaned_data = data

            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def mad_based_cleaning(stream, mad_threshold=3):
        """
        Function to remove outliers based on Median Absolute Deviation (MAD).

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - mad_threshold: float, Multiples of the absolute deviation of the median

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = trace.data
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            cleaned_data = np.where(np.abs(data - median) > mad_threshold * mad, median, data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def moving_average_cleaning(stream, window_size=700, amplitude_threshold=5):
        """
        Function to remove outliers based on a moving average.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - window_size: int, size of the moving average window
        - amplitude_threshold: float, threshold for identifying outliers based on deviation from moving average

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = trace.data
            moving_average = np.convolve(data, np.ones(window_size)/window_size, mode='same')
            cleaned_data = np.where(np.abs(data - moving_average) > amplitude_threshold, moving_average, data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def z_score_cleaning(stream, z_threshold=3):
        """
        Function to remove outliers based on Z-scores.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - z_threshold: float, threshold for identifying outliers based on Z-scores

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = trace.data
            z_scores = (data - np.mean(data)) / np.std(data)
            cleaned_data = np.where(np.abs(z_scores) > z_threshold, np.mean(data), data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream



    def get_user_input(self, prompt, initial_value=""):
        # 弹出输入框，并给定初始值
        top = tk.Toplevel()
        top.withdraw()  # 隐藏默认窗口
        top.title("User Input")
        top.grab_set()  # 阻止与其他窗口交互
        user_input = simpledialog.askstring("Input", prompt, initialvalue=initial_value)
        print("User INPUT:", user_input,"\n")
        top.destroy()  # 关闭子窗口
        return user_input

    def get_user_inputs(self,num_fields, prompts, initial_values):
        root = tk.Tk()
        root.withdraw()  # 隐藏默认窗口
        dialog = MultiInputDialog(root, "Input Your Data", num_fields, prompts, initial_values)
        result = dialog.result
        ## 进一步处理 result，将其转换回原始的格式
        #processed_result = []
        #for param_name, value in zip(param_names, result):
        #    if param_name == 'cache_streams_list':
        #        processed_result.append(eval(value))
        #    else:
        #        processed_result.append(value)

        print("User INPUT:", result,"\n")
        root.destroy()  # 关闭主窗口
        return result
    
    def seismic_stream2data(self):
        if self.stream:
            # 获取每个 trace.data 的样本点数量 m
            max_samples = max(len(trace.data) for trace in self.stream)
            m = max_samples

            # 获取 trace 的数量 n
            n = len(self.stream)

            # 初始化一个 mxn 的矩阵
            data = np.zeros((m, n))

            info = []
            # 将每个 trace.data 的样本点按列存储到 data 中
            for idx, trace in enumerate(self.stream):
                trace_info = [
                    trace.stats.network,
                    trace.stats.station,
                    trace.stats.channel,
                    trace.stats.starttime,
                    trace.stats.endtime
                ]
                info.append(trace_info)
                # 如果 trace.data 的长度小于 m，则在末尾填充0
                if len(trace.data) < m:
                    data[:len(trace.data), idx] = trace.data
                    data[len(trace.data):, idx] = None
                # 如果 trace.data 的长度大于 m，则截断到长度为 m
                else:
                    data[:, idx] = trace.data[:m]

            data = np.array(data)
            return data, info
        else:
            print("Error: stream is empty.")
            return None, None
        
    def seismic_data_sparsification(self):
        #地震数据稀疏化
        data,_= self.seismic_stream2data()



        pass

    def synthetic_seismic_data(self,num_samples, num_time_points, layer_thicknesses, peak_frequency, noise_level):
        """
        生成合成地震数据。

        Parameters:
        - num_samples (int): 生成的地震数据样本数量。
        - num_time_points (int): 每个地震数据样本的时间点数量。
        - layer_thicknesses (list): 地层厚度列表，每个元素表示一个地层的厚度。
        - peak_frequency (float): 雷克子波的峰值频率。
        - noise_level (float): 添加到地震数据中的高斯噪声的标准差。

        Returns:
        - seismic_data (ndarray): 合成的地震数据，形状为(num_samples, num_time_points)的二维数组。

        该函数首先生成雷克子波作为地震波形的基本成分，然后根据地层厚度和反射系数合成地震数据。每个地震数据样本
        都以随机到达时间开始，然后在对应的到达时间点添加地层反射系数乘以雷克子波。最后，为每个地震记录添加不同
        的高斯噪声。

        """
        # 生成雷克子波
        t = np.linspace(-1, 1, 200)
        ricker_wavelet = (1 - 2 * (np.pi ** 2) * (peak_frequency ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (peak_frequency ** 2) * (t ** 2))

        # 生成地层反射系数序列
        layer_sequence = np.random.randint(0.5, 1, size=len(layer_thicknesses))

        # 生成地震到达时间
        arrival_time = np.random.randint(200, num_time_points - len(ricker_wavelet) + 1)  # 调整到达时间范围
        
        # 合成地震数据
        seismic_data = np.zeros((num_samples, num_time_points))
        for i in range(num_samples):
            for j, thickness in enumerate(layer_thicknesses):
                amplitude = 1#np.random.uniform(0.5, 1.0)
                seismic_data[i, arrival_time:arrival_time+len(ricker_wavelet)] += amplitude * layer_sequence[j] * ricker_wavelet
            
            # 为每个地震记录生成不同的高斯噪声
            seismic_data[i] += np.random.normal(scale=noise_level, size=num_time_points)

        return seismic_data

    def plot_seismic_waveforms(self,data=None,info=None):
        from matplotlib.ticker import MultipleLocator
        """
        **Plot seismic waveforms.**

        Args:
        data: numpy array, seismic data with shape (num_samples, num_stations).
            Each column represents data from a station.
        info (list): Information about the seismic data, such as network, station, channel,
            starttime, and endtime.

        Returns:
        None
        """            
        if self.stream:
            data, info = self.seismic_stream2data()
        elif np.any(data):
            if info == None:
                info = []
                for i in range(data.shape[1]):  # 遍历data的列数
                    column_info = []  # 创建一个空列表，用于存储当前列的值
                    for _ in range(5):  # 在当前列中添加5个0
                        column_info.append(0)
                    info.append(column_info)  # 将当前列的值添加到info中
            else:
                pass
        else:
            print("Error: self.stream is empty.")
            sys.exit()
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError("Input data must be 1-dimensional or 2-dimensional")

        
        # 检查数据中是否存在 nan 和 inf 值
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if has_nan:
            data = SeismicProcessor.replace_nan_with_value(data, value_type='mean')
        if has_inf:
            data = SeismicProcessor.replace_inf_with_value(data, value_type='max')
        
        data1 = data
        
        #归一化处理数据
        data = SeismicProcessor.normalize(data)
       
        
        num_samples, num_stations = data.shape
        # print("num_stations",num_stations)
        d_sta = round((np.max(data)-np.min(data))/2,2)
        # print("d_sta:",d_sta)

        ax = plt.figure(figsize=(16, 6))
        # 获取当前图形的大小（以英寸为单位）
        fig_size = plt.gcf().get_size_inches()

        # 设置字体大小与y轴长度的比例
        font_scale = 2
        # 计算新的字体大小
        new_font_size = font_scale * (fig_size[1]/num_stations)
        if new_font_size < 8:
            new_font_size = 8
        if new_font_size > 15:
            new_font_size = 15
        for sta in range(num_stations):
            # print("sta",sta,num_stations)
            plt.plot(data[:, sta] + sta,color='black',linewidth=0.4,  label=f"Sta {sta+1}")
            plt.text(0,sta+0.90, f"{info[sta][0]}.{info[sta][1]}.{info[sta][2]} | {info[sta][3]}-{info[sta][4]}", 
                     fontsize=new_font_size, color='blue')
        plt.title(f"Data Waveform\n{info[0][0]}")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Amplitude")
        # plt.legend(loc='upper left')
        

        # 设置横向和纵向刻度
        num_ticks = 5
        x_tick_positions = np.linspace(0, num_samples, num_ticks)
        # y_tick_positions = np.linspace(0 + d_sta, num_stations - d_sta, num_stations)  # 以 d_sta 为偏移量

        # 绘制刻度
        plt.xticks(x_tick_positions)
        plt.ylim(ymin=0,ymax=num_stations)
        y_major_locator = MultipleLocator(1)
        y_minor_locator = MultipleLocator(d_sta)
        plt.gca().yaxis.set_major_locator(y_major_locator)
        plt.gca().yaxis.set_minor_locator(y_minor_locator)

        # 设置 y 次轴刻度标签
        y_tick_labels = [f"Sta-{i}  " for i in range(num_stations+1)]
        plt.gca().set_yticklabels(y_tick_labels, minor=True, rotation=15, color='red', fontsize = 10 )

        # 绘制网格线
        plt.grid(True, which='minor', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=-1)

        plt.show()

        return data1
    
    @staticmethod
    def replace_nan_with_value(data, value_type='mean'):
        """
        将数组中的 nan 值替换为特定的值
        
        参数：
            data: numpy 数组，需要处理的数据
            value_type: 字符串，指定替换 nan 值的方式，可选值为 'mean' 或 'median'，默认为 'mean'
            
        返回值：
            replaced_data: 处理后的数组
        """
        # 计算指定方式下的替换值
        if value_type == 'mean':
            replace_value = np.nanmean(data)
        elif value_type == 'median':
            replace_value = np.nanmedian(data)
        else:
            raise ValueError("value_type 参数必须是 'mean' 或 'median'")
        
        # 将 nan 值替换为计算得到的替换值
        replaced_data = np.where(np.isnan(data), replace_value, data)
        
        return replaced_data
    @staticmethod
    def replace_inf_with_value(data, value_type='max'):
        """
        将数组中的 inf 值替换为特定的值
        
        参数：
            data: numpy 数组，需要处理的数据
            value_type: 字符串，指定替换 inf 值的方式，可选值为 'max' 或 'min'，默认为 'max'
            
        返回值：
            replaced_data: 处理后的数组
        """
        # 计算指定方式下的替换值
        if value_type == 'max':
            replace_value = np.finfo(data.dtype).max  # 获取最大有限值
        elif value_type == 'min':
            replace_value = np.finfo(data.dtype).min  # 获取最小有限值
        else:
            raise ValueError("value_type 参数必须是 'max' 或 'min'")
        
        # 将 inf 值替换为指定的替换值
        replaced_data = np.where(np.isinf(data), replace_value, data)
        
        return replaced_data

    @staticmethod
    def normalize(data, Reference = None):
        """
        ## 将输入的数据集 data 进行归一化处理，参考量为 Reference。

        参数：
        Reference (float/int): 归一化参考量。
        data (list/NDArray): 需要归一化的数据集,一维或二维数据。

        返回值：
            normalized_data: 归一化后的数组
        归一化规则：
        对于 data 中的每个元素 x：
        - 如果 x 等于 Reference，则将其归一化为 1。
        - 如果 x 小于 Reference，则将其按照与最小值的比例进行缩放。
        - 如果 x 大于 Reference，则将其按照与最小值的比例进行放大。

        示例：
        >>> Reference = 12
        >>> data = [3, 5, 7, 9, 5, 6, 2, 10, 8, 12, 15, 4, 11]
        >>> normalize(Reference, data)
        [0.1, 0.3, 0.5, 0.7, 0.3, 0.4, 0.0, 0.8, 0.6, 1, 1.3, 0.2, 0.9]
        """

        # 确保数据类型为 numpy.ndarray
        data = np.array(data)
        
        # 获取数组的形状
        shape = data.shape
        
        # 如果数组是一维的，则计算每个元素的最大值和最小值
        if len(shape) == 1:
            min_val = np.min(data)
            max_val = np.max(data)
            if Reference is not None:
                
                # 参考归一化处理
                normalized_data = []
                for x in data:
                    if x == Reference:
                        normalized_data.append(1)
                    elif x < Reference:
                        normalized_value = (x - min_val) / (Reference - min_val)
                        normalized_data.append(normalized_value)
                    else:
                        normalized_value = 1 + (x - Reference) / (Reference - min_val)
                        normalized_data.append(normalized_value)
                normalized_data = np.array(normalized_data)
            else:
                if max_val == min_val:
                    normalized_data = np.zeros_like(data)
                else:
                    normalized_data = (data - min_val) / (max_val - min_val)
        
        # 如果数组是二维的，则按列计算每列的最大值和最小值
        elif len(shape) == 2:
            
            if Reference is not None:
                # 参考归一化处理
                Reference = np.array(Reference)
                normalized_data = []
                # Normalize each column of data
                normalized_data = np.zeros_like(data, dtype=float)
                for i in range(data.shape[1]):
                    col = data[:, i]
                    min_val = min(col)
                    max_val = max(col)
                    reference_val = Reference[i]

                    # Normalize each element in the column
                    for j, x in enumerate(col):
                        if x == reference_val:
                            normalized_data[j, i] = 1.0
                        elif x < reference_val:
                            normalized_data[j, i] = (x - min_val) / (reference_val - min_val)
                        else:
                            normalized_data[j, i] = (x - reference_val) / (reference_val - min_val) + 1.0
            else:
                min_vals = np.min(data, axis=0)
                max_vals = np.max(data, axis=0)
                zero_mask = max_vals == min_vals
                normalized_data = np.zeros(shape)
                for i in range(len(min_vals)):
                    if not zero_mask[i]:
                        normalized_data[:, i] = (data[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            
        # 如果数组的维度不是 1 或 2，则抛出异常
        else:
            raise ValueError("输入的数组维度必须是 1 或 2")
        
        return normalized_data
    
    @staticmethod
    def z_score_normalize(data):
        """
        计算数据集的 Z-score 标准化。

        Parameters:
            data (numpy.ndarray or list): 输入的数据集。

        Returns:
            numpy.ndarray: 标准化后的数据集。
        """
        # 将数据集转换为 numpy 数组
        data = np.array(data)
        
        # 计算算术平均值
        mean = np.mean(data)
        
        # 计算标准差
        std_dev = np.std(data)
        
        # 对数据集进行 Z-score 标准化
        standardized_data = (data - mean) / std_dev
        
        return standardized_data

class MultiInputDialog(simpledialog.Dialog):
    def __init__(self, parent, title, num_fields, prompts, initial_values):
        self.num_fields = num_fields
        self.prompts = prompts
        self.initial_values = initial_values
        super().__init__(parent, title=title)

    def body(self, master):
        self.entries = []
        for i, (prompt, initial_value) in enumerate(zip(self.prompts, self.initial_values)):
            tk.Label(master, text=prompt).grid(row=i, sticky=tk.W)
            entry = tk.Entry(master, width=50)
            entry.insert(0, str(initial_value))
            entry.grid(row=i, column=1)
            self.entries.append(entry)

        return self.entries[0]  # 返回第一个输入框，作为焦点

    def apply(self):
        self.result = []
        for entry, initial_value in zip(self.entries, self.initial_values):
            # 根据初始值的类型进行转换
            if isinstance(initial_value, int):
                self.result.append(int(entry.get()))
            elif isinstance(initial_value, float):
                self.result.append(float(entry.get()))
            elif isinstance(initial_value, list):
                # 如果初始值是列表，使用 eval 转换
                # 使用 ast.literal_eval 替代 eval
                try:
                    value = ast.literal_eval(entry.get())
                except (SyntaxError, ValueError):
                    value = entry.get()

                self.result.append(value)
                print(value,":",type(value))
            else:
                # 默认情况下保留为字符串
                self.result.append(entry.get())

if __name__ =="__main__":
    SeismicProcessor.main()

