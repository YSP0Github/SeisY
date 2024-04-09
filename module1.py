
def main():
    
    print("设置缓存文件夹:")
    print(f'cache_file : {self.cache_file}')
    print(f'cache_directory : {self.cache_directory}')
    print('# 设置缓存文件夹（SAC数据）:')
    print(f'save_folder = {save_folder}')
    print(f"  network={network}\n  station={station}\n  channel={channel}\n  location={location}")
    print(f"pre_filt = {pre_filt}")

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

    input_string = "0"
    input_string = "  71 192 1327 1327"         # if you input time manually, please comment tihs line.

    start_time_global,end_time_global  =  TimeProcessor.convert_time_format(input_string)
    print(f"Start Time: {start_time_global}");print(f"Stop Time: {end_time_global}")

    ## Set start and end time
    if input_string == "0":
        start_time_global= UTCDateTime('1971-03-13T07:30:00.0')
        end_time_global = UTCDateTime('1971-03-13T09:30:00.0')
        

    
    ## Replace with your seismic data file path
    #data_file_path ="H:\\APOLLO\\pse\\p14s\\pse.a14.1.10"
    
    ## process and plot local data
    #process_and_plot_local_seismic_data(data_file_path)

    # plot the raw seismogram
    stream_from_raw_seismogram = raw_seismogram()  # Capture the returned stream

# Data preprocessing
    # linear_interpolation and move_response
    stream = interpolate_response(stream = stream_from_raw_seismogram,plot_response=True)  

    # move outliers
    move_outlies(stream)

    # plot amplitude spectrum
    f_s = stream[0].stats.sampling_rate
    plot_amplitude_spectrum(stream[0], f_s,pre_filt[3])
   
    ## output SAC file 
    #file_path = save_data_to_SAC(stream)

    ## Read and Plot SAC file
    #read_sac_file(file_path)



    # Read SAC file
    #file_path = "H:/APOLLO/Cache_seismic_datas/XA.S12.MHZ.19720102_223000-19720103_223000.SAC"
    #data = read_sac_file(file_path)
    #print(data)

    #interpolate_responseo(plot_seismogram=False,plot_response=True)

    return None

    @staticmethod
    def plot_seismic_spectral_data(stream, channel='MHZ', window=('tukey', 0.25), nperseg=256, noverlap=128, plot_type='spectrogram'):
        """
        绘制地震数据的频谱图或频谱振幅图。

        参数：
        - stream: 包含地震数据的 ObsPy Stream 对象
        - channel: 要绘制的通道
        - window: 使用的窗口函数（默认：Tukey 窗口，alpha=0.25）
        - nperseg: 每个 FFT 块中的数据点数（默认：256）
        - noverlap: 块之间的重叠数据点数（默认：128）
        - plot_type: 绘图类型，可以是 'spectrogram' 或 'amplitude_spectrum'，默认为 'spectrogram'
        """
        # 提取指定通道的数据
        trace = stream.select(channel=channel)[0]

        # 应用窗口函数
        taper = cosine_taper(len(trace.data), 0.25)
        tapered_data = trace.data * taper
    
        # 计算频谱图
        f, t, Sxx = spectrogram(tapered_data, fs=trace.stats.sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)


        if plot_type == 'spectrogram':
        
            # 绘制频谱图
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
            plt.colorbar(label='功率/频率 (dB/Hz)')
            plt.ylabel('频率 (Hz)')
            plt.xlabel('时间 (秒)')
            plt.title(f'频谱图 - 通道: {channel}')
            plt.show()
    
        elif plot_type == 'amplitude_spectrum':
            ## 计算频谱
            #f, Pxx = plt.psd(tapered_data, NFFT=nperseg, Fs=trace.stats.sampling_rate, noverlap=noverlap, scale_by_freq=True)

            # 绘制频谱振幅曲线图
            plt.figure(figsize=(10, 6))
            plt.semilogy(f, np.abs(Sxx[:, 0]), color='b')
            plt.xlabel('频率 (Hz)')
            plt.ylabel('振幅')
            plt.title(f'频谱振幅图 - 通道: {channel}')
            plt.grid(True)
            plt.show()
    
        else:
            print("无效的绘图类型。")

        return None

#class InputThread(Thread):
#    def __init__(self, result_queue):
#        super().__init__()
#        self.result_queue = result_queue

#    def run(self):
#        # 在这里执行需要等待用户输入的操作
#        print(f"{INPUT: prompt}\n")

#        # 将结果放入队列，以便主线程获取
#        self.result_queue.put(user_input)

#class InputSimulation:
#    def __init__(self, root):
#        self.root = root
#        self.result_queue = queue.Queue()
#        self.init_ui()

#    def init_ui(self):
#        btn_process = tk.Button(self.root, text="Process", command=self.start_input_thread)
#        btn_process.pack()

#    def start_input_thread(self):
#        # 创建并启动线程
#        input_thread = InputThread(self.result_queue)
#        input_thread.start()

#        # 在主线程中等待用户输入结果
#        self.root.after(100, self.check_input_result)

#    def check_input_result(self):
#        try:
#            user_input = self.result_queue.get_nowait()
#            print(f"Received input: {user_input}")
#            # 在这里可以调用其他函数并传递 user_input
#        except queue.Empty:
#            # 如果队列为空，等待一段时间后再次检查
#            self.root.after(100, self.check_input_result)
