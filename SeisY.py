# SeisY
# VERSION = '1.0.0'
# Author: [余松平(Yu Songping) , China University of Geosciences ]
# Contact: [...]
# Description: SeisY is a seismic data processing software designed for analyzing and visualizing seismic data.
# License: [MIT License]
# Created: [2023/12/12]
# Last Updated: [2024/4/8]
# Email :[ysp@cug.edu.cn or york.stephen19@gmail.com]

import PlotSeismic
import matplotlib.pyplot as plt
import sys
import os
import tkinter as tk
import configparser
import webbrowser
import ast
import paramiko
from tkinter import Menu,Canvas,filedialog,ttk, messagebox
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.cross_correlation import correlate
from obspy import  Stream

class SubMenuExample:
    def __init__(self, root):
        #super().__init__(root)
        self.root = root
        self.data_file_path = " "
        # 创建配置类实例对象
        self.seismic_config = PlotSeismic.SeismicConfig()
        # 创建 SeismicProcessor 实例时时传入配置类
        self.seismic_processor = PlotSeismic.SeismicProcessor(config=self.seismic_config,data_file_path=self.data_file_path)
        # 创建一个ttk.Style对象
        self.style = ttk.Style()
        self.init_ui()
        self.check_and_create_folders_files()

    def init_ui(self):
        # Create a menu bar
        self.menubar = Menu(self.root)

        # 设置窗口属性
        self.root.config(menu=self.menubar)

        # 获取显示器宽度和高度
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 设置窗口尺寸
        self.root.geometry(f"{int(0.90 * screen_width)}x{int(0.90 * screen_height)}+0+0")

        # 最大化窗口
        self.root.state('zoomed')  
        self.root.title('SeisY Tool')

        # 设置窗口的最小尺寸
        self.root.minsize(1000, 600)
        
        # 获取当前窗口的尺寸
        self.root.update_idletasks()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()

#------------------------MENU--DESIGH------------------------------------------------------------------------

        # Create a main menu 'File'--------------------------------------------------------------------------
        file_menu = Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Open File', command=self.open_file_dialog)

        # Create some sub-menus for 'Config Files'---------------------------------------
        file_config_menu = Menu(file_menu, tearoff=0)
        file_config_menu.add_command(label="Load Config", command=self.load_config)
        file_config_menu.add_command(label="Update Config", command=self.update_config_file)
        file_menu.add_cascade(label='Config Files', menu=file_config_menu)
        # Create some sub-menus for 'Save Files'-----------------------------------------
        file_save_menu = Menu(file_menu, tearoff=0)
        file_menu.add_command(label='Save Data', command=self.save_data)
        # file_save_menu.add_command(label='Save File to TXT',command = self.save_data_to_txt)
        # file_save_menu.add_command(label='Save File to SAC',command = self.save_data_to_sac)
        # file_menu.add_cascade(label='Save File', menu=file_save_menu)

        # Create a main menu 'Data Processing'---------------------------------------------------------------
        process_menu = Menu(self.menubar, tearoff=0)
        process_menu.add_command(label="Input Date", command=self.input_date)
        process_menu.add_command(label="Get Seismic Data", command=self.raw_seismogram)
        process_menu.add_command(label="Move Response", command=self.move_response)
        # Create some sub-menus for 'Interpolation'--------------------------------------
        process_interpolate_menu = Menu(process_menu,tearoff=0)
        process_interpolate_menu.add_command(label="Linear", command=lambda: self.interpolate("linear"))
        process_interpolate_menu.add_command(label="Nearest", command=lambda: self.interpolate("nearest"))
        process_interpolate_menu.add_command(label="Polynomial", command=lambda: self.interpolate("polynomial"))
        process_interpolate_menu.add_command(label="Spline", command=lambda: self.interpolate("spline"))
        process_interpolate_menu.add_command(label="Barycentric", command=lambda: self.interpolate("barycentric"))
        process_interpolate_menu.add_command(label="Krogh", command=lambda: self.interpolate("krogh"))
        process_interpolate_menu.add_command(label="Piecewise Polynomial", command=lambda: self.interpolate("piecewise_polynomial"))
        process_interpolate_menu.add_command(label="PCHIP", command=lambda: self.interpolate("pchip"))
        process_interpolate_menu.add_command(label="Akima", command=lambda: self.interpolate("akima"))
        process_interpolate_menu.add_command(label="Cubic Spline", command=lambda: self.interpolate("cubicspline"))
        process_menu.add_cascade(label="Interpolation",menu=process_interpolate_menu)

        process_menu.add_command(label="Windowing", command=self.windowing)
        process_menu.add_command(label="De-pulse and Plot", command=self.de_pulse_and_plot)
        process_menu.add_command(label="Plot Amplitude Spectrum", command=self.plot_amplitude_spectrum)
        process_menu.add_command(label="Plot Amplitude Spectrums", command=self.plot_amplitude_spectrum_s)
        process_menu.add_command(label="Seismic Data Correlation", command=self.seismic_data_correlation)
        process_menu.add_command(label="Seismic Data Sparsification", command=self.seismic_data_sparsification)
        process_menu.add_command(label="Plot Current Stream", command=self.plot_current_stream)
        process_menu.add_command(label="Plot Current Stream 2.0", command=self.plot_current_stream_2)

        # Creat a main menu 'Settings'-----------------------------------------------------------------------
        settings_menu = Menu(self.menubar, tearoff=0)
        settings_menu.add_command(label="Modify configurations",command = self.modify_configurations)
        settings_menu.add_command(label="Show Streams List",command = self.show_streams_list)
        settings_menu.add_command(label="Merge Streams List",command = self.merge_streams_list)
        settings_menu.add_command(label="Save CSL ",command = self.save_CSL)

        # Create some sub-menus for 'Clear CSL'------------------------------------------
        settings_CSL_menu = Menu(settings_menu, tearoff=0)
        settings_CSL_menu.add_command(label="Cut Latest One CSL",command=lambda:self.clear_CSL("1"))
        settings_CSL_menu.add_command(label="Cut Latest Two CSL",command=lambda:self.clear_CSL("2"))
        settings_CSL_menu.add_command(label="Cut All CSL",command=lambda:self.clear_CSL("all"))
        settings_CSL_menu.add_command(label="Cut CSL Customized ",command=lambda:self.clear_CSL("Customized"))
        settings_menu.add_cascade(label="Clear CSL ",menu=settings_CSL_menu)

        #Creat a main menu 'SSH Connect'
        SSH_menu = Menu(self.menubar, tearoff=0)
        SSH_menu.add_command(label="SSH Connect",command = self.ssh_execute_command)
        SSH_menu.add_command(label="SSH Command Input ",command = self.input_ssh_command)
        SSH_menu.add_command(label="SSH Close",command = self.close_ssh)

        # Creat a main menu 'Help'---------------------------------------------------------------------------
        help_menu = Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="De-pulse help",command = self.open_de_pulse_help)
        help_menu.add_command(label="Interpolation help",command = self.show_interpolation_help)
        help_menu.add_command(label="About",command = self.about)
        # Creat a main menu 'Theme'--------------------------------------------------------------------------
        theme_menu = Menu(self.menubar, tearoff=0)
        theme_menu.add_command(label="Defalt Mode",command=lambda: self.Setting_Theme("#EEEEEE","black"))
        theme_menu.add_command(label="Light Mode",command=lambda: self.Setting_Theme("White","black"))
        theme_menu.add_command(label="Gray Mode",command=lambda: self.Setting_Theme("#666666","#33FFFF"))
        theme_menu.add_command(label="Dark Mode",command=lambda: self.Setting_Theme("black","white"))
        theme_menu.add_command(label="Forest Mode",command=lambda: self.Setting_Theme("#003300","#D2B48C"))

        # Add menus to the menu bar--------------------------------------------------------------------------
        self.menubar.add_cascade(label='File', menu=file_menu)
        self.menubar.add_cascade(label='Data Processing', menu=process_menu)
        self.menubar.add_cascade(label="Settings", menu=settings_menu)
        self.menubar.add_cascade(label="SSH Connect", menu=SSH_menu)
        self.menubar.add_cascade(label="Help",menu=help_menu)
        self.menubar.add_cascade(label="Theme",menu=theme_menu)

#------------------------MENU--DESIGH------------------------------------------------------------------------


#------------------------POWERSHELL--DESIGH------------------------------------------------------------------

        # Create a Frame to contain the text widget
        frame = tk.Frame(self.root, bg="black", width=int(0.08 * parent_height), height=int(0.06 * parent_height))
        frame.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=5)

        # Get the width of the frame
        frame_width = frame.winfo_reqwidth()

        # Create a Label as the prompt for the text widget
        self.title_Label = tk.Label(frame, width=frame_width+2, height=int(0.001 * parent_height),
                               text=" SeisY  PowerShell ", font=("Consolas", 11), bg="darkgreen", fg="White", anchor="center")
        self.title_Label.pack(side=tk.TOP, anchor=tk.NW)

        # Create a Text widget for input and output
        self.text_widget = tk.Text(frame, width=frame_width,height=int(0.059 * parent_height), wrap=tk.WORD, bg="black", fg="white",
                                    insertbackground="red", font=("Consolas", 11))
        # Create a Scrollbar for Text widget 
        self.scrollbar = tk.Scrollbar(frame, command=self.text_widget.yview)

        # Configure Text to use Scrollbar
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)

        # Pack Text and Scrollbar inside the Frame
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


        # Define a style tag with yellow foreground color
        self.text_widget.tag_configure("input_tag", foreground="yellow")
        self.text_widget.tag_configure("stream_tag", foreground="red")
        self.text_widget.tag_configure("Trace_tag", foreground="#33FFFF")
        self.text_widget.tag_configure("Config_tag", foreground="#00CC33")
        self.text_widget.tag_configure("streamlist_tag", foreground="#00CC33")
        self.text_widget.tag_configure("______tag", foreground="red")
        

        # Listen for changes in the text widget content
        self.text_widget.bind("<Return>", self.process_input)
        self.text_widget.bind("<<Key>>", self.highlight_input)
        self.text_widget.bind("<KeyRelease>", self.highlight_input)
        self.text_widget.bind("<<Modified>>", self.highlight_input)

        # Redirect sys.stdout
        sys.stdout = self        
        return None
#------------------------POWERSHELL--DESIGH------------------------------------------------------------------
#------------------------INITIALIZATION----------------------------------------------------------------------
    def check_and_create_folders_files(self):
        # 获取当前脚本所在的路径
        self.seismic_config.current_file_path = os.path.dirname(os.path.abspath(__file__))

        # 定义需要检查的文件夹和文件
        folders_to_check = [
            'help',
            'images',
            'DATA/TXT',
            'DATA/SAC',
            'cache/Cache_seismic',
            'cache/Cache_stations',
        ]
        file_to_check = 'Config.ini'

        # 检查文件夹
        for folder in folders_to_check:
            folder_path = os.path.join(self.seismic_config.current_file_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder}' created.")

        # 检查 Config.ini 文件
        file_path = os.path.join(self.seismic_config.current_file_path, file_to_check)
        if os.path.exists(file_path):
            # 如果文件存在，读取配置信息
            existing_config = self.read_existing_config(file_path)

            # 检查配置信息是否存在，以及是否符合期望
            if self.check_config(existing_config):
                print(f"Config file '{file_to_check}' already exists and has the expected network value.")
            else:
                print(f"Updating config file '{file_to_check}' with default values.")
                # 更新配置信息
                self.update_config(file_path,self.seismic_config.current_file_path)
        else:
            # 如果文件不存在，创建新的配置文件并给定初始值
            self.create_initial_config(file_path, self.seismic_config.current_file_path)
            print(f"Config file '{file_to_check}' created with initial values.")

    def check_config(self, existing_config):
        # 检查配置信息是否存在，以及是否符合期望
        expected_keys = ['start_time_global', 'end_time_global', 'network', 'station', 'channel', 'location', 
                         'pre_filt', 'cache_stream', 'cache_stations', 'save_folder', 'csl_info', 
                         'csl_stream', 'cache_streams_list', 'current_file_path']

        if 'SeismicConfig' in existing_config:
            seismic_config_section = existing_config['SeismicConfig']
        
            # 检查所有期望的键是否都在配置中
            if all(key in seismic_config_section for key in expected_keys):
                return True
            else:
                return False
        else:
            return False

    def create_initial_config(self,file_path, current_path):
        config = configparser.ConfigParser()
        config['SeismicConfig'] = {
            'start_time_global': '1971-07-11T13:27:00.000',
            'end_time_global': '1971-07-12T13:27:00.000',
            'network': 'XA',
            'station': 'S12',
            'channel': 'MHZ',
            'location': '*',
            'pre_filt': '[0.0005, 0.005, 0.005, 0.03]',
            'cache_stream': os.path.join(current_path, 'cache/Cache_seismic/stream.db'),
            'cache_stations': os.path.join(current_path, 'cache/Cache_stations'),
            'save_folder': os.path.join(current_path, 'DATA'),
            'csl_info': ' ',
            'csl_stream': ' ',
            'cache_streams_list': [],
            'current_file_path ':self.seismic_config.current_file_path,
        }

        with open(file_path, 'w') as config_file:
            config.write(config_file)
        self.config_file_path = file_path
        self.read_config_file()

    def read_existing_config(self, file_path):
        # 读取已存在的配置文件
        config = configparser.ConfigParser()
        config.read(file_path)
        existing_config = {}
        for section in config.sections():
            existing_config[section] = dict(config.items(section))
        return existing_config

    def update_config(self, file_path, current_path):
        # 读取并更新配置文件
        existing_config = self.read_existing_config(file_path)
    
        existing_config['SeismicConfig']['start_time_global'] = '1971-07-11T13:27:00.000'
        existing_config['SeismicConfig']['end_time_global'] = '1971-07-12T13:27:00.000'
        existing_config['SeismicConfig']['network'] = 'XA'
        existing_config['SeismicConfig']['station'] = 'S12'
        existing_config['SeismicConfig']['channel'] = 'MHZ'
        existing_config['SeismicConfig']['location'] = '*'
        existing_config['SeismicConfig']['pre_filt'] = '[0.0005, 0.005, 0.005, 0.03]'
        existing_config['SeismicConfig']['cache_stream'] = os.path.join(current_path, 'cache/Cache_seismic/stream.db')
        existing_config['SeismicConfig']['cache_stations'] = os.path.join(current_path, 'cache/Cache_stations')
        existing_config['SeismicConfig']['save_folder'] = os.path.join(current_path, 'DATA')
        existing_config['SeismicConfig']['csl_info'] = None
        existing_config['SeismicConfig']['csl_stream'] = None
        existing_config['SeismicConfig']['cache_streams_list'] = [ ]
        existing_config['SeismicConfig']['current_file_path'] = self.seismic_config.current_file_path

        # 写回配置文件
        self.write_config(file_path, existing_config)

    def write_config(self, file_path, config_data):
        # 写回配置文件
        config = configparser.ConfigParser()
        for section, options in config_data.items():
            config[section] = options
        with open(file_path, 'w') as configfile:
            config.write(configfile)


    def Setting_Theme(self,bg,fg):
        # 设置主界面的背景颜色
        self.root.configure(bg=bg)
        # 设置主界面的前景颜色
        self.root.option_add('*foreground', fg)
        ## 设置PowerShell Label颜色
        #self.title_Label.configure(bg=fg, fg=bg)
        #self.text_widget.configure(bg=fg, fg=bg)
        return None


    def write(self, text):
        # 重定向 sys.stdout 的 write 方法
        self.text_widget.insert(tk.END, text)
        self.text_widget.event_generate("<<Modified>>")

    def process_input(self,event):
        # 获取 Text 中的文本
        text_content = self.text_widget.get("1.0", tk.END)

        # 分割成行
        lines = text_content.split('\n')

        # 获取最后一行内容
        last_line = lines[-2].strip()  # 最后一行是空行，所以取倒数第二行

        if last_line.lower() == "clc":
            # 如果最后一行是"clc"，清空文本框
            self.text_widget.delete("1.0", tk.END)
        #elif last_line.startswith("INPUT:"):
        #    input_val = PlotSeismic.UserInput()
        #    # 否则，在文本框中显示处理后的结果
        #    # 如果最后一行以 "INPUT:" 开头，执行相应操作
        #    input_value = last_line[6:]  # 获取输入值，去掉 "INPUT:" 部分
        #    # 在这里添加你希望执行的操作，例如将 input_value 传递给其他函数
        #    input_val.get_input(user_input=input_value)
        #    #print("OK!")
        #else:
            #print(f"{text_content}")
        return None

    def highlight_input(self, event=None):
        # 获取 Text 中的全部文本
        text_content = self.text_widget.get("1.0", tk.END)
        # 清除之前的标签
        self.text_widget.tag_remove("input_tag", "1.0", tk.END)

        # 使用函数高亮关键词
        self.highlight_text("INPUT:", "input_tag")
        self.highlight_text("Stream:", "stream_tag")
        self.highlight_text("Trace(s) ", "Trace_tag")
        self.highlight_text("Config Info:", "Config_tag")
        self.highlight_text("End of Config Info.", "Config_tag")
        self.highlight_text("Show Streams List ","streamlist_tag")
        self.highlight_text("________________________________ ","______tag")
        
        
        # 重置光标位置
        self.text_widget.mark_set(tk.CURRENT, "1.0")

    def highlight_text(self, keyword, tag_name):
        index = "1.0"
        while index:
            index = self.text_widget.search(keyword, index, stopindex=tk.END)
            if index:
                end_index = f"{index}+{len(keyword)}c-1c"
                # 将找到的文本范围应用样式标签
                self.text_widget.tag_add(tag_name, index, end_index)
                index = end_index

    def open_settings(self):
        # 打开设置窗口
        # settings_window = SettingsWindow(self)
        pass
#------------------------INITIALIZATION----------------------------------------------------------------------

#----------------菜单功能-------------------------------------------------------------------------------------
#-------------------File-------------------------------------------------------------------------------------
    def open_file_dialog(self):
        # 打开文件对话框以选择文件
        self.data_file_path = filedialog.askopenfilename(title="Select a Seismic Data File")
        self.seismic_processor.data_file_path = self.data_file_path
        # 检查用户是否选择了文件
        if self.data_file_path:
            print(f"Selected file: {self.data_file_path}")
            self.file_name = os.path.basename(self.data_file_path)
            print("---Start Process---")
            print("Read File:", self.file_name)
            print("Wait for Data Loading...")
            self.seismic_processor.stream = self.seismic_processor.read_seismic_data()
            print("read file succesfully...")
            # # 调用处理和绘图函数
            # root.after(50, self.process_and_plot_local_seismic_data())
            
    def process_and_plot_local_seismic_data(self):
        # 使用 self.data_file_path 进行处理和绘图
        if hasattr(self, 'data_file_path') and self.data_file_path:
            
            #seismic_processor.print_traces()
            self.seismic_processor.plot_selected_traces()
            self.seismic_processor.plot_seismic_data()
        else:
            print("No file selected. Please open a seismic data file first.")

    def load_config(self):
        self.config_file_path = filedialog.askopenfilename(title="Select Config File", filetypes=[("Config", "*.ini")])
        if self.config_file_path:
            self.read_config_file()

    def read_config_file(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)

        if 'SeismicConfig' in self.config:
            self.seismic_config.start_time_global = self.config.get('SeismicConfig', 'start_time_global')
            self.seismic_config.end_time_global = self.config.get('SeismicConfig', 'end_time_global')
            self.seismic_config.network = self.config.get('SeismicConfig', 'network')
            self.seismic_config.station = self.config.get('SeismicConfig', 'station')
            self.seismic_config.channel = self.config.get('SeismicConfig', 'channel')
            self.seismic_config.location = self.config.get('SeismicConfig', 'location')
        
            # Convert pre_filt string to a list of float values
            pre_filt_str = self.config.get('SeismicConfig', 'pre_filt')
            pre_filt_str = pre_filt_str.strip('[]')  # 去掉开头和结尾的方括号
            self.seismic_config.pre_filt = [float(value) for value in pre_filt_str.split(',')]
        
            self.seismic_config.cache_stream = self.config.get('SeismicConfig', 'cache_stream')
            self.seismic_config.cache_stations = self.config.get('SeismicConfig', 'cache_stations')
            self.seismic_config.save_folder = self.config.get('SeismicConfig', 'save_folder')
            self.seismic_config.CSL_info = self.config.get('SeismicConfig', 'CSL_info')
            self.seismic_config.CSL_stream = self.config.get('SeismicConfig', 'CSL_stream')
            # Handle cache_streams_list as a literal_eval
            cache_streams_list_str = self.config.get('SeismicConfig', 'cache_streams_list')
            try:
                self.seismic_config.cache_streams_list = ast.literal_eval(cache_streams_list_str)
            except (SyntaxError, ValueError):
                print("Error parsing cache_streams_list")

            self.seismic_config.current_file_path = os.path.dirname(self.config_file_path)

        print("Config Info:")
        seismic_config_vars = vars(self.seismic_config)
        for key, value in seismic_config_vars.items():
            print(f"{key} = {value}")

        print("End of Config Info.")

    def update_config_file(self):
        if self.config_file_path:
            self.config = configparser.ConfigParser()
            self.config['SeismicConfig'] = {
                'start_time_global': self.seismic_config.start_time_global,
                'end_time_global': self.seismic_config.end_time_global,
                'network': self.seismic_config.network,
                'station': self.seismic_config.station,
                'channel': self.seismic_config.channel,
                'location': self.seismic_config.location,
                'pre_filt': self.seismic_config.pre_filt,
                'cache_stream': self.seismic_config.cache_stream,
                'cache_stations': self.seismic_config.cache_stations,
                'save_folder': self.seismic_config.save_folder,
                'CSL_info': self.seismic_config.CSL_info,
                'CSL_stream': self.seismic_config.CSL_stream,
                'cache_streams_list': repr(self.seismic_config.cache_streams_list),  # Save as a string representation
                'current_file_path ':self.config_file_path
            }

            with open(self.config_file_path, 'w') as config_file:
                self.config.write(config_file)
    def save_data(self):
        self.seismic_processor.save_data()
    def save_data_to_txt(self):
        self.seismic_processor.save_data_to_TXT()
    def save_data_to_sac(self):
        self.seismic_processor.save_data_to_SAC()
#-------------------File-------------------------------------------------------------------------------------


#------------------Data--Processing--------------------------------------------------------------------------

    def input_date(self):
        self.seismic_config = self.seismic_processor.main()

    def raw_seismogram(self):
        # Call the raw_seismogram function from PlotSeismic
        self.seismic_processor.raw_seismogram()

    def interpolate(self,method):
        limit=None
        for trace in self.seismic_processor.stream:
            tr = self.seismic_processor.interpolation(trace,method=method,interpolation_limit=limit)
        self.seismic_config.CSL_info = f"after {method} interpolation"
        self.seismic_config.CSL_stream = self.seismic_processor.stream.copy()
        self.seismic_processor.stream.plot()
 
    def move_response(self):
        # Call the move_response function from PlotSeismic
        #plt.close()
        self.seismic_processor.stream = self.seismic_processor.move_response()
    def windowing(self):
        # Call the windowing function from PlotSeismic
        self.seismic_processor.windowing()

    def de_pulse_and_plot(self):
        # Define the logic for moving pulses or outliers and plotting
        self.seismic_processor.de_pulse()

    def plot_amplitude_spectrum(self):
        # Plot amplitude spectrum of Seismic data
        length_CSL = len(self.seismic_config.cache_streams_list)
        if length_CSL < 2:
            input_stream = self.seismic_processor.stream
        elif length_CSL > 1:
            input_stream = self.seismic_config.cache_streams_list[length_CSL-1][1]

        self.seismic_processor.plot_amplitude_spectrum(input_stream, self.seismic_config.pre_filt[3])

    def plot_amplitude_spectrum_s(self):
        # Plot amplitude spectrum of Some Seismic data
        # 假设self.seismic_config.cache_streams_list的结构为[["info1", stream1], ["info2", stream2]]
        length_CSL = len(self.seismic_config.cache_streams_list)
        if length_CSL < 2:
            self.plot_amplitude_spectrum()
            return
        streams = [item[1] for item in self.seismic_config.cache_streams_list]
        self.seismic_processor.plot_amplitude_spectrum_s(streams, self.seismic_config.pre_filt[3])
    
    def seismic_data_correlation(self):
        # prepare to seismic data correlation
        if len(self.seismic_config.cache_streams_list) == 2:
            self.show_streams_list()
            # for i, (info, stream) in enumerate(self.seismic_config.cache_streams_list, 1):
            #     print(f"Index: {i}")
            #     print(f"Info: {info}")
            #     print(f"Network: {stream[0].stats.network}")
            #     print(f"Station: {stream[0].stats.station}")
            #     print(f"Channel: {stream[0].stats.channel}")
            #     print(f"Start Time: {stream[0].stats.starttime}")
            #     print(f"End Time: {stream[0].stats.endtime}")
            #     print(f"Number of Traces: {len(stream)}\n")
            list = [self.seismic_config.cache_streams_list[i-1][1] for i in [1,2]]
            
        elif len(self.seismic_config.cache_streams_list) > 2:
            self.show_streams_list()
            # for i, (info, stream) in enumerate(self.seismic_config.cache_streams_list, 1):
            #     print(f"Index: {i}")
            #     print(f"Info: {info}")
            #     print(f"Network: {stream[0].stats.network}")
            #     print(f"Station: {stream[0].stats.station}")
            #     print(f"Channel: {stream[0].stats.channel}")
            #     print(f"Start Time: {stream[0].stats.starttime}")
            #     print(f"End Time: {stream[0].stats.endtime}")
            #     print(f"Number of Traces: {len(stream)}\n")

            serial_numbers_input = self.seismic_processor.get_user_input("Please select the serial number of the cross-correlated seismic data(eg:'1 2') ")
            serial_numbers = [int(number) for number in serial_numbers_input.split()]
            list = [self.seismic_config.cache_streams_list[i-1][1] for i in serial_numbers]
        else:
            print(" Cache_streams_list is not enough! Please get more stream! ")
        
        self.sub_seismic_data_correlation(list)

    def sub_seismic_data_correlation(self,stream_list):
        # seismic data correlation
        print("stream: list\n",stream_list)
        self.seismic_processor.seismic_data_correlation(stream_list)

    
    def seismic_data_sparsification(self):
        # 地震数据稀疏化处理
        self.seismic_processor.seismic_data_sparsification()
        

    def plot_current_stream(self):
        #plot_stream =self.seismic_processor.stream
        #print(plot_stream)
        #get_trace = self.seismic_processor.get_user_input("Enter the index of the trace you want to plot:","1")
        self.seismic_processor.stream.plot()

    def plot_current_stream_2(self):
        #plot_stream =self.seismic_processor.stream
        #print(plot_stream)
        #get_trace = self.seismic_processor.get_user_input("Enter the index of the trace you want to plot:","1")
        self.seismic_processor.plot_seismic_waveforms()
#------------------Data--Processing--------------------------------------------------------------------------
#----------------------Setting-------------------------------------------------------------------------------

    def modify_configurations(self):
        # Get the attributes of the SeismicConfig object
        param_names = list(vars(self.seismic_config).keys())

        # Preserve the original format of values
        param_values = [str(value) if param_name not in ['cache_streams_list','pre_filt'] else value for param_name, value in vars(self.seismic_config).items()]
        total_params = len(param_names)
        result = self.seismic_processor.get_user_inputs(total_params, param_names, param_values)

        # Update the SeismicConfig object and sync with self.config
        if result is not None:
            for param_name, new_value in zip(param_names, result):
                setattr(self.seismic_config, param_name, new_value)
                ## Sync with self.config
                self.config['SeismicConfig'][param_name] = str(new_value)

            with open(self.config_file_path, 'w') as config_file:
                self.config.write(config_file)

    def show_streams_list(self):
        print("Show Streams List :")
        print("________________________________ \n")
        for i, (info, stream) in enumerate(self.seismic_config.cache_streams_list, 1):
            print(f"Index: {i}")
            print(f"Info: {info}")
            print(f"Network: {stream[0].stats.network}")
            print(f"Station: {stream[0].stats.station}")
            print(f"Channel: {stream[0].stats.channel}")
            print(f"Start Time: {stream[0].stats.starttime}")
            print(f"End Time: {stream[0].stats.endtime}")
            print(f"Number of Traces: {len(stream)}\n")
        if len(self.seismic_config.cache_streams_list) == 0:
            print("—————— No Stream Data——————")
        print("________________________________ ")
    def merge_streams_list(self):
        # 创建一个空的 merge_stream 对象
        merge_stream = Stream()

        # 遍历 stream_list 中的每个 stream
        for info, stream in self.seismic_config.cache_streams_list:
            # 将当前 stream 中的所有 trace 添加到 merge_stream 中
            for trace in stream:
                merge_stream.append(trace)

        # 将 merge_stream 添加到 stream_list 中
        self.seismic_config.CSL_info   = "merge streams of over"
        self.seismic_config.CSL_stream = merge_stream.copy()
        self.save_CSL()
        self.seismic_processor.stream = merge_stream
        
    def save_CSL(self):
        new_CSL = [self.seismic_config.CSL_info,self.seismic_config.CSL_stream]
        print("Save new sublist:\n",new_CSL)
        self.seismic_config.cache_streams_list.append(new_CSL)
        print("Cache_Streams_List have ",len(self.seismic_config.cache_streams_list)," element(s) \n")

    def clear_CSL(self,flag):
        if flag !='all'and flag != 'Customized':
            self.seismic_config.cache_streams_list = self.seismic_config.cache_streams_list[:-int(flag)]
        elif flag =='all':
            self.seismic_config.cache_streams_list.clear()
        elif flag =="Customized":
            self.show_streams_list()
            custom_indices_str = self.seismic_processor.get_user_input("Input the Index to Cut\n"
                                                                        "e.g. 1,3,5 or 1-10 or 1-4,5,6")
            if custom_indices_str:
                custom_indices_list = custom_indices_str.split(',')
                indices_to_delete = set()
                for idx_str in custom_indices_list:
                    if '-' in idx_str:
                        start, end = map(int, idx_str.split('-'))
                        # 生成范围时包括终止值
                        indices_to_delete.update(range(start - 1, end))
                    else:
                        indices_to_delete.add(int(idx_str) - 1)
                # 自检功能，确保索引在列表长度范围内
                max_index = len(self.seismic_config.cache_streams_list) - 1
                invalid_indices = [idx for idx in indices_to_delete if not (0 <= idx <= max_index)]
                if invalid_indices:
                    print(f"Invalid indices detected: {invalid_indices}. They will be ignored.")
                    # 从要删除的索引集合中移除无效索引
                    indices_to_delete = indices_to_delete - set(invalid_indices)
                
                # 从最大索引开始删除，以避免索引调整带来的问题
                for idx in sorted(indices_to_delete, reverse=True):
                    if 0 <= idx < len(self.seismic_config.cache_streams_list):
                        del self.seismic_config.cache_streams_list[idx]
                    else:
                        print(f"Index {idx + 1} is out of range and will be ignored.")
            else:
                print("No custom indices provided. No action taken.")
        if len(self.seismic_config.cache_streams_list) ==1:
            self.seismic_processor.stream = self.seismic_config.cache_streams_list[0][1]
        elif len(self.seismic_config.cache_streams_list) ==0:
            self.seismic_processor.stream = Stream()
        else:
            self.seismic_processor.stream = self.seismic_config.cache_streams_list[-1][1]
        print("Cache_Streams_List have ",len(self.seismic_config.cache_streams_list)," element(s) \n")
#----------------------Setting-------------------------------------------------------------------------------

#----------------------SSH Connect---------------------------------------------------------------------------
    def ssh_execute_command(self):
        """
        ## 建立 SSH 连接并执行远程命令
        :param hostname: 主机名或 IP 地址
        :param port: 端口号
        :param username: 用户名
        :param password: 密码
        :param command: 要执行的远程命令
        :return: 命令执行结果（输出、错误）
        """
        

        param_names  = ['hostname', 'port', 'username', 'password', 'command']
        param_values = ['172.24.39.4',12306,'Yusongping','cug123456a!','ls']
        
        result = self.seismic_processor.get_user_inputs(5, param_names, param_values)
        [self.hostname, self.port, self.username, self.password, self.command] =result

        self.connect_ssh()

    def connect_ssh(self):
        """
        建立 SSH 连接
        """
        if not all([self.hostname, self.port, self.username, self.password]):
            print("SSH connection parameters are missing.")
            self.ssh_execute_command()

        # 创建 SSH 客户端对象
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        
        # 连接到远程服务器
        self.ssh.connect(hostname=self.hostname, port=self.port, 
                         username=self.username, password=self.password)

        # 执行远程命令
        stdin, stdout, stderr = self.ssh.exec_command(self.command)

        # 读取命令输出
        output = stdout.read().decode()
        error = stderr.read().decode()

        # 打印输出和错误
        print('Output:\n',output)
        print('Error:\n',error)

        return output, error
    
    def input_ssh_command(self):
        self.command = self.seismic_processor.get_user_input("input command:","ls")

        if self.ssh:
            # 执行远程命令
            stdin, stdout, stderr = self.ssh.exec_command(self.command)

            # 读取命令输出
            output = stdout.read().decode()
            error = stderr.read().decode()

            # 打印输出和错误
            print('Output:\n',output)
            print('Error:\n',error)
        else:
            self.connect_ssh()

    def close_ssh(self):
        """
        关闭 SSH 连接
        """
        if self.ssh:
            self.ssh.close()
            print("SSH connection has been closed.")
        else:
            print("No SSH connection exists.")
#----------------------SSH Connect---------------------------------------------------------------------------

#----------------------Help----------------------------------------------------------------------------------

    def open_de_pulse_help(self):
        # 指定帮助文档的路径
        help_document_path = os.path.join(self.seismic_config.current_file_path,"/help/de_pulse_help_document.html")

        try:
            # 使用 webbrowser 打开帮助文档
            webbrowser.open(help_document_path, new=2)
        except Exception as e:
            print(f"Error opening or reading the help document: {e}")

    def show_interpolation_help(self):
        help_message = """
        Interpolation Help / 插值帮助:

        1. Linear / 线性: Linear interpolation using straight lines between data points. 线性插值，使用数据点之间的直线。
        2. Nearest / 最近邻: Nearest-neighbor interpolation using the value of the closest data point. 最近邻插值，使用最近邻数据点的值。
        3. Polynomial / 多项式: Polynomial interpolation using curve fitting with polynomials. 多项式插值，使用多项式进行曲线拟合。
        4. Spline / 样条: Spline interpolation using piecewise continuous low-degree polynomials. 样条插值，使用分段连续的低次多项式。
        5. Barycentric / 重心: Barycentric interpolation using barycentric coordinates. 重心插值，使用重心坐标。
        6. Krogh / Krogh方法: Krogh method, based on cumulative cubic spline interpolation. Krogh方法，基于累积三次样条插值。
        7. Piecewise Polynomial / 分段多项式: Piecewise polynomial interpolation for local approximation. 分段多项式插值，用于局部逼近。
        8. PCHIP / PCHIP方法: Piecewise Cubic Hermite Interpolating Polynomial for monotonic data. PCHIP方法，用于单调数据的分段三次 Hermite 插值。
        9. Akima / Akima方法: Akima spline interpolation for irregularly spaced data. Akima样条插值，用于不规则间隔的数据。
        10. Cubic Spline / 三次样条: Cubic spline interpolation ensuring smoothness and continuity. 三次样条插值，确保平滑和连续性。

        Linear（线性）：
        作用： 快速、简单，适用于数据变化相对缓慢的情况。
        优缺点： 简单易懂，但可能无法准确地捕捉数据中的复杂变化。
        Nearest（最近邻）：
        作用： 用最近邻数据点的值进行插值，适用于离散的数据点。
        优缺点： 简单，但可能产生不平滑的结果，尤其是在数据点之间有较大间隔的情况下。
        Polynomial（多项式）：
        作用： 通过拟合多项式来逼近数据的变化。
        优缺点： 可能对噪声敏感，高次多项式可能导致过拟合，出现震荡。
        Spline（样条）：
        作用： 使用分段低次多项式连接数据点，实现平滑插值。
        优缺点： 提供平滑的结果，适用于大多数情况。但在数据点较少时可能过拟合。
        Barycentric（重心）：
        作用： 使用重心坐标进行插值，适用于非均匀分布的数据点。
        优缺点： 对于非均匀分布的数据点效果较好，但在均匀分布的情况下可能与其他方法相似。
        Krogh（Krogh方法）：
        作用： 基于累积三次样条插值，适用于需要平滑曲线的情况。
        优缺点： 提供平滑结果，但可能对异常值敏感。
        Piecewise Polynomial（分段多项式）：
        作用： 使用分段多项式连接数据点，适用于有局部变化的情况。
        优缺点： 提供局部逼近，但可能产生连接点处的不连续性。
        PCHIP（PCHIP方法）：
        作用： 使用分段三次 Hermite 插值，保持数据的单调性。
        优缺点： 适用于需要保持单调性的情况，但可能对噪声敏感。
        Akima（Akima方法）：
        作用： 使用Akima样条插值，适用于非常不规则的数据。
        优缺点： 对于不规则数据的插值效果较好，但可能对异常值敏感。
        Cubic Spline（三次样条）：
        作用： 使用三次多项式进行插值，确保平滑和连续的一阶、二阶导数。
        优缺点： 提供平滑结果，适用于大多数情况，但在极端情况下可能引入振荡。
        总体而言，选择插值方法应该根据具体情况。例如，对于平滑曲线的需求，可以选择Spline或Cubic Spline；对于非均匀分布的数据，Barycentric可能更合适。在使用插值方法时，始终要注意数据的性质以及对结果的期望，以便选择最合适的插值方法

        Usage Example / 使用示例:
        --------------
        To perform linear interpolation, use: / 执行线性插值，使用：
        >>> Data Processing --> Interpolation --> Linear

        Choose the appropriate method based on your data characteristics and interpolation requirements.
        根据您的数据特性和插值需求选择合适的方法。
        """
        # 使用Tkinter的messagebox来创建小弹窗
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 显示帮助信息的弹窗
        messagebox.showinfo("Interpolation Help / 插值帮助", help_message)

    #----------------------About-----------------------------------------------------------------------
    def about(self):
        # 创建消息框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 定义关于信息
        about_message = "SeisY - Seismic Data Processing Software\n"
        about_message += "Version: 1.0.0\n"
        about_message += "Developed by: [余松平(Yu Songping) , China University of Geosciences]\n"
        about_message += "Description: SeisY is a seismic data processing software designed for analyzing and visualizing seismic data.\n"
        about_message += "Features:\n"
        about_message += "- Load seismic data from various formats\n"
        about_message += "- Process seismic data for analysis\n"
        about_message += "- Visualize seismic data using interactive plots\n"
        about_message += "- Perform basic seismic data manipulation and filtering\n"
        about_message += "- Export processed data to common formats\n"
        about_message += "For more information, visit [https://ysp0github.github.io/YSZJ/ or contact email:[ysp@cug.edu.cn or york.stephen19@gmail.com]]."

        # 显示关于信息的弹窗
        messagebox.showinfo("About SeisY", about_message)
    #----------------------About-----------------------------------------------------------------------

#----------------------Help----------------------------------------------------------------------------------


if __name__ == '__main__':
    root = tk.Tk()
    example = SubMenuExample(root)
    root.mainloop()